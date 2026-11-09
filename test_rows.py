"""
Row-sharded tables (micro_engram_sharding: rows) against the plain engram, with real
processes on CPU (gloo; on GPUs training uses NCCL):  python test_rows.py

3 ranks and a 31-row table, so the split is uneven (11 / 11 / 9 rows).
"""
import copy
import json
import os
import socket
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from micro_engram.module import MicroEngram  # noqa: E402
from micro_engram.optim import LazyAdamW  # noqa: E402
from micro_engram.rows import ROWS_DIR, ROWS_OPTIM_DIR, RowTables, row_range  # noqa: E402
from micro_engram.weights import load_engram_weights  # noqa: E402

W, N, D, E = 3, 31, 16, 4
PREFIX = "model.layers.3.micro_engram."


def make_full():
    torch.manual_seed(0)
    eng = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=50, gate="scalar_sqrt")
    with torch.no_grad():
        eng.value.weight.normal_(std=0.3)  # so the output (and the table gradients) depend on the tables
        eng.q_scale.fill_(0.25)
    return eng


def sharded_copy(full, rank, world):
    e = copy.deepcopy(full)
    e.shard_rows(rank, world)
    return e


def batch(rank, step=0):
    g = torch.Generator().manual_seed(100 + 10 * rank + step)
    ids = torch.randint(5, 50, (2, 9), generator=g)
    ids[0, :3] = 7  # duplicates within a batch, and the same n-grams on every rank
    return torch.randn(2, 9, D, generator=g) * 5, ids


def avg_table_grads(full, world):
    """What FSDP gives the plain engram: every rank's gradient, averaged."""
    out = []
    for t in full.tables:
        g = t.weight.grad.clone()
        dist.all_reduce(g)
        out.append(g / world)
    return out


def worker(rank, world, port, tmp):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    res = []

    def check(name, ok):
        res.append((name, bool(ok)))

    start, count = row_range(N, rank, world)
    full = make_full()
    sh = sharded_copy(full, rank, world)
    check("rows split 11/11/9; tables gone from the module and its state dict",
          count == (9 if rank == 2 else 11) and "tables" not in sh._modules
          and not any(k.startswith("tables.") for k in sh.state_dict()) and len(sh.row_shards) == 4)

    # 1. forward: identical output (each rank its own batch)
    h, ids = batch(rank)
    out_f, out_s = full(h, ids), sh(h, ids)
    check("forward: same output as the full tables", torch.equal(out_f, out_s))

    # 2. backward: the owner's gradient == FSDP's averaged full-table gradient, rows sliced
    out_f.pow(2).sum().backward()
    out_s.pow(2).sum().backward()
    want = avg_table_grads(full, world)
    check("backward: shard gradients == averaged full gradients (sliced to this rank's rows)",
          all(torch.allclose(p.grad, w[start:start + count], atol=1e-6) for p, w in zip(sh.row_shards, want)))
    check("backward: key/value/scale gradients unchanged",
          torch.allclose(full.key.weight.grad, sh.key.weight.grad, atol=1e-6)
          and torch.allclose(full.q_scale.grad, sh.q_scale.grad, atol=1e-6))
    ref_grads = [p.grad.clone() for p in sh.row_shards]

    # 2b. mixed table sizes per n-gram order, with 4-grams: offsets and splits per table
    torch.manual_seed(3)
    fm = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=50, gate="scalar_sqrt",
                     table_sizes={2: 31, 3: 47, 4: 13})
    with torch.no_grad():
        fm.value.weight.normal_(std=0.3)
    sm = sharded_copy(fm, rank, world)
    hm, im = batch(rank)
    om_f, om_s = fm(hm, im), sm(hm, im)
    check("mixed sizes (31/47/13, 2-4-grams): same output as the full tables", torch.equal(om_f, om_s))
    om_f.pow(2).sum().backward()
    om_s.pow(2).sum().backward()
    want_m = avg_table_grads(fm, world)
    starts_m = sm.row_info[2]
    check("mixed sizes: every table's gradient == averaged full gradient, sliced per table",
          all(torch.allclose(p.grad, w[a:a + p.shape[0]], atol=1e-6) for p, w, a in zip(sm.row_shards, want_m, starts_m))
          and [p.shape[0] for p in sm.row_shards] == [row_range(r, rank, world)[1] for r in fm.table_rows])
    mdir = os.path.join(tmp, "mixed", ROWS_DIR)
    RowTables([(PREFIX, sm)]).save(mdir)
    dist.barrier()
    sdm = load_engram_weights(mdir, layer=3)
    fresh_m = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=50, gate="scalar_sqrt",
                          table_sizes={2: 31, 3: 47, 4: 13}, build_tables=False)
    fresh_m.shard_rows(rank, world)
    RowTables([(PREFIX, fresh_m)]).load(mdir)
    check("mixed sizes: saved as full tables of their own sizes, loaded back into each rank's rows",
          [sdm[f"tables.{k}.weight"].shape[0] for k in range(6)] == fm.table_rows
          and all(torch.equal(a, b) for a, b in zip(fresh_m.row_shards, sm.row_shards)))
    wrong = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=50, gate="scalar_sqrt",
                        table_sizes={2: 31, 3: 31, 4: 13}, build_tables=False)
    wrong.shard_rows(rank, world)
    try:
        RowTables([(PREFIX, wrong)]).load(mdir)
        refused = False
    except ValueError as e:
        refused = "can't resume" in str(e)
    check("resuming into different table sizes is refused with a clear message", refused)

    # 3. activation checkpointing (the forward runs again inside the backward, with its exchange)
    from torch.utils.checkpoint import checkpoint

    sc = sharded_copy(full, rank, world)
    checkpoint(lambda x: sc(x, ids), h, use_reentrant=False).pow(2).sum().backward()
    check("activation checkpointing: same gradients",
          all(torch.allclose(p.grad, r, atol=1e-6) for p, r in zip(sc.row_shards, ref_grads)))

    # 4. clipping: the norm of the whole tables' gradient, over all ranks
    rt = RowTables([(PREFIX, sh)])
    full_norm = torch.sqrt(sum(w.pow(2).sum() for w in want))
    norm = rt.clip(0)
    check(f"clip: global table grad norm ({norm:.5f}) == full tables' ({float(full_norm):.5f})",
          abs(norm - float(full_norm)) < 1e-5)
    rt.clip(0.5 * norm)
    check("clip: scaled to max_grad_norm", abs(rt.clip(0) - 0.5 * norm) < 1e-5)

    # 5. training: 3 steps of (lazy) AdamW == the full tables with averaged gradients
    for lazy in (False, True):
        f2, s2 = make_full(), None
        for p in f2.parameters():
            p.requires_grad_(False)
        for t in f2.tables:
            t.weight.requires_grad_(True)
        s2 = sharded_copy(f2, rank, world)
        opt_f = LazyAdamW([{"params": [t.weight for t in f2.tables], "lazy": lazy}], lr=0.05, weight_decay=0.1)
        rt2 = RowTables([(PREFIX, s2)], lr=0.05, weight_decay=0.1, lazy=lazy)
        for step in range(3):
            hh, ii = batch(rank, step)
            f2(hh, ii).pow(2).sum().backward()
            for t, g in zip(f2.tables, avg_table_grads(f2, world)):
                t.weight.grad = g
            opt_f.step()
            opt_f.zero_grad()
            s2(hh, ii).pow(2).sum().backward()
            rt2.step(0.05)
        check(f"3 steps of {'lazy ' if lazy else ''}AdamW: same tables as the full engram",
              all(torch.allclose(p, t.weight[start:start + count], atol=1e-5) for p, t in zip(s2.row_shards, f2.tables)))
        check(f"{'lazy' if lazy else 'AdamW'}: gradients cleared after the step", all(p.grad is None for p in s2.row_shards))

    # 5b. rowwise_adam: same as the full tables with the same optimizer
    f3 = make_full()
    for p in f3.parameters():
        p.requires_grad_(False)
    for t in f3.tables:
        t.weight.requires_grad_(True)
    s3 = sharded_copy(f3, rank, world)
    opt_f3 = LazyAdamW([{"params": [t.weight for t in f3.tables], "lazy": True, "rowwise": True}], lr=0.05)
    rt_rw = RowTables([(PREFIX, s3)], lr=0.05, lazy=True, rowwise=True)
    for step in range(3):
        hh, ii = batch(rank, step)
        f3(hh, ii).pow(2).sum().backward()
        for t, g in zip(f3.tables, avg_table_grads(f3, world)):
            t.weight.grad = g
        opt_f3.step()
        opt_f3.zero_grad()
        s3(hh, ii).pow(2).sum().backward()
        rt_rw.step(0.05)
    check("3 steps of rowwise_adam: same tables as the full engram",
          all(torch.allclose(p, t.weight[start:start + count], atol=1e-5) for p, t in zip(s3.row_shards, f3.tables)))
    rw_dir = os.path.join(tmp, "rw")
    rt_rw.save(os.path.join(rw_dir, ROWS_DIR), os.path.join(rw_dir, ROWS_OPTIM_DIR))

    # 6. save: one DCP folder with the complete engram, read by the normal loaders
    folder, ofolder = os.path.join(tmp, "ckpt", ROWS_DIR), os.path.join(tmp, "ckpt", ROWS_OPTIM_DIR)
    rt2.save(folder, ofolder)
    dist.barrier()
    if rank == 0:
        sd = load_engram_weights(folder, layer=3)
        check("saved folder: the loaders read full tables equal to the trained rows (all ranks)",
              all(sd[f"tables.{k}.weight"].shape == (N, E) for k in range(4)))
        torch.save({k: v for k, v in sd.items()}, os.path.join(tmp, "saved_full.pt"))
        check("saved folder: key/value, q_scale as the scale (1 + delta), mults and canon",
              torch.equal(sd["key.weight"], s2.key.weight) and torch.allclose(sd["q_scale"], torch.full((D,), 1.25))
              and "mults" in sd and "canon" in sd)
    dist.barrier()
    sd = torch.load(os.path.join(tmp, "saved_full.pt"))
    check("saved tables == every rank's rows",
          all(torch.equal(sd[f"tables.{k}.weight"][start:start + count], p) for k, p in enumerate(s2.row_shards)))

    # 7. load into a fresh engram (resume): tables and the optimizer state come back
    fresh = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=50,
                        gate="scalar_sqrt", build_tables=False)
    fresh.shard_rows(rank, world)
    check("fresh rows differ before loading", not torch.equal(fresh.row_shards[0], s2.row_shards[0]))
    rt3 = RowTables([(PREFIX, fresh)], lr=0.05, lazy=True)
    n, opt_ok = rt3.load(folder, ofolder)
    check("load: rows back", n == 4 and all(torch.equal(a, b) for a, b in zip(fresh.row_shards, s2.row_shards)))
    st2, st3 = rt2.optimizer.state[s2.row_shards[1]], rt3.optimizer.state[fresh.row_shards[1]]
    check("load: optimizer state back (exp_avg, exp_avg_sq, step)",
          opt_ok and torch.equal(st2["exp_avg"], st3["exp_avg"]) and torch.equal(st2["exp_avg_sq"], st3["exp_avg_sq"])
          and float(st3["step"]) == 3)

    # 7a. rowwise state round trip, and switching optimizers on resume (second moment converted)
    def fresh_rows():
        e = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=50,
                        gate="scalar_sqrt", build_tables=False)
        e.shard_rows(rank, world)
        return e

    e_rw = fresh_rows()
    rt_back = RowTables([(PREFIX, e_rw)], lazy=True, rowwise=True)
    rt_back.load(os.path.join(rw_dir, ROWS_DIR), os.path.join(rw_dir, ROWS_OPTIM_DIR))
    a_st, b_st = rt_rw.optimizer.state[s3.row_shards[2]], rt_back.optimizer.state[e_rw.row_shards[2]]
    check("rowwise: optimizer state saves and loads (per-row second moment)",
          torch.equal(a_st["exp_avg_sq"], b_st["exp_avg_sq"]) and b_st["exp_avg_sq"].dim() == 1
          and torch.equal(a_st["exp_avg"], b_st["exp_avg"]) and rt_back.converted == 0)
    e_cv = fresh_rows()
    rt_cv = RowTables([(PREFIX, e_cv)], lazy=True, rowwise=True)
    rt_cv.load(folder, ofolder)  # saved by lazy AdamW (per-value second moment)
    lz_st, cv_st = rt2.optimizer.state[s2.row_shards[1]], rt_cv.optimizer.state[e_cv.row_shards[1]]
    check("lazy_adam checkpoint -> rowwise_adam: second moment = the row mean (exact), momentum kept",
          rt_cv.converted == 4 and torch.allclose(cv_st["exp_avg_sq"], lz_st["exp_avg_sq"].mean(dim=1))
          and torch.equal(cv_st["exp_avg"], lz_st["exp_avg"]))
    e_bk = fresh_rows()
    rt_bk = RowTables([(PREFIX, e_bk)], lazy=True)
    rt_bk.load(os.path.join(rw_dir, ROWS_DIR), os.path.join(rw_dir, ROWS_OPTIM_DIR))
    bk_st = rt_bk.optimizer.state[e_bk.row_shards[2]]
    check("rowwise_adam checkpoint -> lazy_adam: every value gets its row's second moment",
          rt_bk.converted == 4 and bk_st["exp_avg_sq"].shape == e_bk.row_shards[2].shape
          and torch.equal(bk_st["exp_avg_sq"][:, 0], a_st["exp_avg_sq"]))

    # 7b. switching a run on: tables from an FSDP checkpoint of a run without row sharding
    import torch.distributed.checkpoint as dcp

    old_dir = os.path.join(tmp, "plain", "pytorch_model_fsdp_0")
    wrapped = "base_model.model.model.language_model.layers.3._checkpoint_wrapped_module.micro_engram."
    dcp.save({wrapped + k: v for k, v in full.state_dict().items()}, checkpoint_id=old_dir)
    sw = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=50,
                     gate="scalar_sqrt", build_tables=False)
    sw.shard_rows(rank, world)
    n_sw, opt_sw = RowTables([(PREFIX, sw)], lazy=True).load(old_dir, os.path.join(tmp, "plain", ROWS_OPTIM_DIR))
    check("resume from a run without row sharding: rows from pytorch_model_fsdp_0, Adam state starts over",
          n_sw == 4 and not opt_sw
          and all(torch.equal(p, t.weight[start:start + count]) for p, t in zip(sw.row_shards, full.tables)))

    # 8. FSDP2 around the engram (key/value/scales sharded by FSDP, tables by rows)
    from torch.distributed.fsdp import fully_shard

    ef = sharded_copy(full, rank, world)
    fully_shard(ef)
    of = ef(h, ids)
    check("FSDP2 + rows: same output", torch.allclose(of, out_f, atol=1e-6))
    of.pow(2).sum().backward()
    check("FSDP2 + rows: same table gradients",
          all(torch.allclose(p.grad, r, atol=1e-6) for p, r in zip(ef.row_shards, ref_grads)))
    ffolder = os.path.join(tmp, "fsdp", ROWS_DIR)
    RowTables([(PREFIX, ef)]).save(ffolder)
    dist.barrier()
    if rank == 0:
        sd = load_engram_weights(ffolder, layer=3)
        check("FSDP2 + rows: saved key.weight is the full matrix, q_scale the scale",
              torch.equal(sd["key.weight"], full.key.weight) and torch.allclose(sd["q_scale"], torch.full((D,), 1.25)))

    with open(os.path.join(tmp, f"rank{rank}.json"), "w") as f:
        json.dump(res, f)
    dist.barrier()
    dist.destroy_process_group()


def reshard(rank, world, port, tmp):
    """A checkpoint from 3 GPUs loaded on 2 (DCP reshards the rows)."""
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    e = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=50,
                    gate="scalar_sqrt", build_tables=False)
    e.shard_rows(rank, world)
    RowTables([(PREFIX, e)], lazy=True).load(os.path.join(tmp, "ckpt", ROWS_DIR), os.path.join(tmp, "ckpt", ROWS_OPTIM_DIR))
    sd = torch.load(os.path.join(tmp, "saved_full.pt"))
    start, count = row_range(N, rank, world)
    ok = all(torch.equal(sd[f"tables.{k}.weight"][start:start + count], p) for k, p in enumerate(e.row_shards))
    with open(os.path.join(tmp, f"reshard{rank}.json"), "w") as f:
        json.dump([["checkpoint of 3 GPUs resumes on 2 (rows resharded)", ok]], f)
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="engram_rows_")
    mp.spawn(worker, args=(W, free_port(), tmp), nprocs=W)
    mp.spawn(reshard, args=(2, free_port(), tmp), nprocs=2)
    merged = {}
    for f in sorted(os.listdir(tmp)):
        if f.endswith(".json"):
            for name, ok in json.load(open(os.path.join(tmp, f))):
                merged[name] = merged.get(name, True) and ok
    bad = 0
    for name, ok in merged.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        bad += not ok
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
    sys.exit(1 if bad or not merged else 0)

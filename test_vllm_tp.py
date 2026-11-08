"""
vLLM engram tables split across tensor-parallel GPUs, with real processes (gloo) and
minimal stand-ins for the vLLM classes model.py builds on:  python test_vllm_tp.py

3 ranks and a 31-row table: an uneven split (11 / 11 / 9 rows).
"""
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import types

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from micro_engram.module import MicroEngram  # noqa: E402
from micro_engram.rows import row_range  # noqa: E402

D, V, N, E, LAYER = 16, 50, 31, 4, 1


def make_full(dtype=torch.float32):
    torch.manual_seed(0)
    eng = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=V, gate="scalar_sqrt")
    with torch.no_grad():
        eng.value.weight.normal_(std=0.3)
        eng.q_scale.fill_(0.25)  # stored as delta: the scale is 1.25
    return eng.to(dtype)


def history(seed, n=24):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, V, (n, 3), generator=g)
    ids[:3, 1:] = -1  # start of a sequence: no earlier tokens
    return ids


def all_reduce(t):
    dist.all_reduce(t)
    return t


def install_vllm_stubs():
    """Just enough of vLLM for model.py: module paths, base classes, TP helpers."""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class Gemma4DecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(D, D)

        def forward(self, positions, hidden_states, residual, per_layer_input=None, **kw):
            return torch.tanh(self.lin(hidden_states)), residual

    class Gemma4ForCausalLM(nn.Module):
        def __init__(self, *, vllm_config, prefix=""):
            super().__init__()
            torch.manual_seed(5)
            self.config = types.SimpleNamespace(hidden_size=D, vocab_size=V, pad_token_id=0)
            self.model = nn.Module()
            self.model.layers = nn.ModuleList(Gemma4DecoderLayer() for _ in range(3))

        def load_weights(self, weights):
            params, out = dict(self.named_parameters()), set()
            for name, w in weights:
                if name not in params:
                    raise KeyError(f"vLLM's loader has no parameter {name}")
                with torch.no_grad():
                    params[name].copy_(w)
                out.add(name)
            return out

    mod("vllm")
    mod("vllm.config", VllmConfig=object)
    mod("vllm.logger", init_logger=logging.getLogger)
    mod("vllm.distributed",
        get_tensor_model_parallel_rank=lambda: dist.get_rank() if dist.is_initialized() else 0,
        get_tensor_model_parallel_world_size=lambda: dist.get_world_size() if dist.is_initialized() else 1,
        tensor_model_parallel_all_reduce=all_reduce)
    mod("vllm.model_executor")
    mod("vllm.model_executor.models")
    mod("vllm.model_executor.models.gemma4", Gemma4DecoderLayer=Gemma4DecoderLayer, Gemma4ForCausalLM=Gemma4ForCausalLM)
    mod("vllm.model_executor.models.gemma4_mm", Gemma4ForConditionalGeneration=nn.Module)
    mod("vllm.model_executor.models.utils", PPMissingLayer=type("PPMissingLayer", (nn.Module,), {}))


def vllm_config(conf=None, model_dir="/nonexistent"):
    return types.SimpleNamespace(
        additional_config={"micro_engram": conf} if conf else {},
        model_config=types.SimpleNamespace(architecture="Gemma4ForCausalLM", model=model_dir),
        use_v2_model_runner=True,
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=64))


def worker(rank, world, port, tmp):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    res = []

    def check(name, ok):
        res.append((f"[TP {world}] {name}", bool(ok)))

    from micro_engram_vllm.tp import load_rows, tp_lookup

    start, count = row_range(N, rank, world)

    # 1. lookup: masked local rows + all-reduce == the full tables, exactly (fp32 and bf16)
    for dtype in (torch.float32, torch.bfloat16):
        full = make_full(dtype)
        sh = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=V, gate="scalar_sqrt",
                         build_tables=False).to(dtype)
        sh.shard_rows(rank, world, dtype=dtype, requires_grad=False, empty=True)
        sh.row_lookup_fn = tp_lookup(all_reduce)
        load_rows(sh, full.state_dict(), "test")
        h = torch.randn(24, D).to(dtype) * 5
        idx = full.hash_prev(history(1))
        same = torch.equal(sh.inject(h, idx), full.inject(h, idx))
        check(f"{str(dtype)[6:]}: same output as the full tables (exact), this rank holds {count} of {N} rows",
              same and sh.row_shards[0].shape[0] == count)

    # 1b. mixed table sizes per n-gram order (with 4-grams): each table its own split
    torch.manual_seed(4)
    fm = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=V, gate="scalar_sqrt",
                     table_sizes={2: 31, 3: 47, 4: 13})
    with torch.no_grad():
        fm.value.weight.normal_(std=0.3)
    sm = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=V, gate="scalar_sqrt",
                     table_sizes={2: 31, 3: 47, 4: 13}, build_tables=False)
    sm.shard_rows(rank, world, requires_grad=False, empty=True)
    sm.row_lookup_fn = tp_lookup(all_reduce)
    load_rows(sm, fm.state_dict(), "mixed")
    hm = torch.randn(24, D) * 5
    g = torch.Generator().manual_seed(9)
    hist4 = torch.randint(5, V, (24, 4), generator=g)
    hist4[:2, 1:] = -1
    im = fm.hash_prev(hist4)
    check("mixed sizes (31/47/13, 2-4-grams): same output as the full tables, each table split on its own",
          torch.equal(sm.inject(hm, im), fm.inject(hm, im))
          and [p.shape[0] for p in sm.row_shards] == [row_range(r, rank, world)[1] for r in fm.table_rows])

    # 2. loading: slices, partial pieces, errors
    full = make_full()
    sd = full.state_dict()
    sh = MicroEngram(model_dim=D, table_size=N, engram_dim=E, num_heads=2, vocab_size=V, gate="scalar_sqrt",
                     build_tables=False)
    sh.shard_rows(rank, world, requires_grad=False, empty=True)
    first = {k: v for k, v in sd.items() if k in ("tables.0.weight", "tables.1.weight", "key.weight")}
    rest = {k: v for k, v in sd.items() if k not in first}
    got = load_rows(sh, first, "part 1", complete=False) | load_rows(sh, rest, "part 2", complete=False)
    check("weights in two pieces: every table's rows sliced right, the scale stored as delta",
          got == {0, 1, 2, 3}
          and all(torch.equal(p, full.tables[k].weight[start:start + count]) for k, p in enumerate(sh.row_shards))
          and torch.allclose(sh.q_scale, torch.full((D,), 0.25)))
    for bad, what in (({k: v for k, v in sd.items() if k != "tables.2.weight"}, "missing tables"),
                      ({**sd, "tables.0.weight": torch.zeros(N + 1, E)}, "is (32, 4)")):
        try:
            load_rows(sh, bad, "bad")
            ok = False
        except ValueError as e:
            ok = what in str(e)
        check(f"refused: {what}", ok)

    # 3. model.py with vLLM stand-ins: separate weights file, and engram inside the checkpoint
    install_vllm_stubs()
    from micro_engram_vllm.model import EngramGemma4ForCausalLM

    full = make_full()
    ids = history(2)
    h = torch.randn(24, D)
    for mode in ("file", "checkpoint"):
        conf = {"layer": LAYER, "table_size": N, "engram_dim": E, "num_heads": 2, "gate": "scalar_sqrt",
                "seed": 0, "orders": [2, 3], "pad_id": 0, "model_dim": D, "vocab_size": V}
        engram_sd = {f"model.layers.{LAYER}.micro_engram.{k}": v for k, v in full.state_dict().items()}
        if mode == "file":
            folder = os.path.join(tmp, "file")
            if rank == 0:
                os.makedirs(folder, exist_ok=True)
                torch.save(engram_sd, os.path.join(folder, "engram.pt"))
                json.dump({**conf, "weights": "engram.pt"}, open(os.path.join(folder, "micro_engram_config.json"), "w"))
            dist.barrier()
            model = EngramGemma4ForCausalLM(vllm_config=vllm_config(os.path.join(folder, "micro_engram_config.json")))
            base_sd = {k: v + 1 for k, v in model.state_dict().items() if ".micro_engram." not in k}
            # a checkpoint that carries an older engram of its own: ignored for the configured one
            stale = {f"model.layers.{LAYER}.micro_engram.tables.0.weight": torch.zeros(N, E)}
            loaded = model.load_weights(iter([*base_sd.items(), *stale.items()]))
        else:
            folder = os.path.join(tmp, "assembled")
            if rank == 0:
                os.makedirs(folder, exist_ok=True)
                json.dump(conf, open(os.path.join(folder, "micro_engram_config.json"), "w"))
            dist.barrier()
            model = EngramGemma4ForCausalLM(vllm_config=vllm_config(model_dir=folder))
            base_sd = {k: v + 1 for k, v in model.state_dict().items() if ".micro_engram." not in k}
            stream = list(base_sd.items()) + list(engram_sd.items())
            # like AutoWeightsLoader with vision weights between shards: two calls
            loaded = model.load_weights(iter(stream[:len(stream) // 2])) | model.load_weights(iter(stream[len(stream) // 2:]))
        layer = model.model.layers[LAYER]
        eng = layer.micro_engram
        check(f"{mode}: no table is a registered parameter (vLLM's loader never sees them), 1/TP of the rows",
              not any(".tables." in n for n, _ in model.named_parameters())
              and sum(p.numel() for p in eng.row_shards) == 4 * count * E)
        want_names = {f"model.layers.{LAYER}.micro_engram.{n}" for n in ("key.weight", "value.weight", "q_scale", "k_scale")}
        check(f"{mode}: the engram's registered weights are reported loaded (vLLM checks that)",
              want_names <= loaded and set(base_sd) <= loaded)
        layer.micro_engram_ids.copy_(torch.cat([ids, torch.full((64 - 24, 3), -1)])[:64].to(torch.int32))
        out, _ = layer(None, h, None)
        ref = full.inject(torch.tanh(layer.lin(h)), full.hash_prev(ids))
        check(f"{mode}: decoder layer output == base layer + the full-table engram", torch.allclose(out, ref, atol=1e-6))

    with open(os.path.join(tmp, f"res{world}_{rank}.json"), "w") as f:
        json.dump(res, f)
    dist.barrier()
    dist.destroy_process_group()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    tmp = tempfile.mkdtemp(prefix="vllm_tp_")
    for world in (3, 1):
        mp.spawn(worker, args=(world, free_port(), tmp), nprocs=world)
    merged = {}
    for f in sorted(os.listdir(tmp)):
        if f.startswith("res"):
            for name, ok in json.load(open(os.path.join(tmp, f))):
                merged[name] = merged.get(name, True) and ok
    shutil.rmtree(tmp, ignore_errors=True)
    bad = 0
    for name, ok in merged.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        bad += not ok
    sys.exit(1 if bad or not merged else 0)

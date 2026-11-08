"""Row-sharded engram tables (micro_engram_sharding: rows), TorchRec-style.

Without it, FSDP2 shards each table by rows too, but before every use it all-gathers
the whole table to every GPU (once in the forward, again in the backward under
activation checkpointing) and reduce-scatters a full-size gradient, although a batch
only touches a few thousand rows.

Here each GPU keeps only its own rows, outside FSDP. A lookup sends the wanted row
numbers to their owners and gets just those rows back (all_to_all); the backward sends
the rows' gradients back to the owners. Traffic is a few MB per micro-batch, and the
table's size only costs 1/world of its memory (plus its Adam state, also 1/world).

Gradients match FSDP's: the owner sums the contributions of every rank and divides by
the world size (FSDP averages across ranks). The tables have their own optimizer
(stepped right after the Trainer's), their own gradient clipping (same max_grad_norm,
over the tables only), and are saved as a DCP folder (micro_engram_rows) that holds the
complete engram under the usual names, so every loader reads it like any checkpoint.
"""
import os

import torch
import torch.distributed as dist

from .optim import LazyAdamW
from .weights import _local_name, layer_of_key

ROWS_DIR = "micro_engram_rows"          # complete engram (tables + the rest), DCP
ROWS_OPTIM_DIR = "micro_engram_rows_optim"  # the tables' optimizer state, DCP

_GROUP = {}


def dist_rank_world():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))


def row_range(n_rows, rank, world):
    """(first row, row count) owned by `rank`: ceil(n / world) rows each, the last ranks get
    the rest (the same split as DTensor's Shard(0), so checkpoints can be resharded)."""
    per = -(-n_rows // world)
    start = min(rank * per, n_rows)
    return start, max(0, min(per, n_rows - start))


def _group(world):
    """A process group of its own: the exchange never interleaves with FSDP's collectives
    on one communicator. Created on first use, which every rank reaches at the same point."""
    if world == 1:
        return None
    if "g" not in _GROUP:
        _GROUP["g"] = dist.new_group(list(range(world)))
    return _GROUP["g"]


def _a2a(x, out_rows, in_rows, group):
    if group is None:
        return x
    out = x.new_empty((sum(out_rows),) + tuple(x.shape[1:]))
    dist.all_to_all_single(out, x.contiguous(), out_rows, in_rows, group=group)
    return out


def _split_ids(g, offs):
    """Global ids (offset of the table + row) -> (table, row); offs = [0, end of table 0, ...]."""
    t = torch.searchsorted(offs, g, right=True) - 1
    return t, g - offs[t]


class _RowLookup(torch.autograd.Function):
    """flat global ids (offset of the table + row) -> their rows [M, dim], from the owners' shards.
    Tables can have different sizes (per n-gram order), so each has its own offset and split."""

    @staticmethod
    def forward(ctx, flat, info, *shards):
        offsets, pers, rank, world = info
        group = _group(world)
        offs = torch.tensor(offsets, dtype=torch.long, device=flat.device)
        per = torch.tensor(pers, dtype=torch.long, device=flat.device)
        uniq, inv = torch.unique(flat, return_inverse=True)  # each row is sent once
        t_u, r_u = _split_ids(uniq, offs)
        owner = r_u // per[t_u]
        order = torch.argsort(owner, stable=True)
        send_ids = uniq[order]
        send_counts = torch.bincount(owner, minlength=world)
        if group is not None:
            recv_counts = torch.empty_like(send_counts)
            dist.all_to_all_single(recv_counts, send_counts, group=group)
        else:
            recv_counts = send_counts
        sc, rc = send_counts.tolist(), recv_counts.tolist()
        req = _a2a(send_ids, rc, sc, group)  # rows the others want from me
        table, row = _split_ids(req, offs)
        local = row - rank * per[table]
        rows = shards[0].new_empty(req.numel(), shards[0].shape[1])
        for k, shard in enumerate(shards):
            sel = (table == k).nonzero().squeeze(1)
            if sel.numel():
                rows[sel] = shard.index_select(0, local[sel])
        got = _a2a(rows, sc, rc, group)
        emb = torch.empty_like(got)
        emb[order] = got
        ctx.save_for_backward(inv, order, table, local)
        ctx.sc, ctx.rc, ctx.info, ctx.n_uniq = sc, rc, info, uniq.numel()
        ctx.shapes = [(s.shape, s.dtype) for s in shards]
        return emb[inv]

    @staticmethod
    def backward(ctx, grad):
        inv, order, table, local = ctx.saved_tensors
        world = ctx.info[3]
        group = _group(world)
        # fp32 for the sums (duplicates within a batch, then across ranks)
        g_uniq = torch.zeros(ctx.n_uniq, grad.shape[1], dtype=torch.float32, device=grad.device)
        g_uniq.index_add_(0, inv, grad.float())
        g_req = _a2a(g_uniq[order], ctx.rc, ctx.sc, group)
        grads = []
        for k, (shape, dtype) in enumerate(ctx.shapes):
            if not ctx.needs_input_grad[2 + k]:
                grads.append(None)
                continue
            sel = (table == k).nonzero().squeeze(1)
            g = torch.zeros(shape, dtype=torch.float32, device=grad.device)
            g.index_add_(0, local[sel], g_req[sel])
            grads.append(g.div_(world).to(dtype))  # FSDP averages gradients across ranks
        return (None, None, *grads)


def row_lookup(engram, idx):
    """Per-table rows [*idx shape, dim] for the hashed ids `idx` (list, one per table)."""
    shards = engram.row_shards
    rank, world, _starts, pers = engram.row_info
    offsets = [0]
    for rows in engram.table_rows:
        offsets.append(offsets[-1] + rows)
    dev = shards[0].device
    flat = torch.cat([i.reshape(-1).to(dev) + offsets[k] for k, i in enumerate(idx)])
    out = _RowLookup.apply(flat, (offsets, pers, rank, world), *shards)
    m = idx[0].numel()
    return [out[k * m:(k + 1) * m].reshape(*idx[0].shape, -1) for k in range(len(shards))]


# ----------------------------------------------------------------------------- training side
def _dtensor(t, n_rows, world):
    """The local rows as a DTensor of the whole table (shares storage: DCP loads in place)."""
    from torch.distributed.device_mesh import DeviceMesh
    from torch.distributed.tensor import DTensor, Shard

    key = ("mesh", t.device.type)
    if key not in _GROUP:
        group = _group(world) if world > 1 else dist.group.WORLD
        _GROUP[key] = DeviceMesh.from_group(group, t.device.type)
    shape = (n_rows,) + tuple(t.shape[1:])
    stride = tuple(torch.empty(shape, device="meta").stride())
    return DTensor.from_local(t, _GROUP[key], [Shard(0)], run_check=False, shape=shape, stride=stride)


class RowTables:
    """The row-sharded tables of every engram of a run: optimizer, clipping, save/load.

    entries: [(prefix, engram)], prefix like "model.language_model.layers.0.micro_engram."
    (any name containing layers.<n>.micro_engram. works for the loaders).
    """

    def __init__(self, entries, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, lazy=False,
                 rowwise=False, stochastic_rounding=True):
        self.entries = entries
        self.world = dist_rank_world()[1]
        train = [p for _, e in entries for p in e.row_shards if p.requires_grad]
        self.optimizer = LazyAdamW([{"params": train, "lazy": lazy, "rowwise": rowwise}], lr=lr, betas=betas,
                                   eps=eps, weight_decay=weight_decay,
                                   stochastic_rounding=stochastic_rounding) if train else None
        self.last_grad_norm = None
        self.converted = 0  # tables whose second moment was converted on load (other optimizer)

    def params(self):
        return [p for _, e in self.entries for p in e.row_shards]

    def clip(self, max_norm):
        """Global norm of the table gradients (every row lives on exactly one rank); scales them
        down to max_norm if larger. Collective: every rank calls it."""
        ps = [p for p in self.params() if p.requires_grad]
        if not ps:
            return None
        sq = torch.zeros((), dtype=torch.float32, device=ps[0].device)
        for p in ps:
            if p.grad is not None:
                sq += p.grad.float().pow(2).sum()
        if self.world > 1:
            dist.all_reduce(sq, group=_group(self.world))
        norm = sq.sqrt()
        if max_norm and max_norm > 0:
            coef = (max_norm / (norm + 1e-6)).clamp(max=1.0)
            for p in ps:
                if p.grad is not None:
                    p.grad.mul_(coef.to(p.grad.dtype))
        self.last_grad_norm = float(norm)
        return self.last_grad_norm

    def step(self, lr):
        if self.optimizer is None:
            return
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    # --- checkpoints (collective: every rank calls these) ---
    def _table_items(self, frozen_too=False):
        for prefix, e in self.entries:
            if not frozen_too and not any(p.requires_grad for p in e.row_shards):
                continue
            for k, p in enumerate(e.row_shards):
                yield f"{prefix}tables.{k}.weight", e.table_rows[k], p

    def save(self, folder, optim_folder=None):
        """The complete engram (tables + key/value/scales/buffers, the usual names) to `folder`;
        the tables' optimizer state to `optim_folder`. Frozen engrams are skipped (their config
        points to the files they came from)."""
        import torch.distributed.checkpoint as dcp

        sd = {}
        for prefix, e in self.entries:
            if not any(p.requires_grad for p in e.row_shards):
                continue
            sd.update({prefix + k: v for k, v in e.state_dict().items()})
        for key, rows, p in self._table_items():
            sd[key] = _dtensor(p.detach(), rows, self.world)
        if sd:
            dcp.save(sd, checkpoint_id=folder)
        if optim_folder and self.optimizer is not None:
            osd = {}
            for key, rows, p in self._table_items():
                st = self.optimizer.state.get(p)
                if st:
                    osd[f"{key}.exp_avg"] = _dtensor(st["exp_avg"], rows, self.world)
                    osd[f"{key}.exp_avg_sq"] = _dtensor(st["exp_avg_sq"], rows, self.world)
                    osd[f"{key}.step"] = st["step"].detach().clone().reshape(())
            if osd:
                dcp.save(osd, checkpoint_id=optim_folder)

    def load(self, folder, optim_folder=None):
        """Tables from a DCP folder (a micro_engram_rows folder, or an FSDP checkpoint of a run
        without row sharding), resharded to this run's GPUs. Returns (tables loaded, whether
        optimizer state was loaded)."""
        import torch.distributed.checkpoint as dcp

        meta = dcp.FileSystemReader(folder).read_metadata().state_dict_metadata
        by_name = {}
        for k in meta:
            if (local := _local_name(k)) is not None and local.startswith("tables."):
                by_name[(layer_of_key(k), local)] = k
        want = {}
        for prefix, e in self.entries:
            layer = layer_of_key(prefix + "x")
            for k, p in enumerate(e.row_shards):
                found = by_name.get((layer, f"tables.{k}.weight")) or by_name.get((None, f"tables.{k}.weight"))
                if found is None:
                    raise ValueError(f"[micro_engram] {folder} has no tables.{k}.weight for {prefix.rstrip('.')}")
                saved_rows = meta[found].size[0]
                if saved_rows != e.table_rows[k]:
                    raise ValueError(f"[micro_engram] {folder}: tables.{k}.weight has {saved_rows:,} rows, this "
                                     f"engram's table {k} (order {e._table_orders[k]}) has {e.table_rows[k]:,}: "
                                     f"different table sizes can't resume (start a new run, or keep the sizes)")
                want[found] = (e.table_rows[k], p)
        templ = {k: _dtensor(p.data, rows, self.world) for k, (rows, p) in want.items()}
        dcp.load(templ, checkpoint_id=folder)
        optim_loaded = False
        if optim_folder and self.optimizer is not None and os.path.isdir(optim_folder):
            ometa = dcp.FileSystemReader(optim_folder).read_metadata().state_dict_metadata
            otempl, steps, convert = {}, {}, {}
            group = self.optimizer.param_groups[0]
            for key, rows, p in self._table_items():
                if f"{key}.exp_avg" not in ometa:
                    continue
                st = self.optimizer.state[p]
                if not st:
                    LazyAdamW.init_state(st, p, group)
                otempl[f"{key}.exp_avg"] = _dtensor(st["exp_avg"], rows, self.world)
                saved = len(ometa[f"{key}.exp_avg_sq"].size)
                sq = st["exp_avg_sq"]
                if saved != sq.dim():  # saved by the other optimizer (lazy_adam <-> rowwise_adam)
                    sq = torch.zeros(p.shape if saved == p.dim() else p.shape[:1], dtype=torch.float32,
                                     device=p.device)
                    convert[p] = sq
                otempl[f"{key}.exp_avg_sq"] = _dtensor(sq, rows, self.world)
                otempl[f"{key}.step"] = steps[p] = torch.zeros((), dtype=torch.float32)
            if otempl:
                dcp.load(otempl, checkpoint_id=optim_folder)
                for p, s in steps.items():
                    self.optimizer.state[p]["step"] = s
                for p, sq in convert.items():
                    st = self.optimizer.state[p]
                    if sq.dim() > st["exp_avg_sq"].dim():
                        # per value -> per row: the row-wise v is the mean of g^2 over the row, and
                        # v is linear in g^2, so the mean of the per-value v's is exactly it
                        st["exp_avg_sq"].copy_(sq.flatten(1).mean(dim=1))
                    else:  # per row -> per value: every value of a row gets the row's v
                        st["exp_avg_sq"].copy_(sq.reshape(-1, *([1] * (p.dim() - 1))).expand_as(p))
                self.converted = len(convert)
                optim_loaded = True
        return len(want), optim_loaded

"""Engram tables split across tensor-parallel GPUs (like vLLM's VocabParallelEmbedding).

Each TP rank keeps rows [start, start + count) of every table (micro_engram.rows.row_range).
A lookup takes the rows this rank owns, zeros for the others, and one all-reduce sums the
ranks: every row comes from exactly one rank, so the sum is exact in any dtype. Shapes
never depend on the data (mask, no indexing by nonzero), so it works under CUDA graphs
and torch.compile. Memory: 1/TP of the tables per GPU instead of all of them.

No vLLM import here, so it can be tested with plain torch.distributed.
"""
import torch
import torch.nn.functional as F

from micro_engram.module import check_loaded


def tp_lookup(all_reduce):
    """A MicroEngram.row_lookup_fn: masked local lookup, then `all_reduce` (sum over TP)."""

    def lookup(engram, idx):
        _rank, world, starts, _pers = engram.row_info
        parts = []
        for shard, i, start in zip(engram.row_shards, idx, starts):  # tables can differ in size
            n = shard.shape[0]
            local = i.long() - start
            if n == 0:  # more ranks than rows: this rank owns none
                parts.append(shard.new_zeros(*i.shape, shard.shape[1]))
                continue
            mine = (local >= 0) & (local < n)
            rows = F.embedding(local.clamp(0, n - 1), shard)
            parts.append(rows * mine.unsqueeze(-1).to(rows.dtype))
        out = torch.cat(parts, dim=-1)
        if world > 1:
            out = all_reduce(out)
        return list(out.split(engram.engram_dim, dim=-1))

    return lookup


def load_rows(engram, sd, where, complete=True):
    """Load an engram state dict (full tables, usual names) into a row-sharded engram:
    each table is sliced to this rank's rows, the rest loads as usual (scales converted by
    the module's hook). complete=False: `sd` may be part of the engram (weights streamed
    from a checkpoint), nothing is reported missing. Returns the table indices loaded."""
    sd = dict(sd)
    _rank, _world, starts, _pers = engram.row_info
    loaded = set()
    for k, shard in enumerate(engram.row_shards):
        t = sd.pop(f"tables.{k}.weight", None)
        if t is None:
            continue
        start = starts[k]
        want = (engram.table_rows[k], engram.engram_dim)
        if tuple(t.shape) != want:
            raise ValueError(f"[micro_engram] {where}: tables.{k}.weight is {tuple(t.shape)}, this engram "
                             f"needs {want}")
        with torch.no_grad():
            shard.copy_(t[start:start + shard.shape[0]])
        loaded.add(k)
    extra = [k for k in sd if k.startswith("tables.")]
    if extra:
        raise ValueError(f"[micro_engram] {where}: tables this engram does not have: {extra[:5]}")
    if complete:
        missing = [k for k in range(len(engram.row_shards)) if k not in loaded]
        if missing:
            raise ValueError(f"[micro_engram] {where}: missing tables {missing}")
        check_loaded(engram, sd, where)
    elif sd:
        if "mults" in sd and not torch.equal(sd["mults"].to(engram.mults.device).long(), engram.mults.long()):
            raise ValueError(f"[micro_engram] {where}: the saved hash multipliers differ from this engram's "
                             f"seed; the seed is wrong for these weights")
        result = engram.load_state_dict(sd, strict=False)
        if result.unexpected_keys:
            raise ValueError(f"[micro_engram] {where}: unexpected engram tensors {result.unexpected_keys[:10]}")
    return loaded

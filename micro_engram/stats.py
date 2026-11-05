"""Weight statistics for an engram, from a live module or from saved tensors.

Used by the training plugin (logged every micro_engram_stats_steps) and by
engram_stats.py (reads a checkpoint from disk while training keeps running).
"""
import re

import torch

INIT_STD = 0.02  # tables start as N(0, 0.02)
LEARNED_FACTOR = 1.5  # a row counts as "learned" once its RMS is 1.5x the init RMS

_TABLE = re.compile(r"^tables\.(\d+)\.weight$")


def _local(t):
    """Plain tensor: the local shard of an FSDP2 DTensor (no communication), else t itself."""
    t = t.detach()
    to_local = getattr(t, "to_local", None)
    return to_local() if to_local is not None else t


def weight_stats(named, orders=(2, 3), num_heads=None):
    """Stats from {local name: tensor} (tables.0.weight, value.weight, ...).

    Table i belongs to n-gram order orders[i // num_heads] (the module's layout).
    Returns {metric name: float}. Under FSDP2 each rank holds a slice of the rows, which
    is a random sample of n-grams, so stats from one rank's shard are representative.
    """
    tables = {int(m.group(1)): t for k, t in named.items() if (m := _TABLE.match(k))}
    out = {}
    if tables:
        num_heads = num_heads or max(1, len(tables) // len(orders))
        by_order = {}
        for i, t in tables.items():
            n = orders[min(i // num_heads, len(orders) - 1)]
            by_order.setdefault(n, []).append(_local(t).float())
        for n, ts in sorted(by_order.items()):
            sq = sum(float(t.pow(2).sum()) for t in ts)
            count = sum(t.numel() for t in ts)
            rows = torch.cat([t.pow(2).mean(dim=1).sqrt() for t in ts if t.numel()])
            out[f"engram/table_rms_{n}g"] = (sq / max(count, 1)) ** 0.5
            out[f"engram/rows_learned_{n}g"] = float((rows > LEARNED_FACTOR * INIT_STD).float().mean()) \
                if rows.numel() else 0.0
    for name, key in (("out_abs_mean", "value.weight"), ("out_abs_mean", "up.weight"),
                      ("key_abs_mean", "key.weight"), ("gate_in_abs_mean", "down.weight")):
        if key in named:
            out[f"engram/{name}"] = float(_local(named[key]).float().abs().mean())
    for name in ("q_scale", "k_scale"):
        if name in named:
            v = _local(named[name]).float()
            out[f"engram/{name}_mean"] = float(v.mean())
            # ups and downs cancel in the mean; the distance from 1 shows whether the scales learn
            out[f"engram/{name}_dev"] = float((v - 1).abs().mean())
    return out


def module_stats(engram):
    """weight_stats for a live MicroEngram (parameters only, no buffers)."""
    named = dict(engram.named_parameters())
    for name in getattr(engram, "DELTA_SCALES", ()):
        if name in named:  # the parameter holds scale - 1; report the scale, as saved files do
            named[name] = _local(named[name]).float() + 1
    return weight_stats(named, engram.orders, len(engram._table_orders) // len(engram.orders))


def format_stats(stats):
    """One compact line, e.g. 'out|w| 0.00213  rms 2g 0.0231 3g 0.0225  learned 2g 41% 3g 18%'."""
    parts = []
    if "engram/out_abs_mean" in stats:
        parts.append(f"out|w| {stats['engram/out_abs_mean']:.5f}")
    rms = [(k.rsplit("_", 1)[1], v) for k, v in stats.items() if k.startswith("engram/table_rms_")]
    if rms:
        parts.append("rms " + " ".join(f"{n} {v:.4f}" for n, v in rms))
    learned = [(k.rsplit("_", 1)[1], v) for k, v in stats.items() if k.startswith("engram/rows_learned_")]
    if learned:
        parts.append("learned " + " ".join(f"{n} {v:.0%}" for n, v in learned))
    if "engram/gate_mean" in stats:
        parts.append(f"gate {stats['engram/gate_mean']:.3f} (>0.99: {stats['engram/gate_saturated']:.0%})")
    if "engram/q_scale_mean" in stats:
        parts.append(f"scales |q-1| {stats['engram/q_scale_dev']:.2e} |k-1| {stats['engram/k_scale_dev']:.2e}")
    return "  ".join(parts)

"""
Engram weight stats from checkpoints on disk -- safe to run while training continues
(CPU only, reads only the engram tensors, never touches the GPUs).

    python engram_stats.py /path/to/output_dir              # newest checkpoint
    python engram_stats.py /path/to/output_dir --all        # every checkpoint still on disk
    python engram_stats.py /path/to/output_dir --watch 120  # check every 2 min, print new ones
    python engram_stats.py /path/to/output_dir/checkpoint-400
    python engram_stats.py micro_engram.pt --config micro_engram_config.json

Columns:
  out|w|      mean |weight| of the output layer (value, or up for the ple gate); 0 = not trained
  rms Ng      RMS of the N-gram tables (they start at 0.02)
  learned Ng  share of table rows whose RMS grew past 1.5x the start, i.e. rows that have
              clearly been trained -- how much of the table is in use
  scales      mean learnable gate norm scales (scalar_sqrt gate only; start at 1)

The gate mean is not in the weights; training logs it every micro_engram_stats_steps.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

from micro_engram.stats import format_stats, weight_stats
from micro_engram.weights import load_engram_weights, resolve_weights_path

CONFIG_NAME = "micro_engram_config.json"
_CKPT = re.compile(r"checkpoint-(\d+)$")
_STEP_PT = re.compile(r"micro_engram-step(\d+)\.pt$")


def _read_config(path):
    if path and os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _checkpoint_weights(ckpt):
    """Where the engram tensors of one checkpoint folder are."""
    conf_path = os.path.join(ckpt, CONFIG_NAME)
    w = resolve_weights_path(_read_config(conf_path), conf_path)
    if w and os.path.exists(w):
        return w
    for cand in (os.path.join(ckpt, "pytorch_model_fsdp_0"), os.path.join(ckpt, "micro_engram.pt")):
        if os.path.exists(cand):
            return cand
    if glob.glob(os.path.join(ckpt, "*.safetensors")):
        return ckpt
    return None


def find_snapshots(path):
    """[(step, weights path, config path)], oldest first. Only complete checkpoints."""
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        return [(None, path, None)]
    if _CKPT.search(path):
        return [(int(_CKPT.search(path).group(1)), _checkpoint_weights(path), os.path.join(path, CONFIG_NAME))]
    out = []
    for ckpt in glob.glob(os.path.join(path, "checkpoint-*")):
        m = _CKPT.search(ckpt)
        # trainer_state.json is written after the weights, so its presence means "complete"
        if m and os.path.isfile(os.path.join(ckpt, "trainer_state.json")):
            w = _checkpoint_weights(ckpt)
            if w:
                out.append((int(m.group(1)), w, os.path.join(ckpt, CONFIG_NAME)))
    for pt in glob.glob(os.path.join(path, "micro_engram-step*.pt")):  # non-FSDP runs
        out.append((int(_STEP_PT.search(pt).group(1)), pt, os.path.join(path, CONFIG_NAME)))
    if not out and _checkpoint_weights(path):  # a model folder or a DCP folder itself
        out.append((None, _checkpoint_weights(path), os.path.join(path, CONFIG_NAME)))
    return sorted(out, key=lambda s: s[0] or 0)


def stats_line(step, weights, conf_path, override_conf):
    conf = _read_config(override_conf) or _read_config(conf_path) or \
        _read_config(os.path.join(os.path.dirname(os.path.dirname(weights)), CONFIG_NAME))
    sd = load_engram_weights(weights)
    if not any(k.startswith("tables.") for k in sd):
        return f"{'step ' + str(step) if step is not None else weights}: no engram tensors found"
    orders = tuple(conf.get("orders", (2, 3)))
    stats = weight_stats(sd, orders, conf.get("num_heads"))
    label = f"step {step:>6}" if step is not None else os.path.basename(weights)
    return f"{label}  {format_stats(stats)}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="output_dir, a checkpoint-N folder, a DCP folder, a .pt or .safetensors")
    ap.add_argument("--config", default=None, help=f"{CONFIG_NAME} (default: found next to the weights)")
    ap.add_argument("--all", action="store_true", help="every checkpoint on disk, not just the newest")
    ap.add_argument("--watch", type=float, default=0, metavar="SECONDS",
                    help="keep checking for new checkpoints every SECONDS")
    args = ap.parse_args()

    seen = set()
    while True:
        snaps = find_snapshots(args.path)
        if not snaps and not args.watch:
            sys.exit(f"no complete checkpoint with engram weights under {args.path}")
        todo = snaps if (args.all or args.watch) else snaps[-1:]
        for step, weights, conf_path in todo:
            if (step, weights) in seen:
                continue
            try:
                print(stats_line(step, weights, conf_path, args.config), flush=True)
                seen.add((step, weights))
            except Exception as e:  # noqa: BLE001  -- e.g. deleted by save_total_limit mid-read
                print(f"step {step}: could not read ({type(e).__name__}: {e}); will retry", flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()

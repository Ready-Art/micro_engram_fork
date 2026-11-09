"""
Copy an engram with the tables of some n-gram orders zeroed, e.g. its 4-gram memory switched off.

    python zero_ngram_order.py SRC --order 4 --out /path/to/new_engram_folder

SRC: a micro_engram_config.json, or a folder holding one (a checkpoint-N, a training
output_dir = its newest checkpoint, or a model folder with consolidated.pth + config).
Every engram the config lists is read (DCP, .pt/.pth or safetensors, wherever its
"weights" points), the tables of the given orders are set to zero in all of them, and the
result is written to --out as:

    consolidated.pth            every engram, full key names (...layers.<N>.micro_engram....)
    micro_engram_config.json    the same settings, "weights" -> consolidated.pth

Load it like any engram (vLLM --additional-config, or swap it into the model folder).

What it tests: a zeroed table contributes nothing to the memory vector, so the gate and the
output projection only see the other orders. It is NOT the same as an engram trained without
those orders (the projections were trained next to them), but if the output changes for the
better with the 4-grams off, the 4-grams are what's pushing it. The tables keep their size
(vLLM allocates the same memory).

Holds every engram in CPU RAM while writing (~10 GB per engram for the stage-1 layout in bf16).
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from micro_engram.weights import CONFIG_NAME, engram_specs, latest_checkpoint, load_engram_weights  # noqa: E402

OUT_WEIGHTS = "consolidated.pth"


def find_config(src):
    src = os.path.abspath(src)
    if src.endswith(".json"):
        return src
    conf = os.path.join(src, CONFIG_NAME)
    if os.path.isfile(conf):
        return conf
    ckpt = latest_checkpoint(src)
    if ckpt and os.path.isfile(os.path.join(ckpt, CONFIG_NAME)):
        return os.path.join(ckpt, CONFIG_NAME)
    sys.exit(f"no {CONFIG_NAME} in {src} (give the config, a checkpoint-N folder or an output_dir)")


def table_orders(spec):
    """Order of each table, in the module's layout: every order's heads, orders ascending."""
    orders = sorted(int(n) for n in spec["orders"])
    return [n for n in orders for _ in range(int(spec["num_heads"]))]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--order", type=int, action="append", required=True, help="n-gram order to zero (repeatable)")
    ap.add_argument("--out", required=True, help="new folder (created)")
    ap.add_argument("--force", action="store_true", help="write into an existing folder")
    args = ap.parse_args()

    conf_path = find_config(args.src)
    with open(conf_path) as f:
        conf = json.load(f)
    specs = engram_specs(conf, conf_path)
    if os.path.exists(os.path.join(args.out, OUT_WEIGHTS)) and not args.force:
        sys.exit(f"{args.out} already has {OUT_WEIGHTS} (use --force)")
    os.makedirs(args.out, exist_ok=True)

    zero = set(args.order)
    out = {}
    for spec in specs:
        layer = spec["layer"]
        src_w = spec["weights"] or os.path.dirname(conf_path)
        sd = load_engram_weights(src_w, layer=layer, need_layer=len(specs) > 1)
        orders = table_orders(spec)
        n_tables = sum(1 for k in sd if k.startswith("tables."))
        if n_tables != len(orders):
            sys.exit(f"layer {layer}: {n_tables} tables in {src_w}, but the config's orders x heads say {len(orders)}")
        missing = zero - set(orders)
        if missing:
            print(f"layer {layer}: has no order {sorted(missing)} (orders {sorted(set(orders))}), nothing to zero there")
        zeroed = 0
        for k, n in enumerate(orders):
            if n in zero:
                t = sd[f"tables.{k}.weight"]
                zeroed += t.shape[0]
                t.zero_()
        print(f"layer {layer}: zeroed {zeroed:,} rows of order(s) {sorted(zero & set(orders))} "
              f"({len(orders)} tables, from {src_w})")
        for name, t in sd.items():
            out[f"model.language_model.layers.{layer}.micro_engram.{name}"] = t

    torch.save(out, os.path.join(args.out, OUT_WEIGHTS))
    new = dict(conf)
    new["weights"] = OUT_WEIGHTS
    if new.get("engrams"):
        new["engrams"] = [{**e, "weights": OUT_WEIGHTS} for e in new["engrams"]]
    new["zeroed_orders"] = sorted(zero)
    new["zeroed_from"] = conf_path
    with open(os.path.join(args.out, CONFIG_NAME), "w") as f:
        json.dump(new, f, indent=2)
    print(f"wrote {args.out}/{OUT_WEIGHTS} and {CONFIG_NAME} ({len(specs)} engram(s))")


if __name__ == "__main__":
    main()

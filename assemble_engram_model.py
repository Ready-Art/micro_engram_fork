"""
Build a servable model folder from the base weights plus a trained engram, without
copying the base weights (they are symlinked).

    python assemble_engram_model.py --base /path/to/base_model \\
        --engram /path/to/output/checkpoint-400/pytorch_model_fsdp_0 \\
        --out /path/to/base_model-engram

--engram can be:
  * a DCP checkpoint folder (checkpoint-N/pytorch_model_fsdp_0), e.g. from a run with
    micro_engram_save_trainable_only: true -- no merge step needed
  * a folder of .safetensors (e.g. a merged checkpoint) containing micro_engram weights
  * a .pt file (micro_engram.pt from a non-FSDP run, or extract_engram.py output)

micro_engram_config.json is taken from --config, else from next to the engram (the
engram folder, or its checkpoint folder), else from the base folder.

The output folder holds symlinks to every base file, micro_engram.safetensors, an
extended model.safetensors.index.json and micro_engram_config.json. vLLM (with the
micro_engram_vllm plugin) and serve_engram.py / infer_engram.py load it like any model.
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

from micro_engram.weights import base_mismatch, engram_specs, load_engram_weights

CONFIG_NAME = "micro_engram_config.json"
INDEX_NAME = "model.safetensors.index.json"
ENGRAM_FILE = "micro_engram.safetensors"
TAG = "micro_engram."


def find_config(args):
    candidates = [args.config] if args.config else []
    eng = os.path.abspath(args.engram)
    eng_dir = eng if os.path.isdir(eng) else os.path.dirname(eng)
    candidates += [os.path.join(eng_dir, CONFIG_NAME), os.path.join(os.path.dirname(eng_dir), CONFIG_NAME),
                   os.path.join(args.base, CONFIG_NAME)]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    sys.exit(f"{CONFIG_NAME} not found (looked in: {', '.join(c for c in candidates if c)}). Pass --config.")


def base_weight_map(base):
    index = os.path.join(base, INDEX_NAME)
    if os.path.isfile(index):
        with open(index) as f:
            return json.load(f)
    files = sorted(glob.glob(os.path.join(base, "*.safetensors")))
    if not files:
        sys.exit(f"no safetensors weights in {base}")
    weight_map, total = {}, 0
    for f in files:
        total += os.path.getsize(f)
        with safe_open(f, framework="pt", device="cpu") as fh:
            weight_map.update({k: os.path.basename(f) for k in fh.keys()})
    return {"metadata": {"total_size": total}, "weight_map": weight_map}


def layer_prefix(weight_map, layer):
    """Prefix of decoder layer `layer` in the base checkpoint, e.g. 'model.language_model.layers.0.'"""
    pat = re.compile(r"^(.*\.layers\.)(\d+)\.")
    prefixes = {m.group(1) for k in weight_map if (m := pat.match(k)) and TAG not in k}
    text = [p for p in prefixes if "vision" not in p and "audio" not in p]
    best = sorted(text, key=lambda p: ("language_model" not in p, len(p)))
    if not best:
        sys.exit("could not find the decoder layers in the base checkpoint")
    return f"{best[0]}{layer}."


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base model folder (the frozen backbone)")
    ap.add_argument("--engram", required=True, help="DCP checkpoint folder, safetensors folder or .pt")
    ap.add_argument("--out", required=True, help="new model folder to create")
    ap.add_argument("--config", default=None, help=f"path to {CONFIG_NAME} (default: found automatically)")
    ap.add_argument("--copy", action="store_true", help="copy the base files instead of symlinking")
    ap.add_argument("--force", action="store_true", help="write into an existing, non-empty --out")
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    out = os.path.abspath(args.out)
    if os.path.exists(out) and os.listdir(out) and not args.force:
        sys.exit(f"{out} is not empty (use --force to write into it)")

    config_path = find_config(args)
    with open(config_path) as f:
        conf = json.load(f)
    specs = engram_specs(conf, config_path)

    if (msg := base_mismatch(conf, base)):
        print(f"WARNING: {msg}")
    index = base_weight_map(base)
    weight_map = {k: v for k, v in index["weight_map"].items() if TAG not in k}  # replace any old engram
    engram = {}
    for spec in specs:
        # one engram: --engram; several: each from its config "weights" (a frozen one points to
        # the run it came from), else from --engram, read by layer
        src = args.engram if len(specs) == 1 else (spec["weights"] or args.engram)
        sd = load_engram_weights(src, layer=spec["layer"] if len(specs) > 1 else None)
        if not any(k.startswith("tables.") for k in sd):
            sys.exit(f"no micro_engram weights for layer {spec['layer']} in {src}")
        prefix = layer_prefix(weight_map, spec["layer"]) + TAG
        engram.update({prefix + k: v.contiguous() for k, v in sd.items()})
        print(f"layer {spec['layer']}: {len(sd)} tensors from {src}")

    os.makedirs(out, exist_ok=True)
    for src in sorted(glob.glob(os.path.join(base, "*"))):
        name = os.path.basename(src)
        if name in (INDEX_NAME, CONFIG_NAME, ENGRAM_FILE):
            continue
        dst = os.path.join(out, name)
        if os.path.lexists(dst):
            os.remove(dst) if not os.path.isdir(dst) or os.path.islink(dst) else shutil.rmtree(dst)
        if args.copy:
            (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
        else:
            os.symlink(src, dst)

    save_file(engram, os.path.join(out, ENGRAM_FILE), metadata={"format": "pt"})
    weight_map.update({k: ENGRAM_FILE for k in engram})
    meta = dict(index.get("metadata", {}))
    meta["total_size"] = int(meta.get("total_size", 0)) + sum(v.numel() * v.element_size() for v in engram.values())
    with open(os.path.join(out, INDEX_NAME), "w") as f:
        json.dump({"metadata": meta, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)
    # the engram is now inside this folder, so the config must not point elsewhere
    out_conf = {k: v for k, v in conf.items() if k != "weights"}
    if "engrams" in out_conf:
        out_conf["engrams"] = [{k: v for k, v in e.items() if k != "weights"} for e in out_conf["engrams"]]
    with open(os.path.join(out, CONFIG_NAME), "w") as f:
        json.dump(out_conf, f, indent=2)

    size = sum(v.numel() * v.element_size() for v in engram.values()) / 2**30
    print(f"engram: {len(engram)} tensors, {size:.2f} GiB, keys like {next(iter(engram))}")
    print(f"config: {config_path} (" + "; ".join(f"layer {sp['layer']}, {sp['gate']} gate, seed {sp['seed']}"
                                                for sp in specs) + ")")
    print(f"wrote {out} ({'copied' if args.copy else 'symlinked'} base files from {base})")


if __name__ == "__main__":
    main()

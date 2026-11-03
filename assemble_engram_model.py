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
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CONFIG_NAME = "micro_engram_config.json"
INDEX_NAME = "model.safetensors.index.json"
ENGRAM_FILE = "micro_engram.safetensors"
TAG = "micro_engram."


def flatten(d, prefix=""):
    """{"model": {"a.b": t}} -> {"model.a.b": t} (DCP checkpoints come back nested)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def local_names(sd):
    """Keep the engram tensors, named relative to the engram (tables.0.weight, canon, ...)."""
    out = {}
    for k, v in sd.items():
        if not torch.is_tensor(v):
            continue
        if f".{TAG}" in k:
            out[k.split(f".{TAG}", 1)[1]] = v
        elif k.startswith(TAG):
            out[k[len(TAG):]] = v
        elif not any(c in k for c in (".layers.", "embed", "lm_head")):
            out[k] = v  # already local (micro_engram.pt)
    return out


def load_engram(path):
    if os.path.isfile(path):
        return local_names(flatten(torch.load(path, map_location="cpu", weights_only=True)))
    if os.path.isfile(os.path.join(path, ".metadata")):  # DCP checkpoint folder
        from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "full.pt")
            dcp_to_torch_save(path, out)
            return local_names(flatten(torch.load(out, map_location="cpu", weights_only=True)))
    sd = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as fh:
            for k in fh.keys():
                if TAG in k:
                    sd[k] = fh.get_tensor(k)
    return local_names(sd)


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

    sd = load_engram(args.engram)
    if not any(k.startswith("tables.") for k in sd):
        sys.exit(f"no micro_engram weights found in {args.engram}")
    config_path = find_config(args)
    with open(config_path) as f:
        conf = json.load(f)

    index = base_weight_map(base)
    weight_map = {k: v for k, v in index["weight_map"].items() if TAG not in k}  # replace any old engram
    prefix = layer_prefix(weight_map, conf.get("layer", 0)) + TAG
    engram = {prefix + k: v.contiguous() for k, v in sd.items()}

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
    shutil.copy2(config_path, os.path.join(out, CONFIG_NAME))

    size = sum(v.numel() * v.element_size() for v in engram.values()) / 2**30
    print(f"engram: {len(engram)} tensors, {size:.2f} GiB, keys like {next(iter(engram))}")
    print(f"config: {config_path} (layer {conf.get('layer', 0)}, gate {conf.get('gate', 'scalar')})")
    print(f"wrote {out} ({'copied' if args.copy else 'symlinked'} base files from {base})")


if __name__ == "__main__":
    main()

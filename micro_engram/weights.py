"""Read engram weights from wherever they were saved, named relative to the engram
(tables.0.weight, key.weight, canon, ...).

Supported: a .pt file (micro_engram.pt, extract_engram.py output), a .safetensors file,
a folder of .safetensors (merged or assembled model), and a DCP checkpoint folder
(checkpoint-N/pytorch_model_fsdp_0). From a DCP checkpoint only the engram tensors are
read, so this is fast even when the checkpoint holds the whole model.
"""
import glob
import json
import os
import re

import torch

TAG = "micro_engram."


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _local_name(key):
    """'model.language_model.layers.0._checkpoint_wrapped_module.micro_engram.tables.0.weight'
    -> 'tables.0.weight'; None for keys that are not engram weights."""
    if f".{TAG}" in key:
        return key.split(f".{TAG}", 1)[1]
    if key.startswith(TAG):
        return key[len(TAG):]
    return None


def _from_full_keys(sd):
    return {name: v for k, v in sd.items() if torch.is_tensor(v) and (name := _local_name(k))}


def _load_dcp(path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = dcp.FileSystemReader(path)
    meta = reader.read_metadata().state_dict_metadata
    wanted = {
        k: torch.empty(tuple(m.size), dtype=m.properties.dtype)
        for k, m in meta.items()
        if isinstance(m, TensorStorageMetadata) and _local_name(k) is not None
    }
    if wanted:
        # no_dist: every process reads its own copy, no collectives (safe inside vLLM workers)
        dcp.load(wanted, storage_reader=reader, no_dist=True)
    return _from_full_keys(wanted)


def _load_safetensors(files):
    from safetensors import safe_open

    sd = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as fh:
            for k in fh.keys():
                if (name := _local_name(k)) is not None:
                    sd[name] = fh.get_tensor(k)
    return sd


def load_engram_weights(path):
    """Engram tensors from `path`, keyed by their name inside the engram module."""
    path = os.path.abspath(path)
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, ".metadata")):
            return _load_dcp(path)
        return _load_safetensors(sorted(glob.glob(os.path.join(path, "*.safetensors"))))
    if path.endswith(".safetensors"):
        sd = _load_safetensors([path])
        if not sd:  # a file saved with local names already
            from safetensors.torch import load_file

            sd = load_file(path)
        return sd
    sd = _flatten(torch.load(path, map_location="cpu", weights_only=True))
    full = _from_full_keys(sd)
    return full or {k: v for k, v in sd.items() if torch.is_tensor(v)}  # micro_engram.pt has local names


def resolve_weights_path(conf, conf_path):
    """The engram weights a micro_engram_config.json points to ("weights", relative to the
    config file), or None."""
    w = conf.get("weights")
    if not w:
        return None
    return w if os.path.isabs(w) else os.path.join(os.path.dirname(os.path.abspath(conf_path)), w)


# ----------------------------------------------------------------------------- base check
_FP_LAYER = re.compile(r"^(.*\.layers\.)(\d+)\.(self_attn\.[qkvo]_proj|mlp\.(?:gate|up|down)_proj)\.weight$")
_FP_ROWS = 64


def _weight_files(model_dir):
    """{tensor name: safetensors file} of a local model folder, or {} if there is none."""
    index = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.isfile(index):
        with open(index) as f:
            return {k: os.path.join(model_dir, v) for k, v in json.load(f)["weight_map"].items()}
    from safetensors import safe_open

    files = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as fh:
            files.update({k: path for k in fh.keys()})
    return files


def base_fingerprint(model_dir):
    """Short hash of the base model's weights, independent of the folder name.

    Hashes the first rows of the token embeddings and of every attention / MLP projection
    in the first, middle and last text layer, read straight from the safetensors files (fast even
    for a 62 GB model). An SFT merge or retrained embedding rows change it; renaming or
    copying the folder does not. None if the folder has no safetensors weights.
    """
    import hashlib

    from safetensors import safe_open

    if not os.path.isdir(model_dir):
        return None
    files = {k: f for k, f in _weight_files(model_dir).items()
             if TAG not in k and "vision" not in k and "audio" not in k}
    if not files:
        return None
    # role -> tensor name; roles (not names) are hashed, so a multimodal checkpoint
    # (model.language_model.*) and a text-only one (model.*) with the same weights match
    chosen = {"embed": k for k in files if k.endswith("embed_tokens.weight")}
    layers = {}
    for k in files:
        if (m := _FP_LAYER.match(k)):
            layers.setdefault(int(m.group(2)), {})[m.group(3)] = k
    if layers:
        ids = sorted(layers)
        for where, i in (("first", ids[0]), ("middle", ids[len(ids) // 2]), ("last", ids[-1])):
            chosen.update({f"{where}.{r}": k for r, k in layers[i].items()})
    if not chosen:
        return None
    h = hashlib.sha256()
    for role in sorted(chosen):
        k = chosen[role]
        with safe_open(files[k], framework="pt", device="cpu") as fh:
            t = fh.get_slice(k)[:_FP_ROWS].contiguous()
        h.update(role.encode())
        h.update(str((t.dtype, tuple(t.shape))).encode())
        h.update(t.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()[:16]


def base_mismatch(conf, model_dir):
    """A warning text if the engram was trained on different base weights, else None."""
    want = conf.get("base_fingerprint")
    if not want:
        return None  # older engram: nothing recorded to compare with
    have = base_fingerprint(model_dir)
    if have is None or have == want:
        return None
    return (f"this engram was trained on different base weights (trained on {conf.get('base_model', '?')}, "
            f"fingerprint {want}; this model: {have}). It still loads, but it injects into hidden "
            f"states it never saw: expect worse output unless this is really the same base.")

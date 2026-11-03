"""Read engram weights from wherever they were saved, named relative to the engram
(tables.0.weight, key.weight, canon, ...).

Supported: a .pt file (micro_engram.pt, extract_engram.py output), a .safetensors file,
a folder of .safetensors (merged or assembled model), and a DCP checkpoint folder
(checkpoint-N/pytorch_model_fsdp_0). From a DCP checkpoint only the engram tensors are
read, so this is fast even when the checkpoint holds the whole model.
"""
import glob
import os

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

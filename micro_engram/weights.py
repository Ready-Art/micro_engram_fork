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


_LAYER_OF = re.compile(r"(?:^|\.)layers\.(\d+)\.(?:_checkpoint_wrapped_module\.)?micro_engram\.")


def layer_of_key(key):
    """Decoder layer index an engram key belongs to, or None if the key does not say."""
    m = _LAYER_OF.search(key)
    return int(m.group(1)) if m else None


def _wanted(key, layer):
    """Is `key` an engram weight of `layer` (any layer when layer is None)?"""
    if _local_name(key) is None:
        return False
    if layer is None:
        return True
    k_layer = layer_of_key(key)
    return k_layer is None or k_layer == layer


def _from_full_keys(sd, layer=None):
    return {_local_name(k): v for k, v in sd.items() if torch.is_tensor(v) and _wanted(k, layer)}


def _load_dcp(path, layer=None):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = dcp.FileSystemReader(path)
    meta = reader.read_metadata().state_dict_metadata
    wanted = {
        k: torch.empty(tuple(m.size), dtype=m.properties.dtype)
        for k, m in meta.items()
        if isinstance(m, TensorStorageMetadata) and _wanted(k, layer)
    }
    if wanted:
        # no_dist: every process reads its own copy, no collectives (safe inside vLLM workers)
        dcp.load(wanted, storage_reader=reader, no_dist=True)
    return _from_full_keys(wanted, layer)


def _load_safetensors(files, layer=None):
    from safetensors import safe_open

    sd = {}
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as fh:
            for k in fh.keys():
                if _wanted(k, layer):
                    sd[_local_name(k)] = fh.get_tensor(k)
    return sd


def engram_layers_in(path):
    """Layer indices of the engrams stored at `path` (empty if the keys don't say)."""
    path = os.path.abspath(path)
    keys = []
    if os.path.isdir(path) and os.path.isfile(os.path.join(path, ".metadata")):
        import torch.distributed.checkpoint as dcp

        keys = list(dcp.FileSystemReader(path).read_metadata().state_dict_metadata)
    elif os.path.isdir(path) or path.endswith(".safetensors"):
        from safetensors import safe_open

        files = sorted(glob.glob(os.path.join(path, "*.safetensors"))) if os.path.isdir(path) else [path]
        for f in files:
            with safe_open(f, framework="pt", device="cpu") as fh:
                keys += list(fh.keys())
    else:
        keys = list(_flatten(torch.load(path, map_location="cpu", weights_only=True)))
    return sorted({lay for k in keys if _local_name(k) is not None and (lay := layer_of_key(k)) is not None})


def load_engram_weights(path, layer=None):
    """Engram tensors from `path`, keyed by their name inside the engram module.

    With `layer`, only the engram of that decoder layer is read (checkpoints of runs with
    several engrams hold one per layer). Files saved with local names (micro_engram.pt)
    carry no layer and are returned as they are.
    """
    path = os.path.abspath(path)
    if os.path.isdir(path):
        if os.path.isfile(os.path.join(path, ".metadata")):
            return _load_dcp(path, layer)
        return _load_safetensors(sorted(glob.glob(os.path.join(path, "*.safetensors"))), layer)
    if path.endswith(".safetensors"):
        sd = _load_safetensors([path], layer)
        if not sd:  # a file saved with local names already
            from safetensors.torch import load_file

            sd = load_file(path)
        return sd
    sd = _flatten(torch.load(path, map_location="cpu", weights_only=True))
    full = _from_full_keys(sd, layer)
    return full or {k: v for k, v in sd.items() if torch.is_tensor(v)}  # micro_engram.pt has local names


def resolve_weights_path(conf, conf_path):
    """The engram weights a micro_engram_config.json points to ("weights", relative to the
    config file), or None."""
    w = conf.get("weights")
    if not w:
        return None
    return w if os.path.isabs(w) else os.path.join(os.path.dirname(os.path.abspath(conf_path)), w)


# ----------------------------------------------------------------------------- engram specs
SEED_STRIDE = 10007  # default seed = SEED_STRIDE * layer (DeepSeek's scheme); layer 0 -> 0

_SPEC_KEYS = ("layer", "table_size", "engram_dim", "num_heads", "orders", "gate", "seed",
              "pad_id", "token_merge", "canon_path", "frozen", "init_from")


def default_seed(layer):
    return SEED_STRIDE * int(layer)


def engram_specs(conf, conf_path=None):
    """One dict per engram described by a micro_engram_config.json.

    New configs list every engram under "engrams" (each with its own "weights"); older
    ones describe a single engram at the top level. "weights" is resolved against the
    config's folder (None: the engram is inside the model checkpoint itself). Configs
    without a "seed" are from before seeds were recorded: those engrams used seed 0.
    """
    top = {k: conf.get(k) for k in _SPEC_KEYS}
    entries = conf.get("engrams")
    if not entries:
        entries = [{**top, "weights": conf.get("weights")}]
    specs = []
    for e in entries:
        spec = {**top, **{k: v for k, v in e.items() if v is not None or k == "weights"}}
        spec["layer"] = int(spec.get("layer") or 0)
        spec["orders"] = tuple(spec.get("orders") or (2, 3))
        spec["gate"] = spec.get("gate") or "scalar"  # runs from before the gate option
        spec["seed"] = 0 if spec.get("seed") is None else int(spec["seed"])
        w = e.get("weights")
        if w and conf_path and not os.path.isabs(w):
            w = os.path.join(os.path.dirname(os.path.abspath(conf_path)), w)
        spec["weights"] = w or None
        specs.append(spec)
    layers = [s["layer"] for s in specs]
    if len(set(layers)) != len(layers):
        raise ValueError(f"[micro_engram] two engrams in the same layer: {layers}")
    return specs


CONFIG_NAME = "micro_engram_config.json"
_CKPT = re.compile(r"checkpoint-(\d+)$")


def latest_checkpoint(folder):
    """Newest complete checkpoint-N inside `folder` (trainer_state.json is written last), or None."""
    done = [(int(m.group(1)), c) for c in glob.glob(os.path.join(folder, "checkpoint-*"))
            if (m := _CKPT.search(c)) and os.path.isfile(os.path.join(c, "trainer_state.json"))]
    return max(done)[1] if done else None


def resolve_engram_source(path, layer):
    """Where the trained engram for decoder `layer` is: (weights path, its spec or None).

    `path` may be a micro_engram_config.json, a folder holding one (checkpoint-N, an
    assembled model, an output_dir -> its newest checkpoint), or the weights themselves
    (DCP folder, .pt, .safetensors, safetensors folder). The spec (from the config) gives
    the engram's settings, seed included, so it is rebuilt exactly as it was trained.
    """
    path = os.path.abspath(path)
    conf_path = path if path.endswith(".json") else os.path.join(path, CONFIG_NAME)
    if os.path.isdir(path) and not os.path.isfile(os.path.join(path, ".metadata")):
        has_conf = os.path.isfile(conf_path)
        has_weights = bool(glob.glob(os.path.join(path, "*.safetensors")))
        if not has_weights and (ckpt := latest_checkpoint(path)) is not None:
            spec_conf = conf_path if has_conf else None
            weights, spec = resolve_engram_source(ckpt, layer)
            if spec is None and spec_conf:
                spec = _spec_for_layer(spec_conf, layer)
            return weights, spec
    if os.path.isfile(conf_path):
        spec = _spec_for_layer(conf_path, layer)
        weights = spec["weights"] or os.path.dirname(conf_path)  # None: inside the model folder
        return weights, spec
    return path, None


def _spec_for_layer(conf_path, layer):
    with open(conf_path) as f:
        specs = engram_specs(json.load(f), conf_path)
    for spec in specs:
        if spec["layer"] == layer:
            return spec
    raise ValueError(f"[micro_engram] {conf_path} has no engram at layer {layer} "
                     f"(it has layers {[s['layer'] for s in specs]})")


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

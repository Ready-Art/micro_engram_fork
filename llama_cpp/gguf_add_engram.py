"""
Add trained micro-engram(s) to a Gemma 4 GGUF, for llama.cpp built with the micro_engram patch.

    python llama_cpp/gguf_add_engram.py --gguf gemma-4-31B-it-Q4_K_M.gguf \\
        --engram /path/to/output/checkpoint-400 --out gemma-4-31B-it-Q4_K_M-engram.gguf

--engram: a checkpoint-N folder, an assembled model folder, or a micro_engram_config.json
(every engram it lists is added, each from its own "weights"). The base GGUF can be
quantized (Q4/Q8/...): its tensors are copied unchanged, the engram is added as BF16
(--dtype f16/f32 also possible). Run from the micro_engram repo root. Needs the `gguf`
package: `pip install gguf`, or PYTHONPATH=/path/to/llama.cpp/gguf-py.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

try:
    import gguf
except ImportError as e:
    sys.exit(f"could not import the gguf package ({type(e).__name__}: {e}) with {sys.executable}. "
             "Install it for this Python (pip install gguf) or use llama.cpp's copy: "
             "PYTHONPATH=/path/to/llama.cpp/gguf-py")

from micro_engram.module import MicroEngram, check_loaded, engram_from_spec
from micro_engram.weights import engram_specs, load_engram_weights

CONFIG_NAME = "micro_engram_config.json"
GATES = {"scalar": 0, "scalar_sqrt": 1, "ple": 2}
PREFIX = "micro_engram."


def find_config(path):
    path = os.path.abspath(path)
    conf_path = path if path.endswith(".json") else os.path.join(path, CONFIG_NAME)
    if not os.path.isfile(conf_path):
        sys.exit(f"{CONFIG_NAME} not found at {conf_path}")
    return conf_path


def field_value(reader, key):
    f = reader.fields.get(key)
    return None if f is None else f.contents()


def load_engrams(conf_path, n_embd):
    """[(spec, MicroEngram with trained weights)], in layer order."""
    with open(conf_path) as f:
        conf = json.load(f)
    specs = engram_specs(conf, conf_path)
    out = []
    for spec in sorted(specs, key=lambda s: s["layer"]):
        src = spec["weights"] or os.path.dirname(conf_path)  # None: inside the model folder
        sd = load_engram_weights(src, layer=spec["layer"] if len(specs) > 1 else None)
        if not any(k.startswith("tables.") for k in sd):
            sys.exit(f"no engram weights for layer {spec['layer']} in {src}")
        vocab = int(sd["canon"].numel()) if "canon" in sd else int(conf.get("vocab_size") or 262144)
        canon = None
        if "canon" not in sd and spec.get("token_merge"):
            canon = torch.load(spec["canon_path"], map_location="cpu")
        eng = engram_from_spec(spec, n_embd, vocab, conf.get("pad_id") or 0, canon)
        check_loaded(eng, sd, src)
        print(f"layer {spec['layer']}: {spec['gate']} gate, {len(eng.tables)} tables x {eng.table_size} rows x "
              f"{eng.tables[0].embedding_dim}, seed {spec['seed']}, from {src}")
        out.append((spec, eng))
    return out


def as_dtype(t, dtype):
    """(numpy array, raw GGML type or None) for a torch tensor."""
    t = t.detach().float()
    if dtype == "f32":
        return t.numpy(), None
    if dtype == "f16":
        return t.half().numpy(), None
    return t.bfloat16().view(torch.int16).numpy().view(np.uint16), gguf.GGMLQuantizationType.BF16


def engram_tensors(layer, eng: MicroEngram, dtype):
    """{gguf name: (array, raw dtype)}. Names match the patch: blk.N.engram_kv.mengram_* (matrices)
    and blk.N.engram_gate.mengram_* (norm scales, always F32)."""
    kv, vec = f"blk.{layer}.engram_kv.", f"blk.{layer}.engram_gate."
    tables = torch.cat([t.weight for t in eng.tables], dim=0)  # [n_tables*table_size, dim]
    out = {kv + "mengram_tables": as_dtype(tables, dtype)}
    if eng.gate == "ple":
        out[kv + "mengram_down"] = as_dtype(eng.down.weight, dtype)
        out[kv + "mengram_up"] = as_dtype(eng.up.weight, dtype)
    else:
        out[kv + "mengram_key"] = as_dtype(eng.key.weight, dtype)
        out[kv + "mengram_value"] = as_dtype(eng.value.weight, dtype)
        if eng.gate == "scalar_sqrt":
            sd = eng.state_dict()  # holds the scales themselves (the parameters hold scale - 1)
            out[vec + "mengram_q_scale"] = as_dtype(sd["q_scale"], "f32")
            out[vec + "mengram_k_scale"] = as_dtype(sd["k_scale"], "f32")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="Gemma 4 GGUF to add the engram to (any quantization)")
    ap.add_argument("--engram", required=True, help="checkpoint-N folder, assembled folder or micro_engram_config.json")
    ap.add_argument("--out", required=True, help="new GGUF file")
    ap.add_argument("--dtype", choices=["bf16", "f16", "f32"], default="bf16", help="engram tensor type (default bf16)")
    ap.add_argument("--force", action="store_true", help="overwrite --out; replace an engram already in --gguf")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.force:
        sys.exit(f"{args.out} exists (use --force)")
    reader = gguf.GGUFReader(args.gguf)
    arch = field_value(reader, gguf.Keys.General.ARCHITECTURE)
    if arch != "gemma4":
        sys.exit(f"{args.gguf} is a {arch!r} model; the micro_engram patch supports gemma4")
    if field_value(reader, "split.count"):
        sys.exit("split GGUFs are not supported: merge it first (llama-gguf-split --merge)")
    has_old = any(k.startswith(PREFIX) for k in reader.fields)
    if has_old and not args.force:
        sys.exit(f"{args.gguf} already has an engram (use --force to replace it)")

    n_embd = int(field_value(reader, f"{arch}.embedding_length"))
    n_layer = int(field_value(reader, f"{arch}.block_count"))
    engrams = load_engrams(find_config(args.engram), n_embd)

    canon = engrams[0][1].canon
    for spec, eng in engrams[1:]:
        if not torch.equal(eng.canon, canon):
            sys.exit(f"the engram at layer {spec['layer']} uses a different token-merge map than the first; "
                     f"one GGUF holds one map")
    tokens = field_value(reader, gguf.Keys.Tokenizer.LIST)
    if tokens is not None and len(tokens) != canon.numel():
        print(f"WARNING: the GGUF vocab has {len(tokens)} tokens, the engram's token map {canon.numel()}: "
              f"same tokenizer as training?")
    for spec, _ in engrams:
        if spec["layer"] >= n_layer:
            sys.exit(f"engram layer {spec['layer']} but the model has {n_layer} layers")

    max_order = max(max(eng.orders) for _, eng in engrams)
    orders, mults = [], []
    for _, eng in engrams:
        m = eng.mults.long()
        for k, n in enumerate(eng._table_orders):
            orders.append(int(n))
            row = [int(v) for v in m[k, :n]] + [0] * (max_order - n)
            mults.extend(row)

    T = gguf.GGUFValueType
    meta = {
        "layers": ([s["layer"] for s, _ in engrams], T.UINT32),
        "table_size": ([e.table_size for _, e in engrams], T.UINT32),
        "dim": ([e.tables[0].embedding_dim for _, e in engrams], T.UINT32),
        "n_tables": ([len(e.tables) for _, e in engrams], T.UINT32),
        "gate": ([GATES[e.gate] for _, e in engrams], T.UINT32),
        "pad_id": ([int(e.pad_id) for _, e in engrams], T.INT32),
        "table_orders": (orders, T.UINT32),
        "mults": (mults, T.UINT64),
        "canon": ([int(v) for v in canon.long()], T.INT32),
    }

    writer = gguf.GGUFWriter(args.out, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF.") or field.name.startswith(PREFIX):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == T.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    for key, (val, sub) in meta.items():
        writer.add_key_value(PREFIX + key, val, T.ARRAY, sub_type=sub)
    writer.add_key_value(PREFIX + "max_order", max_order, T.UINT32)

    base = [t for t in reader.tensors if ".mengram_" not in t.name]
    new = {}
    for spec, eng in engrams:
        new.update(engram_tensors(spec["layer"], eng, args.dtype))
    for t in base:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for name, (arr, raw) in new.items():
        writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, raw)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in base:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    for arr, _ in new.values():
        writer.write_tensor_data(arr)
    writer.close()

    size = sum(a.nbytes for a, _ in new.values()) / 2**30
    print(f"wrote {args.out}: {len(base)} base tensors + {len(new)} engram tensors ({size:.2f} GiB, {args.dtype})")


if __name__ == "__main__":
    main()

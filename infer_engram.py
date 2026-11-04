"""
Test inference for a Gemma 4 model trained with micro-engram.

Usage:
  python infer_engram.py --model /path/to/merged_model --prompt "Hello there"
  python infer_engram.py --model /path/to/merged_model --prompt "Hello" --no-engram
  python infer_engram.py --model /path/to/base_model --engram-pt /path/to/micro_engram.pt --prompt "Hello"

Engram settings (layer, table size, dim, heads) are found automatically:
  explicit flag > micro_engram_config.json > read from the saved tensors > defaults.
Copy micro_engram_config.json from the training output folder into the merged model
folder (or pass --engram-config).

Put this file next to the micro_engram package folder (so `from micro_engram.module import ...` works),
or edit the import below.
"""
import argparse
import glob
import json
import os
import re

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

from micro_engram.module import MicroEngram, find_text_layers
from micro_engram.weights import load_engram_weights as load_engram_file
from micro_engram.weights import base_mismatch, resolve_weights_path

CONFIG_NAME = "micro_engram_config.json"
_LAYER_RE = re.compile(r"layers\.(\d+)\.micro_engram\.")


def load_engram_weights(model_path, engram_pt):
    """Returns (state dict with the key prefix stripped, layer index seen in key names or None)."""
    if engram_pt:  # .pt, .safetensors, a safetensors folder or a DCP checkpoint folder
        return load_engram_file(engram_pt), None

    sd, layer = {}, None

    def take(key, val):
        nonlocal layer
        m = _LAYER_RE.search(key)
        if m:
            layer = int(m.group(1))
        sd[key.split("micro_engram.", 1)[1]] = val

    for f in sorted(glob.glob(os.path.join(model_path, "*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as sf:
            for key in sf.keys():
                if "micro_engram." in key:
                    take(key, sf.get_tensor(key))
    if not sd:
        # fall back to a pytorch .bin checkpoint
        for f in sorted(glob.glob(os.path.join(model_path, "*.bin"))):
            part = torch.load(f, map_location="cpu")
            for key, val in part.items():
                if "micro_engram." in key:
                    take(key, val)
    return sd, layer


def engram_pt_from_config(args):
    """--engram-config with a "weights" field works like --engram-pt (unless one is given)."""
    if args.engram_pt or not args.engram_config:
        return args.engram_pt
    with open(args.engram_config) as f:
        return resolve_weights_path(json.load(f), args.engram_config)


def warn_if_other_base(args):
    """Print a warning if the engram was trained on different base weights than --model."""
    conf_path = args.engram_config or os.path.join(args.model, CONFIG_NAME)
    if os.path.isfile(conf_path):
        with open(conf_path) as f:
            if (msg := base_mismatch(json.load(f), args.model)):
                print(f"[engram] WARNING: {msg}")


def pick(*values):
    """First value that is not None."""
    for v in values:
        if v is not None:
            return v
    return None


def resolve_settings(args, sd, layer_from_key):
    conf, source = {}, None
    conf_path = args.engram_config or os.path.join(args.model, CONFIG_NAME)
    if os.path.isfile(conf_path):
        with open(conf_path) as f:
            conf = json.load(f)
        source = conf_path
    elif args.engram_config:
        raise SystemExit(f"--engram-config file not found: {args.engram_config}")

    orders = tuple(conf.get("orders", (2, 3)))

    # what the saved tensors say
    n_tables = sum(1 for k in sd if k.startswith("tables.") and k.endswith(".weight"))
    t0 = sd.get("tables.0.weight")
    from_shapes = {
        "table_size": int(t0.shape[0]) if t0 is not None else None,
        "engram_dim": int(t0.shape[1]) if t0 is not None else None,
        "num_heads": (n_tables // len(orders)) if n_tables else None,
        "gate": "ple" if "up.weight" in sd else ("scalar_sqrt" if "q_scale" in sd
                                                  else ("scalar" if "value.weight" in sd else None)),
    }

    settings = {
        "layer": pick(args.layer, conf.get("layer"), layer_from_key, 0),
        "table_size": pick(args.table_size, conf.get("table_size"), from_shapes["table_size"], 65_536),
        "engram_dim": pick(args.dim, conf.get("engram_dim"), from_shapes["engram_dim"], 128),
        "num_heads": pick(args.heads, conf.get("num_heads"), from_shapes["num_heads"], 4),
        "orders": orders,
        "pad_id": conf.get("pad_id"),
        "model_dim": conf.get("model_dim"),
        "token_merge": conf.get("token_merge"),
        "canon_path": conf.get("canon_path"),
        # configs from before the gate option have no "gate": those runs were scalar
        "gate": pick(getattr(args, "gate", None), conf.get("gate"), from_shapes["gate"], "scalar"),
    }

    if source:
        print(f"[engram] settings: {source} (explicit flags override it)")
    elif t0 is not None:
        print("[engram] no micro_engram_config.json found, settings read from the saved tensors "
              "(assumes n-gram orders 2 and 3)")
    else:
        print("[engram] no config and no tensors to read, using defaults")
    shown = {k: v for k, v in settings.items() if k not in ("pad_id", "model_dim", "canon_path")}
    print(f"[engram] using: {shown}")
    return settings


class EngramRunner:
    """Attaches the engram to the model and keeps a full token history for generation."""

    def __init__(self, model, engram, layer_idx):
        self.model = model
        self.engram = engram
        self.hist = None  # all token ids seen so far in the current generation
        self.enabled = True

        _, layers = find_text_layers(model)
        target = layers[layer_idx]
        ref = next(target.parameters())
        engram.to(device=ref.device, dtype=ref.dtype)
        target.add_module("micro_engram", engram)

        model.register_forward_pre_hook(self._grab_ids, with_kwargs=True)
        target.register_forward_hook(self._inject)

    def reset(self):
        self.hist = None

    def _grab_ids(self, _mod, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args:
            ids = args[0]
        if ids is None:
            return
        if self.hist is None:
            self.hist = ids
        else:
            self.hist = torch.cat([self.hist, ids.to(self.hist.device)], dim=1)

    def _inject(self, _mod, _args, output):
        if not self.enabled or self.hist is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        new = self._engram_forward(hidden)
        if isinstance(output, tuple):
            return (new,) + tuple(output[1:])
        return new

    @torch.no_grad()
    def _engram_forward(self, hidden):
        eng = self.engram
        T = hidden.shape[1]
        # Hash only the tail: the T new positions plus the (max order - 1) tokens before them,
        # which is all their n-grams can reach. Same result as hashing the whole history,
        # without redoing the whole conversation for every generated token.
        tail = self.hist[:, -(T + max(eng.orders) - 1):]
        idx = eng.hash_ids(tail.to(hidden.device))
        idx = [i[:, -T:] for i in idx]
        return eng.inject(hidden, idx)  # same gate code as training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged model dir (or base model dir if using --engram-pt)")
    ap.add_argument("--engram-pt", default=None, help="separate engram: .pt, .safetensors, a safetensors folder or a DCP checkpoint folder")
    ap.add_argument("--engram-config", default=None,
                    help=f"path to {CONFIG_NAME} (default: looked up inside --model)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--raw", action="store_true", help="skip the chat template, feed the prompt as plain text")
    ap.add_argument("--no-engram", action="store_true", help="turn the engram off to compare outputs")
    ap.add_argument("--max-new-tokens", type=int, default=2000)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    # optional overrides; normally found automatically
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--table-size", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--heads", type=int, default=None)
    ap.add_argument("--gate", choices=MicroEngram.GATES, default=None)
    args = ap.parse_args()

    # load the engram weights first, so a wrong path fails before the slow model load
    args.engram_pt = engram_pt_from_config(args)
    warn_if_other_base(args)
    sd, layer_from_key = load_engram_weights(args.model, args.engram_pt)
    if not sd:
        raise SystemExit("No micro_engram weights found in the checkpoint. Wrong path, or they were not saved.")
    settings = resolve_settings(args, sd, layer_from_key)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",  # spreads layers over all 5 GPUs
    )
    model.eval()

    text_cfg = getattr(model.config, "text_config", model.config)
    if settings["model_dim"] and settings["model_dim"] != text_cfg.hidden_size:
        raise SystemExit(f"Config says model_dim {settings['model_dim']} but the model has "
                         f"hidden_size {text_cfg.hidden_size}. Wrong config for this model?")

    # The token-merge map is normally saved with the weights. If the run used one but the
    # weights lack it, hashing would silently differ from training: load it or stop.
    canon = None
    if "canon" not in sd and settings["token_merge"]:
        path = settings["canon_path"]
        if not path or not os.path.isfile(path):
            raise SystemExit(f"Training used token merging but the weights have no 'canon' tensor and "
                             f"the canon file is not available ({path}). Copy it here and fix canon_path "
                             f"in {CONFIG_NAME}.")
        canon = torch.load(path, map_location="cpu")
        print(f"[engram] token-merge map loaded from {path}")

    engram = MicroEngram(
        model_dim=text_cfg.hidden_size,
        table_size=settings["table_size"],
        engram_dim=settings["engram_dim"],
        num_heads=settings["num_heads"],
        orders=settings["orders"],
        pad_id=pick(settings["pad_id"], text_cfg.pad_token_id, 0),
        vocab_size=text_cfg.vocab_size,
        canon=canon,
        gate=settings["gate"],
    )
    runner = EngramRunner(model, engram, settings["layer"])

    missing, unexpected = engram.load_state_dict(sd, strict=False)
    print(f"[engram] loaded {len(sd)} tensors | missing: {missing} | unexpected: {unexpected}")
    # 'mults' missing is fine: it is rebuilt from the seed. Anything else missing is a problem.

    # Sanity check: the output layer starts at zero, so a trained engram must have a non-zero one.
    out_layer = engram.up if engram.gate == "ple" else engram.value
    print(f"[engram] output-weight magnitude ({engram.gate} gate): "
          f"{out_layer.weight.float().abs().mean().item():.6f} "
          f"(0.000000 would mean it never trained or did not load)")

    runner.enabled = not args.no_engram
    print(f"[engram] enabled: {runner.enabled}")

    if args.raw or not getattr(tok, "chat_template", None):
        input_ids = tok(args.prompt, return_tensors="pt").input_ids
    else:
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not torch.is_tensor(input_ids):  # some tokenizers return a dict-like
            input_ids = input_ids["input_ids"]

    input_ids = input_ids.to("cuda:0")
    runner.reset()  # forget the previous generation's history
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p,
            use_cache=True,
        )

    new_tokens = out[0, input_ids.shape[1]:]
    print("\n--- output ---")
    print(tok.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
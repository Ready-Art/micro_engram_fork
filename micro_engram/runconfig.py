"""Engram run setup shared by the training entry points (the Axolotl plugin and
micro_engram.train): which engrams a config asks for, building/attaching them, and the
micro_engram_config.json written next to the weights."""
import dataclasses
import json
import logging
import os
from collections.abc import Mapping

import torch

from .module import attach_engram, check_loaded, engram_from_spec, resolve_order_sizes
from .weights import (base_fingerprint, default_seed, load_engram_weights, norm_table_sizes,
                      resolve_engram_source)

LOG = logging.getLogger(__package__)

CONFIG_NAME = "micro_engram_config.json"


@dataclasses.dataclass
class _Slot:
    """One engram of the run."""
    spec: dict                    # settings as written to micro_engram_config.json
    engram: torch.nn.Module
    frozen: bool = False
    source: str | None = None     # weights it was loaded from (init_from), absolute
    where: str | None = None      # decoder layer's module name, e.g. model.language_model.layers.0

    @property
    def layer(self):
        return self.spec["layer"]

    def file_name(self, step=None):
        """Non-FSDP .pt name: the first engram keeps the old name, others get their layer."""
        base = "micro_engram" if self.spec.get("_main") else f"micro_engram_L{self.layer}"
        return f"{base}-step{step}.pt" if step is not None else f"{base}.pt"


def build_config(base, slots, trainable_weights):
    """micro_engram_config.json content: the first engram at the top level (older loaders
    read only that) plus every engram under "engrams", each with its own "weights".

    `trainable_weights(slot)` says where a trained engram's tensors are relative to the
    config's folder (None: inside the model checkpoint there). A frozen engram is not in
    trainable-only checkpoints, so it points to the file it was loaded from.
    """
    entries = []
    for slot in slots:
        entry = {k: v for k, v in slot.spec.items() if not k.startswith("_")}
        entry["weights"] = slot.source if slot.frozen else trainable_weights(slot)
        entries.append(entry)
    first = entries[0]
    conf = {**base, **{k: v for k, v in first.items() if k != "weights"}, "engrams": entries}
    if first["weights"]:
        conf["weights"] = first["weights"]
    return conf


def write_config(conf, folder, weights=None):
    """micro_engram_config.json in `folder`. `weights` (old single-engram form) sets the
    top-level "weights"; build_config output already has per-engram weights."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, CONFIG_NAME)
    with open(path, "w") as f:
        json.dump({**conf, "weights": weights} if weights else conf, f, indent=2)
    LOG.info(f"[micro_engram] wrote {path}")


def _params(engram):
    """Every parameter of an engram, including its row-sharded table rows."""
    get = getattr(engram, "all_parameters", None)
    return get() if get is not None else list(engram.parameters())


def _freeze_frozen(slots):
    for slot in slots:
        if slot.frozen:
            for p in _params(slot.engram):
                p.requires_grad_(False)


def _safe_fingerprint(path):
    try:
        return base_fingerprint(path) if path and os.path.isdir(path) else None
    except Exception as e:  # noqa: BLE001  -- never block training over this
        LOG.warning(f"[micro_engram] could not fingerprint the base weights: {e}")
        return None


def check_canon(canon, vocab_size):
    """Validate a token-merge map loaded from disk."""
    if not torch.is_tensor(canon) or canon.dim() != 1 or canon.numel() != vocab_size:
        raise ValueError(f"[micro_engram] canon file must be a 1-D tensor of length {vocab_size}")
    canon = canon.long()
    if int(canon.min()) < 0 or int(canon.max()) >= vocab_size:
        raise ValueError("[micro_engram] canon file has ids outside the vocabulary")
    if not torch.equal(canon[canon], canon):
        raise ValueError("[micro_engram] canon map is not idempotent (a canonical id must map to itself)")
    return canon


def epoch_step_scheduler(optimizer, num_training_steps, warmup_steps, epochs, factor):
    """Linear warmup from 0 (like Axolotl's schedulers), then factor ** epoch.

    Epoch boundaries come from the step count: num_training_steps / epochs steps each.
    Resuming restores the step, so the LR picks up where it was.
    """
    from torch.optim.lr_scheduler import LambdaLR

    per_epoch = max(1, round(num_training_steps / max(epochs, 1e-9)))

    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return step / warmup_steps
        return factor ** (step // per_epoch)

    drops = list(range(per_epoch, num_training_steps, per_epoch))
    LOG.warning(f"[micro_engram] LR schedule: warmup {warmup_steps} steps, then x{factor} at the start of each "
                f"epoch (steps {drops}), {per_epoch} steps per epoch")
    return LambdaLR(optimizer, lr_lambda)


def engram_requests(cfg):
    """[(spec settings, init_from, frozen)] for the main engram and every micro_engram_extra."""
    main = {
        "layer": cfg.micro_engram_layer,
        "table_size": cfg.micro_engram_table_size,
        "table_sizes": norm_table_sizes(cfg.micro_engram_table_sizes),
        "engram_dim": cfg.micro_engram_dim,
        "num_heads": cfg.micro_engram_heads,
        "gate": cfg.micro_engram_gate,
        "seed": cfg.micro_engram_seed,
        "_main": True,
    }
    reqs = [(main, cfg.micro_engram_init_from, bool(cfg.micro_engram_frozen))]
    for e in cfg.micro_engram_extra or []:
        # Axolotl hands these over as DictDefault, which answers every attribute with None:
        # check for a mapping first, never hasattr()
        if isinstance(e, Mapping):
            e = dict(e)
        elif callable(getattr(e, "model_dump", None)):
            e = e.model_dump()
        else:
            e = dict(vars(e))
        spec = {
            "layer": int(e["layer"]),
            "table_size": e.get("table_size") or main["table_size"],
            # own per-order sizes; with only its own table_size, one size for all; else the main's
            "table_sizes": norm_table_sizes(e.get("table_sizes")) if e.get("table_sizes")
            else (None if e.get("table_size") else main["table_sizes"]),
            "engram_dim": e.get("dim") or main["engram_dim"],
            "num_heads": e.get("heads") or main["num_heads"],
            "gate": e.get("gate") or main["gate"],
            "seed": e.get("seed"),
        }
        reqs.append((spec, e.get("init_from"), bool(e.get("frozen"))))
    layers = [r[0]["layer"] for r in reqs]
    if len(set(layers)) != len(layers):
        raise ValueError(f"[micro_engram] one engram per layer, got layers {layers}")
    for spec, init_from, frozen in reqs:
        if frozen and not init_from:
            raise ValueError(f"[micro_engram] the engram at layer {spec['layer']} is frozen but has no "
                             "init_from: a frozen untrained engram does nothing")
        if spec["seed"] is None:
            spec["seed"] = default_seed(spec["layer"])
    return reqs


_ARCH_KEYS = ("table_size", "table_sizes", "engram_dim", "num_heads", "orders", "gate", "seed")


def build_slot(spec, init_from, frozen, model, text_cfg, canon, dtype=None, sharding=None, get_ids=None):
    """Build one engram (fresh or from init_from), row-shard it if asked, attach it to its layer.

    dtype: "fp32" / "bf16" / None (the layer's). get_ids: see attach_engram."""
    pad = text_cfg.pad_token_id or 0
    source, sd = None, None
    if init_from:
        source, src_spec = resolve_engram_source(init_from, spec["layer"])
        source = os.path.abspath(source)
        if src_spec is not None:
            # rebuild it exactly as trained (sizes, gate, seed); the YAML can't change those
            def norm(v):
                if isinstance(v, dict):
                    return norm_table_sizes(v)
                return tuple(v) if isinstance(v, (list, tuple)) else v

            changed = {k: (spec[k], src_spec[k]) for k in _ARCH_KEYS
                       if spec.get(k) is not None and src_spec.get(k) is not None
                       and norm(spec[k]) != norm(src_spec[k])}
            if changed:
                LOG.warning(f"[micro_engram] layer {spec['layer']}: using the trained engram's "
                            f"settings, not the YAML's: " + ", ".join(
                                f"{k} {a} -> {b}" for k, (a, b) in changed.items()))
            spec.update({k: src_spec[k] for k in _ARCH_KEYS if src_spec.get(k) is not None})
            # its own per-order sizes, also "none" (one size for every table, older runs)
            spec["table_sizes"] = norm_table_sizes(src_spec.get("table_sizes"))
        sd = load_engram_weights(source, layer=spec["layer"])
        if not any(k.startswith("tables.") for k in sd):
            raise ValueError(f"[micro_engram] no engram weights for layer {spec['layer']} in {source}")
        t0 = sd["tables.0.weight"]
        if src_spec is None:  # bare weights: sizes from the tensors, the rest from the YAML
            spec["table_size"], spec["engram_dim"] = int(t0.shape[0]), int(t0.shape[1])
            rows = [int(sd[f"tables.{k}.weight"].shape[0])
                    for k in range(sum(1 for x in sd if x.startswith("tables.")))]
            orders = list(resolve_order_sizes(spec["table_size"], spec.get("orders") or (2, 3),
                                              spec.get("table_sizes")))
            heads = spec["num_heads"]
            if len(rows) != heads * len(orders):
                raise ValueError(f"[micro_engram] {source}: {len(rows)} tables, but {heads} heads x orders "
                                 f"{orders} need {heads * len(orders)}: set the sizes to match these weights")
            spec["orders"] = orders
            spec["table_sizes"] = {n: rows[i * heads] for i, n in enumerate(orders)}
            spec["gate"] = "ple" if "up.weight" in sd else (
                "scalar_sqrt" if "q_scale" in sd else "scalar")

    engram = engram_from_spec(spec, text_cfg.hidden_size, text_cfg.vocab_size, pad, canon,
                              build_tables=not (sharding and sd is None))
    if sd is not None:
        if "canon" in sd and canon is not None and not torch.equal(sd["canon"].long(), canon):
            LOG.warning(f"[micro_engram] layer {spec['layer']}: keeps the token-merge map it was "
                        f"trained with, which differs from micro_engram_canon_path")
        if "canon" not in sd and canon is None and spec.get("token_merge"):
            raise ValueError(f"[micro_engram] layer {spec['layer']} was trained with token merging "
                             f"but its weights have no canon; set micro_engram_canon_path")
        check_loaded(engram, sd, source)
        LOG.warning(f"[micro_engram] layer {spec['layer']}: loaded trained engram from {source}"
                    + (" (frozen)" if frozen else " (continues training)"))
    has_canon = not torch.equal(engram.canon, torch.arange(engram.canon.numel()))
    spec.update({
        "orders": list(engram.orders),
        # rows per order as built; table_size is None when they differ, so a loader that only
        # knows one size fails instead of building the wrong tables
        "table_sizes": dict(engram.order_sizes),
        "table_size": engram.table_size,
        "pad_id": engram.pad_id,
        "token_merge": has_canon,
        "frozen": frozen,
        "init_from": source,
    })
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(dtype)
    if sharding:
        from .module import find_text_layers, layer_device_dtype
        from .rows import dist_rank_world

        rank, world = dist_rank_world()
        dev = (torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if torch.cuda.is_available()
               else torch.device("cpu"))
        _, layer_dtype = layer_device_dtype(find_text_layers(model)[1][spec["layer"]])
        # Before attach_engram: that moves the engram to its layer's device, which is meta on
        # ranks > 0 with cpu_ram_efficient_loading (rank 0's state dict is broadcast later, but
        # the rows are not in it). The tables live outside FSDP, on this GPU from the start.
        engram.shard_rows(rank, world, device=dev, dtype=dtype or layer_dtype, requires_grad=not frozen)
    where, _ = attach_engram(model, engram, spec["layer"], dtype=dtype, get_ids=get_ids)
    n = sum(p.numel() for p in _params(engram))
    LOG.warning(f"[micro_engram] attached at {where}: {n / 1e6:.1f}M params, {spec['gate']} gate, "
                f"seed {spec['seed']}" + (", FROZEN" if frozen else ""))
    return _Slot(spec=spec, engram=engram, frozen=frozen, source=source, where=where)

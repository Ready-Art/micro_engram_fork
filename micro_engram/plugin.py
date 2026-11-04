import dataclasses
import json
import logging
import os

import torch
from transformers import TrainerCallback

from axolotl.integrations.base import BasePlugin

from .module import MicroEngram, attach_engram
from .stats import module_stats
from .weights import base_fingerprint

LOG = logging.getLogger(__package__)  # keeps the old "[micro_engram...]" logger name

_WRAP = "._checkpoint_wrapped_module."
CONFIG_NAME = "micro_engram_config.json"


def save_engram(engram, out_dir, fname="micro_engram.pt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    torch.save(engram.state_dict(), path)
    LOG.info(f"[micro_engram] saved {path}")


class _SaveEngramCallback(TrainerCallback):
    def __init__(self, engram, every):
        self.engram, self.every = engram, every

    def on_step_end(self, args, state, control, **kwargs):
        if self.every and state.global_step % self.every == 0 and state.is_world_process_zero:
            save_engram(self.engram, args.output_dir, f"micro_engram-step{state.global_step}.pt")


def write_config(conf, folder, weights=None):
    """micro_engram_config.json in `folder`; `weights` (relative to it) says where the
    engram tensors are, so vLLM / assemble_engram_model.py can load them from the config."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, CONFIG_NAME)
    with open(path, "w") as f:
        json.dump({**conf, "weights": weights} if weights else conf, f, indent=2)
    LOG.info(f"[micro_engram] wrote {path}")


class _WriteConfigCallback(TrainerCallback):
    """Writes the engram settings next to the output and next to every checkpoint."""

    def __init__(self, conf, checkpoint_weights=None):
        self.conf = conf
        self.checkpoint_weights = checkpoint_weights  # e.g. the FSDP checkpoint folder

    def _write(self, folder, weights=None):
        write_config(self.conf, folder, weights)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._write(args.output_dir)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self._write(os.path.join(args.output_dir, f"checkpoint-{state.global_step}"),
                        self.checkpoint_weights)


class _ReportTrainableCallback(TrainerCallback):
    """Logs the real trainable count at train start, after Axolotl has applied its freezing."""

    def __init__(self, engram):
        self.engram = engram

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_eng = sum(p.numel() for p in self.engram.parameters())
        n_eng_train = sum(p.numel() for p in self.engram.parameters() if p.requires_grad)
        LOG.warning(f"[micro_engram] at train start: {n_train / 1e6:.1f}M trainable in total, "
                    f"engram {n_eng_train / 1e6:.1f}M of {n_eng / 1e6:.1f}M trainable")
        if n_eng_train == 0:
            LOG.error("[micro_engram] the engram is FROZEN: check that unfrozen_parameters matches "
                      "the engram's parameter names (with an adapter, PEFT adds a 'base_model.model.' "
                      "prefix: use '.+micro_engram')")
        others = [n for n, p in model.named_parameters() if p.requires_grad and "micro_engram" not in n]
        if others:
            LOG.warning(f"[micro_engram] {len(others)} non-engram tensors are ALSO training "
                        f"(e.g. {others[0]}). Fine if intended (engram + LoRA); for an engram-only run "
                        f"on a 4-bit base, set unfrozen_parameters: ['.+micro_engram']")
        # module class, not parameter class: FSDP2 turns the weights into DTensors
        n_4bit = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
        if n_4bit:
            LOG.warning(f"[micro_engram] base is 4-bit ({n_4bit} quantized linear layers); the engram "
                        f"trains in {next(self.engram.parameters()).dtype} against it")


class _StatsCallback(TrainerCallback):
    """Every `every` steps, adds engram weight/gate stats to the training log line.

    Rank 0 only, from its own FSDP shard (a random sample of rows), so there is no
    communication and no risk of a hang. Never raises: stats must not stop a run.
    """

    def __init__(self, engram, every):
        self.engram, self.every = engram, every

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.every or not state.is_world_process_zero:
            return
        if logs is None or "loss" not in logs:
            return  # only training-step logs (not eval or the end-of-training summary)
        gate = self.engram.pop_gate_stats()  # pop every log so the mean covers the recent steps
        if state.global_step % self.every:
            return
        try:
            stats = module_stats(self.engram)
            if gate is not None:
                stats["engram/gate_mean"], stats["engram/gate_saturated"] = gate
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"[micro_engram] stats failed (training continues): {e}")
            return
        logs.update(stats)  # shown in the normal log line, and reaches wandb / tensorboard


def _safe_fingerprint(path):
    try:
        return base_fingerprint(path) if path and os.path.isdir(path) else None
    except Exception as e:  # noqa: BLE001  -- never block training over this
        LOG.warning(f"[micro_engram] could not fingerprint the base weights: {e}")
        return None


def _uses_fsdp(cfg):
    return bool(cfg.fsdp_config or cfg.fsdp)


def _check_canon(canon, vocab_size):
    """Validate a token-merge map loaded from disk."""
    if not torch.is_tensor(canon) or canon.dim() != 1 or canon.numel() != vocab_size:
        raise ValueError(f"[micro_engram] canon file must be a 1-D tensor of length {vocab_size}")
    canon = canon.long()
    if int(canon.min()) < 0 or int(canon.max()) >= vocab_size:
        raise ValueError("[micro_engram] canon file has ids outside the vocabulary")
    if not torch.equal(canon[canon], canon):
        raise ValueError("[micro_engram] canon map is not idempotent (a canonical id must map to itself)")
    return canon


def _strip_wrapper_prefix(state_dict, prefix, local_metadata, strict,
                          missing_keys, unexpected_keys, error_msgs):
    """(kept from the earlier attempt; harmless) rename wrapper-prefixed engram keys."""
    for k in list(state_dict.keys()):
        if "micro_engram" in k and _WRAP in k:
            state_dict[k.replace(_WRAP, ".")] = state_dict.pop(k)


def _is_engram_key(k):
    return isinstance(k, str) and "micro_engram" in k


def _find_param(model, key):
    """Walk the model by name to the engram tensor, ignoring any wrapper name in the key."""
    clean = key.replace(_WRAP, ".")
    parts = clean.split(".")
    obj = model
    for p in parts[:-1]:
        obj = getattr(obj, p)  # the checkpoint wrapper forwards unknown names to the layer inside
    return getattr(obj, parts[-1])


def _patch_accelerate_resume():
    """On resume, load the engram tensors directly and let PyTorch load everything else.

    PyTorch's own key renaming breaks for the engram (it re-adds the checkpoint-wrapper
    name), so the engram entries are taken out of the state dict, copied straight into
    the engram weights, and the rest is loaded non-strictly. Any problem with a
    non-engram key still raises.

    `mults` is rebuilt from a seed, so it is skipped. `canon` (token merging) comes from
    the file named in the YAML, so it is not copied but CHECKED against the checkpoint:
    if they differ, the run would silently train on different hashes, so it raises.
    """
    try:
        from accelerate.utils import fsdp_utils
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"[micro_engram] could not import accelerate fsdp_utils, resume patch skipped: {e}")
        return
    orig = getattr(fsdp_utils, "_set_model_state_dict", None)
    if orig is None:
        LOG.warning("[micro_engram] accelerate has no _set_model_state_dict, resume patch skipped")
        return
    if getattr(orig, "_micro_engram_patched", False):
        return

    def patched(model, state_dict, *args, **kwargs):
        if not isinstance(state_dict, dict):
            return orig(model, state_dict, *args, **kwargs)
        engram_items = {k: v for k, v in state_dict.items() if _is_engram_key(k)}
        if not engram_items:
            return orig(model, state_dict, *args, **kwargs)
        rest = {k: v for k, v in state_dict.items() if not _is_engram_key(k)}

        loaded, expected = 0, 0
        for k, v in engram_items.items():
            if not (k.endswith(".mults") or k.endswith(".canon")):
                expected += 1
            if k.endswith(".mults"):
                continue  # integer hash multipliers: rebuilt from the seed, identical every run
            target = _find_param(model, k)
            if k.endswith(".canon"):
                try:
                    same = torch.equal(v.detach().cpu(), target.detach().cpu())
                except Exception:  # noqa: BLE001
                    same = True  # could not compare (unusual tensor type): do not block the resume
                if not same:
                    raise RuntimeError("[micro_engram] resume: the token-merge map in the checkpoint differs "
                                       "from micro_engram_canon_path in the YAML. Use the same canon file.")
                continue
            with torch.no_grad():
                target.copy_(v)
            loaded += 1
        example = next(iter(engram_items))
        LOG.warning(f"[micro_engram] resume: copied {loaded} engram tensors directly "
                    f"(expected {expected}; example checkpoint key: {example})")

        if not rest:
            return None  # trainable-only checkpoint: everything else comes from the base weights

        opts = kwargs.get("sd_options")
        if opts is not None:
            kwargs["sd_options"] = dataclasses.replace(opts, strict=False)
        result = orig(model, rest, *args, **kwargs)

        if result is not None:
            # With trainable-only checkpoints the frozen weights are expected to be missing
            # (they come from base_model); keys the model does not have are always an error.
            missing = [] if _TRAINABLE_ONLY["on"] else list(getattr(result, "missing_keys", []) or [])
            problems = [k for k in missing + list(getattr(result, "unexpected_keys", []) or [])
                        if "micro_engram" not in k]
            if problems:
                raise RuntimeError(f"[micro_engram] resume: non-engram keys did not match: {problems[:10]}")
        return result

    patched._micro_engram_patched = True
    fsdp_utils._set_model_state_dict = patched
    LOG.warning("[micro_engram] resume patch installed")


_TRAINABLE_ONLY = {"on": False}


def _patch_save_trainable_only():
    """FSDP2 checkpoints keep only trainable weights (the engram), not the frozen model.

    Accelerate builds the state-dict options for both saving and resuming in
    _prepare_sd_options; adding ignore_frozen_params makes checkpoints ~0.5-3 GB instead
    of the whole model, and resuming then reads only those weights back (the frozen
    ones come from base_model, as at the start of training). Older full checkpoints
    still resume: only their trainable part is read.
    """
    try:
        from accelerate.utils import fsdp_utils
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"[micro_engram] could not import accelerate fsdp_utils, full checkpoints kept: {e}")
        return
    orig = getattr(fsdp_utils, "_prepare_sd_options", None)
    if orig is None:
        LOG.warning("[micro_engram] accelerate has no _prepare_sd_options, full checkpoints kept")
        return
    if not getattr(orig, "_micro_engram_patched", False):
        def patched(fsdp_plugin):
            opts = orig(fsdp_plugin)
            return None if opts is None else dataclasses.replace(opts, ignore_frozen_params=True)

        patched._micro_engram_patched = True
        fsdp_utils._prepare_sd_options = patched
    _TRAINABLE_ONLY["on"] = True
    LOG.warning("[micro_engram] checkpoints keep only trainable weights; the merged output is NOT "
                "a full model: build a servable folder with assemble_engram_model.py")


class MicroEngramPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.engram = None
        self.conf = None

    def get_input_args(self):
        # Relative to this package, so it resolves however the package was imported
        # (pip-installed `micro_engram`, or the repo folder as `micro_engram.micro_engram`).
        return f"{__package__}.args.MicroEngramArgs"

    def post_model_load(self, cfg, model):
        text_cfg = getattr(model.config, "text_config", model.config)
        vocab_size = text_cfg.vocab_size

        canon = None
        if cfg.micro_engram_canon_path:
            canon = _check_canon(torch.load(cfg.micro_engram_canon_path, map_location="cpu"), vocab_size)
            classes = int(torch.unique(canon).numel())
            LOG.warning(f"[micro_engram] token merging ON: {canon.numel():,} ids -> {classes:,} classes "
                        f"({cfg.micro_engram_canon_path})")
        else:
            LOG.warning("[micro_engram] token merging OFF (no micro_engram_canon_path)")

        self.engram = MicroEngram(
            model_dim=text_cfg.hidden_size,  # 5376 for Gemma 4 31B
            table_size=cfg.micro_engram_table_size,
            engram_dim=cfg.micro_engram_dim,
            num_heads=cfg.micro_engram_heads,
            pad_id=text_cfg.pad_token_id or 0,
            vocab_size=vocab_size,
            canon=canon,
            gate=cfg.micro_engram_gate,
        )
        LOG.warning(f"[micro_engram] gate: {cfg.micro_engram_gate}")
        where, _ = attach_engram(model, self.engram, cfg.micro_engram_layer)

        self.conf = {
            "layer": cfg.micro_engram_layer,
            "table_size": cfg.micro_engram_table_size,
            "engram_dim": cfg.micro_engram_dim,
            "num_heads": cfg.micro_engram_heads,
            "orders": list(self.engram.orders),
            "model_dim": text_cfg.hidden_size,
            "pad_id": text_cfg.pad_token_id or 0,
            "vocab_size": vocab_size,
            "token_merge": canon is not None,
            "canon_path": cfg.micro_engram_canon_path,
            # which base weights the engram belongs to (folder-name independent), so loaders
            # can warn when it is put on a different model
            "base_model": cfg.base_model,
            "base_fingerprint": _safe_fingerprint(cfg.base_model),
            "gate": cfg.micro_engram_gate,
        }

        model._register_load_state_dict_pre_hook(_strip_wrapper_prefix)
        _patch_accelerate_resume()
        if cfg.micro_engram_save_trainable_only:
            fsdp_version = cfg.fsdp_version or (cfg.fsdp_config or {}).get("fsdp_version")
            if _uses_fsdp(cfg) and str(fsdp_version) == "2":
                _patch_save_trainable_only()
            else:
                LOG.warning("[micro_engram] micro_engram_save_trainable_only needs FSDP2; ignored "
                            "(without FSDP the engram is also saved on its own as micro_engram.pt)")

        # If unfrozen_parameters / an adapter controls freezing, leave it to Axolotl
        # (add the engram to unfrozen_parameters). Otherwise freeze Gemma here.
        if not cfg.unfrozen_parameters and not cfg.adapter:
            for p in model.parameters():
                p.requires_grad_(False)
            for p in self.engram.parameters():
                p.requires_grad_(True)

        n_eng = sum(p.numel() for p in self.engram.parameters())
        LOG.warning(f"[micro_engram] attached at {where}: {n_eng / 1e6:.1f}M engram params "
                    f"(trainable count is logged at train start, after Axolotl's freezing)")

    def add_callbacks_post_trainer(self, cfg, trainer):
        # FSDP checkpoints hold the engram in their DCP folder (Accelerate's name for model 0)
        ckpt_weights = "pytorch_model_fsdp_0" if _uses_fsdp(cfg) else None
        callbacks = [_WriteConfigCallback(self.conf, ckpt_weights), _ReportTrainableCallback(self.engram)]
        if cfg.micro_engram_stats_steps:
            # first in line, so the stats are in `logs` before the printer/loggers see it
            trainer.callback_handler.callbacks.insert(
                0, _StatsCallback(self.engram, cfg.micro_engram_stats_steps))
        if not _uses_fsdp(cfg):  # under FSDP the checkpoint already contains the engram
            callbacks.append(_SaveEngramCallback(self.engram, cfg.micro_engram_save_steps))
        return callbacks

    def post_train(self, cfg, model):
        if _uses_fsdp(cfg):
            LOG.info("[micro_engram] FSDP run: each checkpoint's micro_engram_config.json points to its "
                     "engram; load it in vLLM with --additional-config, or use assemble_engram_model.py")
            return
        save_engram(self.engram, cfg.output_dir)
        write_config(self.conf, cfg.output_dir, "micro_engram.pt")
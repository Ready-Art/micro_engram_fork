import dataclasses
import json
import logging
import os
from collections.abc import Mapping

import torch
from transformers import TrainerCallback

from axolotl.integrations.base import BasePlugin

from .module import MicroEngram, attach_engram, check_loaded, engram_from_spec, resolve_order_sizes
from .stats import module_stats
from .weights import (base_fingerprint, default_seed, load_engram_weights, norm_table_sizes,
                      resolve_engram_source)

LOG = logging.getLogger(__package__)  # keeps the old "[micro_engram...]" logger name

_WRAP = "._checkpoint_wrapped_module."
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


def save_engram(engram, out_dir, fname="micro_engram.pt"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    torch.save(engram.state_dict(), path)
    LOG.info(f"[micro_engram] saved {path}")


class _SaveEngramCallback(TrainerCallback):
    def __init__(self, slots, every):
        self.slots, self.every = slots, every

    def on_step_end(self, args, state, control, **kwargs):
        if self.every and state.global_step % self.every == 0 and state.is_world_process_zero:
            for slot in self.slots:
                if not slot.frozen:
                    save_engram(slot.engram, args.output_dir, slot.file_name(state.global_step))


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


class _WriteConfigCallback(TrainerCallback):
    """Writes the engram settings next to the output and next to every checkpoint."""

    def __init__(self, base, slots, checkpoint_weights=None):
        self.base, self.slots = base, slots
        self.checkpoint_weights = checkpoint_weights  # e.g. the FSDP checkpoint folder

    def on_train_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            write_config(build_config(self.base, self.slots, lambda s: None), args.output_dir)

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            ckpt = f"checkpoint-{state.global_step}"
            conf = build_config(self.base, self.slots, lambda s: self.checkpoint_weights)
            write_config(conf, os.path.join(args.output_dir, ckpt))
            if self.checkpoint_weights:
                # the output_dir copy (written at train start with no weights yet) now points to
                # the newest checkpoint, so it can be served directly too
                write_config(build_config(self.base, self.slots,
                                          lambda s: f"{ckpt}/{self.checkpoint_weights}"), args.output_dir)


def _params(engram):
    """Every parameter of an engram, including its row-sharded table rows."""
    get = getattr(engram, "all_parameters", None)
    return get() if get is not None else list(engram.parameters())


def _freeze_frozen(slots):
    for slot in slots:
        if slot.frozen:
            for p in _params(slot.engram):
                p.requires_grad_(False)


class _ReportTrainableCallback(TrainerCallback):
    """Logs the real trainable count at train start, after Axolotl has applied its freezing."""

    def __init__(self, slots, dtype=None):
        self.slots, self.dtype = slots, dtype

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if not state.is_world_process_zero or model is None:
            return
        dtypes = {str(next(s.engram.parameters()).dtype).replace("torch.", "") for s in self.slots}
        want = {"fp32": "float32", "bf16": "bfloat16"}.get(self.dtype)
        LOG.warning(f"[micro_engram] engram weights are {', '.join(sorted(dtypes))}"
                    + (f" (micro_engram_dtype: {self.dtype})" if self.dtype else ""))
        if want and dtypes != {want}:
            LOG.error(f"[micro_engram] micro_engram_dtype is {self.dtype} but the engram weights are "
                      f"{', '.join(sorted(dtypes))}: something cast them after loading")
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_train += sum(p.numel() for s in self.slots for p in (s.engram.row_shards or []) if p.requires_grad)
        parts, n_eng_train = [], 0
        for slot in self.slots:
            n = sum(p.numel() for p in _params(slot.engram))
            t = sum(p.numel() for p in _params(slot.engram) if p.requires_grad)
            n_eng_train += t
            parts.append(f"layer {slot.layer}: {t / 1e6:.1f}M of {n / 1e6:.1f}M"
                         + (" (frozen)" if slot.frozen else ""))
        LOG.warning(f"[micro_engram] at train start: {n_train / 1e6:.1f}M trainable in total; "
                    f"engram {', '.join(parts)}")
        for slot in self.slots:
            info = slot.engram.row_info
            if info is not None:
                eng, seen, parts = slot.engram, set(), []
                for k, n in enumerate(eng._table_orders):
                    if n not in seen:
                        seen.add(n)
                        a = info[2][k]
                        parts.append(f"{n}-grams {a:,}..{a + eng.row_shards[k].shape[0] - 1:,} of {eng.table_rows[k]:,}")
                LOG.warning(f"[micro_engram] layer {slot.layer}: tables row-sharded, this GPU holds rows "
                            f"{'; '.join(parts)} (counts above are this GPU's share)")
            t = sum(p.numel() for p in _params(slot.engram) if p.requires_grad)
            if slot.frozen and t:
                LOG.error(f"[micro_engram] engram at layer {slot.layer} should be frozen but is training")
            if not slot.frozen and t == 0:
                LOG.error(f"[micro_engram] the engram at layer {slot.layer} is FROZEN: check that "
                          "unfrozen_parameters matches the engram's parameter names (with an adapter, "
                          "PEFT adds a 'base_model.model.' prefix: use '.+micro_engram')")
        others = [n for n, p in model.named_parameters() if p.requires_grad and "micro_engram" not in n]
        if others:
            LOG.warning(f"[micro_engram] {len(others)} non-engram tensors are ALSO training "
                        f"(e.g. {others[0]}). Fine if intended (engram + LoRA); for an engram-only run "
                        f"on a 4-bit base, set unfrozen_parameters: ['.+micro_engram']")
        # module class, not parameter class: FSDP2 turns the weights into DTensors
        n_4bit = sum(1 for m in model.modules() if type(m).__name__ == "Linear4bit")
        if n_4bit:
            LOG.warning(f"[micro_engram] base is 4-bit ({n_4bit} quantized linear layers)")


class _StatsCallback(TrainerCallback):
    """Every `every` steps, adds engram weight/gate stats to the training log line.

    Rank 0 only, from its own FSDP shard (a random sample of rows), so there is no
    communication and no risk of a hang. Never raises: stats must not stop a run.
    One engram: keys "engram/..."; several: "engram_L<layer>/...".
    """

    def __init__(self, slots, every, rows=None):
        self.slots, self.every, self.rows = slots, every, rows

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self.every or not state.is_world_process_zero:
            return
        if logs is None or "loss" not in logs:
            return  # only training-step logs (not eval or the end-of-training summary)
        gates = [s.engram.pop_gate_stats() for s in self.slots]  # pop every log: recent steps only
        if state.global_step % self.every:
            return
        for slot, gate in zip(self.slots, gates):
            if slot.frozen:
                continue  # unchanged since it was loaded
            try:
                stats = module_stats(slot.engram)
                if gate is not None:
                    stats["engram/gate_mean"], stats["engram/gate_saturated"] = gate
            except Exception as e:  # noqa: BLE001
                LOG.warning(f"[micro_engram] stats failed (training continues): {e}")
                continue
            if len(self.slots) > 1:
                stats = {k.replace("engram/", f"engram_L{slot.layer}/", 1): v for k, v in stats.items()}
            logs.update(stats)  # shown in the normal log line, and reaches wandb / tensorboard
        rows = self.rows() if callable(self.rows) else self.rows
        if rows is not None and rows.last_grad_norm is not None:
            logs["engram/table_grad_norm"] = rows.last_grad_norm  # tables are not in grad_norm


class _MemorySnapshotCallback(TrainerCallback):
    """Records CUDA allocations for `steps` optimizer steps on every rank, then dumps a
    snapshot per rank for https://pytorch.org/memory_viz. Never stops the run on error."""

    def __init__(self, steps):
        self.steps, self.start, self.done = steps, None, False

    def on_train_begin(self, args, state, control, **kwargs):
        if not torch.cuda.is_available():
            LOG.warning("[micro_engram] memory snapshot needs CUDA; skipped")
            self.done = True
            return
        self.start = state.global_step
        torch.cuda.memory._record_memory_history(max_entries=200_000)
        if state.is_world_process_zero:
            LOG.warning(f"[micro_engram] recording CUDA memory for {self.steps} step(s) (slower steps)")

    def on_step_end(self, args, state, control, **kwargs):
        if self.done or self.start is None or state.global_step - self.start < self.steps:
            return
        self.done = True
        rank = int(os.environ.get("RANK", 0))
        folder = os.path.join(args.output_dir, "memory_snapshot")
        path = os.path.join(folder, f"rank{rank}_step{state.global_step}.pickle")
        try:
            os.makedirs(folder, exist_ok=True)
            torch.cuda.memory._dump_snapshot(path)
            LOG.warning(f"[micro_engram] memory snapshot written: {path} (open it at https://pytorch.org/memory_viz); "
                        f"you can stop this run now")
        except Exception as e:  # noqa: BLE001
            LOG.warning(f"[micro_engram] memory snapshot failed (training continues): {e}")
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)


def _install_lazy_adam(trainer):
    """Make the trainer build LazyAdamW when it creates its optimizer.

    It must stay lazy: with FSDP the Trainer creates the optimizer only after sharding,
    which replaces every parameter object, so an optimizer built earlier would hold stale
    parameters. Engram tables get lazy row updates; other trainable params plain AdamW.
    """
    import types

    from .optim import LazyAdamW

    if getattr(trainer, "_micro_engram_lazy_adam", False):
        return

    def create_optimizer(self, model=None):
        if self.optimizer is None:
            lazy, dense = [], []
            for name, p in (model or self.model).named_parameters():
                if p.requires_grad:
                    (lazy if ".micro_engram.tables." in name.replace(_WRAP, ".") else dense).append(p)
            groups = [g for g in ({"params": lazy, "lazy": True}, {"params": dense, "lazy": False}) if g["params"]]
            a = self.args
            self.optimizer = LazyAdamW(groups, lr=a.learning_rate, betas=(a.adam_beta1, a.adam_beta2),
                                       eps=a.adam_epsilon, weight_decay=a.weight_decay)
            tables = (f"{sum(p.numel() for p in lazy) / 1e6:.1f}M table params with lazy row updates" if lazy
                      else "row-sharded tables have their own optimizer")
            LOG.warning(f"[micro_engram] optimizer: LazyAdamW ({tables}, {sum(p.numel() for p in dense) / 1e6:.1f}M "
                        f"as AdamW); the YAML `optimizer` is not used")
        return self.optimizer

    trainer.create_optimizer = types.MethodType(create_optimizer, trainer)
    trainer._micro_engram_lazy_adam = True


class _RowShardCallback(TrainerCallback):
    """Row-sharded tables: clip and step them with the Trainer's optimizer step, save the
    final ones at the end of training (checkpoints are saved by the save_model wrapper)."""

    def __init__(self, rows, trainer, on_final):
        self.rows, self.trainer, self.on_final = rows, trainer, on_final

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        self.rows.clip(args.max_grad_norm)  # after the Trainer clipped the rest

    def on_optimizer_step(self, args, state, control, **kwargs):
        # this step's LR (the scheduler steps right after), relative to the Trainer's first group
        g0 = self.trainer.optimizer.param_groups[0]
        base = g0.get("initial_lr") or args.learning_rate
        self.rows.step(g0["lr"] * args.learning_rate / base if base else g0["lr"])

    def on_train_end(self, args, state, control, **kwargs):
        from .rows import ROWS_DIR

        self.rows.save(os.path.join(args.output_dir, ROWS_DIR))
        if state.is_world_process_zero:
            self.on_final(args.output_dir)


def _install_row_sharding(trainer, slots, lazy, on_final, rowwise=False):
    """Table optimizer + checkpoint save/resume for row-sharded tables. Returns the callback.

    Checkpoints: Trainer._save_checkpoint calls save_model on every rank before it writes
    trainer_state.json (what marks a checkpoint complete), so the rows are saved there.
    Resume: _load_optimizer_and_scheduler gets the checkpoint path on every rank.
    """
    import types

    from .rows import ROWS_DIR, ROWS_OPTIM_DIR, RowTables

    if getattr(trainer, "_micro_engram_rows", None) is not None:
        return trainer._micro_engram_rows
    a = trainer.args
    rows = RowTables([(f"{s.where}.micro_engram.", s.engram) for s in slots],
                     lr=a.learning_rate, betas=(a.adam_beta1, a.adam_beta2), eps=a.adam_epsilon,
                     weight_decay=a.weight_decay, lazy=lazy, rowwise=rowwise)
    orig_save, orig_load = trainer.save_model, trainer._load_optimizer_and_scheduler

    def save_model(self, output_dir=None, _internal_call=False):
        out = orig_save(output_dir, _internal_call=_internal_call)
        folder = output_dir or self.args.output_dir
        if os.path.basename(os.path.normpath(folder)).startswith("checkpoint-"):
            rows.save(os.path.join(folder, ROWS_DIR), os.path.join(folder, ROWS_OPTIM_DIR))
        return out

    def load_optimizer_and_scheduler(self, checkpoint):
        out = orig_load(checkpoint)
        if checkpoint:
            src = os.path.join(checkpoint, ROWS_DIR)
            if not os.path.isdir(src):  # a checkpoint from a run without row sharding
                src = os.path.join(checkpoint, "pytorch_model_fsdp_0")
            if not os.path.isdir(src):
                raise RuntimeError(f"[micro_engram] resume: no engram tables in {checkpoint} "
                                   f"(neither {ROWS_DIR}/ nor pytorch_model_fsdp_0/)")
            n, opt = rows.load(src, os.path.join(checkpoint, ROWS_OPTIM_DIR))
            LOG.warning(f"[micro_engram] resume: loaded {n} row-sharded tables from {src}"
                        + ("" if opt else "; NO table optimizer state there, the tables' Adam restarts")
                        + (f"; second moment of {rows.converted} tables converted from the other optimizer "
                           f"(lazy_adam <-> rowwise_adam)" if rows.converted else ""))
        return out

    trainer.save_model = types.MethodType(save_model, trainer)
    trainer._load_optimizer_and_scheduler = types.MethodType(load_optimizer_and_scheduler, trainer)
    trainer._micro_engram_rows = _RowShardCallback(rows, trainer, on_final)
    n = sum(p.numel() for p in rows.params())
    kind = "partial row-wise Adam (one fp32 second moment per row)" if rowwise else (
        "lazy AdamW" if lazy else "AdamW")
    LOG.warning(f"[micro_engram] row-sharded tables: {n / 1e6:.1f}M rows' params on this GPU, own optimizer "
                f"({kind}) and clipping (logged as engram/table_grad_norm)")
    return trainer._micro_engram_rows


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
            if k.rsplit(".", 1)[-1] in MicroEngram.DELTA_SCALES:
                v = v - 1  # checkpoints hold the scale; the parameter holds scale - 1
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
        if kwargs.get("adapter_only") and opts is not None:
            # adapter run: the rest are trainable parameters under their full names (see
            # _patch_adapter_only_checkpoints); PEFT's own loader expects adapter-style names
            from torch.distributed.checkpoint.state_dict import set_model_state_dict

            result = set_model_state_dict(model, rest, options=kwargs["sd_options"])
        else:
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


def _norm_key(k):
    return k.replace(_WRAP, ".")


def _has_engram(model):
    return any("micro_engram" in n for n, _ in model.named_parameters())


def _patch_adapter_only_checkpoints():
    """With an adapter (e.g. the placeholder QLoRA for a 4-bit base), FSDP checkpoints must
    still hold the engram.

    The Trainer saves and resumes PEFT models with adapter_only=True. For sharded
    checkpoints Accelerate then uses PEFT's get_peft_model_state_dict, which returns
    only the LoRA, so the engram was never saved (a ~100 KB checkpoint). Accelerate's
    adapter-only *full* state dict path already means "every trainable parameter"
    (ignore_frozen_params); use that for sharded checkpoints too when the model has an
    engram. Loading builds its template with the same function, so resume reads the
    engram back.
    """
    try:
        from accelerate.utils import fsdp_utils
    except Exception as e:  # noqa: BLE001
        LOG.warning(f"[micro_engram] could not import accelerate fsdp_utils: {e}. With an adapter, "
                    f"checkpoints may NOT contain the engram")
        return
    orig = getattr(fsdp_utils, "_get_model_state_dict", None)
    if orig is None:
        LOG.warning("[micro_engram] accelerate has no _get_model_state_dict: with an adapter, "
                    "checkpoints may NOT contain the engram")
        return
    if getattr(orig, "_micro_engram_patched", False):
        return

    def patched(model, adapter_only=False, sd_options=None):
        if adapter_only and sd_options is not None and _has_engram(model):
            from torch.distributed.checkpoint.state_dict import get_model_state_dict

            sd = get_model_state_dict(model, options=dataclasses.replace(sd_options, ignore_frozen_params=True))
            # ignore_frozen_params skips frozen *parameters* only. Keep the engram (with its
            # buffers) and trainable parameters; drop the rest, e.g. bitsandbytes 4-bit
            # quant state (absmax, quant_map, ...) that a 4-bit base adds to its state dict.
            # The template for resuming is built here too, so old checkpoints that still
            # hold those entries resume fine (only the requested keys are read).
            trainable = {_norm_key(n) for n, p in model.named_parameters() if p.requires_grad}
            return {k: v for k, v in sd.items() if "micro_engram" in k or _norm_key(k) in trainable}
        return orig(model, adapter_only=adapter_only, sd_options=sd_options)

    patched._micro_engram_patched = True
    fsdp_utils._get_model_state_dict = patched
    LOG.warning("[micro_engram] adapter run: checkpoints will hold the engram (all trainable weights), "
                "not only the LoRA")


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


def _engram_requests(cfg):
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


class MicroEngramPlugin(BasePlugin):
    def __init__(self):
        super().__init__()
        self.slots = []
        self.base_conf = None
        self.dtype = None
        self.sharding = None
        self.rows_cb = None

    @property
    def engram(self):  # the main engram (older code paths)
        return self.slots[0].engram if self.slots else None

    def get_input_args(self):
        # Relative to this package, so it resolves however the package was imported
        # (pip-installed `micro_engram`, or the repo folder as `micro_engram.micro_engram`).
        return f"{__package__}.args.MicroEngramArgs"

    def _build_slot(self, spec, init_from, frozen, model, text_cfg, canon):
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
                                  build_tables=not (self.sharding and sd is None))
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
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}.get(self.dtype)
        if self.sharding:
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
        where, _ = attach_engram(model, engram, spec["layer"], dtype=dtype)
        n = sum(p.numel() for p in _params(engram))
        LOG.warning(f"[micro_engram] attached at {where}: {n / 1e6:.1f}M params, {spec['gate']} gate, "
                    f"seed {spec['seed']}" + (", FROZEN" if frozen else ""))
        return _Slot(spec=spec, engram=engram, frozen=frozen, source=source, where=where)

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

        self.dtype = cfg.micro_engram_dtype
        self.sharding = cfg.micro_engram_sharding
        if self.sharding and not _uses_fsdp(cfg):
            raise ValueError("[micro_engram] micro_engram_sharding: rows needs FSDP (fsdp_config)")
        if cfg.micro_engram_optimizer == "rowwise_adam" and not self.sharding:
            raise ValueError("[micro_engram] micro_engram_optimizer: rowwise_adam needs micro_engram_sharding: rows "
                             "(the tables then have their own optimizer)")
        self.slots = []
        for spec, init_from, frozen in _engram_requests(cfg):
            spec["canon_path"] = cfg.micro_engram_canon_path
            self.slots.append(self._build_slot(spec, init_from, frozen, model, text_cfg, canon))

        self.base_conf = {
            "model_dim": text_cfg.hidden_size,
            "vocab_size": vocab_size,
            "canon_path": cfg.micro_engram_canon_path,
            # which base weights the engram belongs to (folder-name independent), so loaders
            # can warn when it is put on a different model
            "base_model": cfg.base_model,
            "base_fingerprint": _safe_fingerprint(cfg.base_model),
        }

        model._register_load_state_dict_pre_hook(_strip_wrapper_prefix)
        _patch_accelerate_resume()
        if cfg.adapter and _uses_fsdp(cfg):
            _patch_adapter_only_checkpoints()
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
            for slot in self.slots:
                for p in slot.engram.parameters():
                    p.requires_grad_(True)
        _freeze_frozen(self.slots)
        if all(s.frozen for s in self.slots):
            LOG.warning("[micro_engram] every engram is frozen: nothing of the engrams will train")
        LOG.warning("[micro_engram] trainable count is logged at train start, after Axolotl's freezing")

    def _row_sharding(self, cfg, trainer):
        if not self.sharding or self.rows_cb is not None:
            return None
        from .rows import ROWS_DIR

        def on_final(output_dir):
            write_config(build_config(self.base_conf, self.slots, lambda s: ROWS_DIR), output_dir)

        opt = cfg.micro_engram_optimizer
        self.rows_cb = _install_row_sharding(trainer, self.slots, opt in ("lazy_adam", "rowwise_adam"), on_final,
                                             rowwise=opt == "rowwise_adam")
        return self.rows_cb

    def post_trainer_create(self, cfg, trainer):
        # Axolotl applies unfrozen_parameters after post_model_load, and '.+micro_engram'
        # matches every engram: freeze the frozen ones again before the optimizer is built.
        _freeze_frozen(self.slots)
        if cfg.micro_engram_optimizer == "lazy_adam":
            _install_lazy_adam(trainer)
        cb = self._row_sharding(cfg, trainer)  # normally already done in add_callbacks_post_trainer
        if cb is not None:
            trainer.add_callback(cb)

    def add_callbacks_post_trainer(self, cfg, trainer):
        _freeze_frozen(self.slots)  # in case post_trainer_create is not called by this Axolotl
        if cfg.micro_engram_optimizer == "lazy_adam":
            _install_lazy_adam(trainer)
        rows_cb = self._row_sharding(cfg, trainer)
        # FSDP checkpoints hold the engram in their DCP folder (Accelerate's name for model 0);
        # with row sharding the complete engram is in micro_engram_rows/
        ckpt_weights = "pytorch_model_fsdp_0" if _uses_fsdp(cfg) else None
        if self.sharding:
            from .rows import ROWS_DIR

            ckpt_weights = ROWS_DIR
        callbacks = [_WriteConfigCallback(self.base_conf, self.slots, ckpt_weights),
                     _ReportTrainableCallback(self.slots, self.dtype)]
        if cfg.micro_engram_stats_steps:
            # first in line, so the stats are in `logs` before the printer/loggers see it
            trainer.callback_handler.callbacks.insert(
                0, _StatsCallback(self.slots, cfg.micro_engram_stats_steps,
                                  rows=lambda: self.rows_cb.rows if self.rows_cb else None))
        if not _uses_fsdp(cfg):  # under FSDP the checkpoint already contains the engram
            callbacks.append(_SaveEngramCallback(self.slots, cfg.micro_engram_save_steps))
        if cfg.micro_engram_memory_snapshot:
            callbacks.append(_MemorySnapshotCallback(int(cfg.micro_engram_memory_snapshot)))
        if rows_cb is not None:
            callbacks.append(rows_cb)
        return callbacks

    def create_lr_scheduler(self, cfg, trainer, optimizer, num_training_steps):
        """micro_engram_lr_epoch_decay: warmup, then LR x factor at the start of each epoch."""
        factor = cfg.micro_engram_lr_epoch_decay
        if not factor:
            return None  # Axolotl's own lr_scheduler
        return epoch_step_scheduler(optimizer, num_training_steps,
                                    trainer.args.get_warmup_steps(num_training_steps),
                                    float(cfg.num_epochs or trainer.args.num_train_epochs or 1), factor)

    def post_train(self, cfg, model):
        if _uses_fsdp(cfg):
            LOG.info("[micro_engram] FSDP run: each checkpoint's micro_engram_config.json points to its "
                     "engram; load it in vLLM with --additional-config, or use assemble_engram_model.py")
            return
        for slot in self.slots:
            if not slot.frozen:
                save_engram(slot.engram, cfg.output_dir, slot.file_name())
        write_config(build_config(self.base_conf, self.slots, lambda s: s.file_name()), cfg.output_dir)

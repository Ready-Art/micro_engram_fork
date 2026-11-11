"""YAML config of micro_engram.train.

Same key names as the Axolotl configs the engram used to train with, so an existing
YAML works after deleting the placeholder-LoRA lines. Keys that only meant something
to Axolotl are accepted and listed in one warning; any other unknown key is an error
(nothing is silently ignored).
"""
import logging
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, model_validator

from ..args import MicroEngramArgs

LOG = logging.getLogger("micro_engram.train")

# Axolotl keys with no meaning here: accepted, listed in one warning line
IGNORED_KEYS = {
    "plugins", "strict", "tokenizer_type", "trust_remote_code", "bf16", "fp16", "save_safetensors",
    "eval_strategy", "freeze_mm_modules", "pad_to_sequence_len", "micro_engram_save_trainable_only",
    "micro_engram_save_steps", "train_on_inputs", "fsdp", "eval_sample_packing", "special_tokens",
    "chat_template", "flash_attention", "sdp_attention", "xformers_attention", "fsdp_version",
}
IGNORED_FSDP_KEYS = {
    "sync_module_states", "auto_wrap_policy", "transformer_layer_cls_to_wrap", "state_dict_type",
    "fsdp_version", "use_orig_params", "limit_all_gathers",
}
# Fused-loss switches: micro_engram.train always uses its own chunked vocab-parallel loss
# (full logits never exist), so these are accepted and have nothing to switch
LOSS_KEYS = {"cut_cross_entropy", "liger_cross_entropy", "liger_fused_linear_cross_entropy",
             "chunked_cross_entropy", "chunked_cross_entropy_num_chunks"}
# The placeholder QLoRA a 4-bit base needed under Axolotl: gone, and must not come back
LORA_KEYS = {"adapter", "unfrozen_parameters"}
LORA_PREFIX = ("lora_", "peft_")


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    type: Literal["completion", "chat_template"]
    field: Optional[str] = None                       # completion: text field (default "text")
    field_messages: Optional[str] = None              # chat: list of turns (default "messages")
    message_property_mappings: Optional[Dict[str, str]] = None  # {role: "from", content: "value", ...}
    roles: Optional[Dict[str, List[str]]] = None      # our role -> names used in the file
    roles_to_train: Optional[List[str]] = None        # chat: default ["assistant"]
    split: Optional[str] = None                       # accepted for compatibility; .jsonl only has "train"
    ds_type: Optional[str] = None


class FsdpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    offload_params: bool = False
    cpu_ram_efficient_loading: bool = False
    reshard_after_forward: bool = True
    activation_checkpointing: Optional[bool] = None   # refused: use gradient_checkpointing (owner: FSDP AC OOMs)

    @model_validator(mode="before")
    @classmethod
    def _drop_ignored(cls, data):
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k not in IGNORED_FSDP_KEYS}
        return data


class TrainConfig(MicroEngramArgs):
    model_config = ConfigDict(extra="forbid")

    # --- model ---
    base_model: str
    tokenizer_config: Optional[str] = None
    load_in_4bit: bool = False
    attn_implementation: Literal["sdpa", "flash_attention_2", "eager"] = "sdpa"
    gemma4_hybrid_attn_impl: bool = False             # FA2 on sliding layers, packed SDPA on the 512-dim globals
    large_head_attention: Optional[Literal["sdpa", "auto", "triton_flash"]] = None
    flash_attn_d512: Optional[bool] = None            # old name: true -> large_head_attention: auto
    fused_attn_kernel: bool = True                    # Gemma 4 fused q/k/v norm + RoPE (Axolotl always used it)
    liger_rms_norm: bool = False
    liger_layer_norm: bool = False                    # Gemma 4 has no LayerNorm in the text model: no-op
    liger_glu_activation: bool = False
    liger_rope: bool = False                          # covered by fused_attn_kernel; must stay off
    tf32: bool = False

    # --- data ---
    datasets: List[DatasetConfig]
    sequence_len: int = 2048
    sample_packing: bool = True
    dataset_prepared_path: Optional[str] = None
    axolotl_prepared_dataset: Optional[str] = None    # read an Axolotl prepared dataset folder as-is (exact parity)
    dataset_num_proc: Optional[int] = None
    chat_template_jinja: Optional[str] = None         # file path or the template itself
    chat_template_kwargs: Optional[Dict[str, Any]] = None
    eot_tokens: Optional[List[str]] = None            # end-of-turn tokens trained with the assistant turn
    shuffle_each_epoch: bool = True                   # bins reshuffled every epoch (seed + epoch)

    # --- training ---
    output_dir: str
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    num_epochs: float = 1
    max_steps: Optional[int] = None
    learning_rate: float = 1e-3                       # engram tables (and everything, without micro_engram_dense_lr)
    micro_engram_dense_lr: Optional[float] = None     # key/value/gates/scales; same schedule; default learning_rate
    micro_engram_train_scales: bool = True            # false: scalar_sqrt's gate norm scales stay as they are (1.0 if new)
    lr_scheduler: Literal["constant", "constant_with_warmup", "cosine", "linear",
                          "warmup_stable_decay"] = "constant_with_warmup"
    lr_scheduler_kwargs: Optional[Dict[str, Any]] = None
    cosine_min_lr_ratio: Optional[float] = None
    warmup_ratio: Optional[float] = None
    warmup_steps: Optional[int] = None
    weight_decay: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    optimizer: Optional[str] = None                   # key/value/scales: AdamW (fused when on CUDA)
    seed: int = 42
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None
    activation_offloading: bool = False
    no_grad_prefix: bool = True                       # layers before the first trained engram run without autograd
    sharded_vocab: bool = True                        # embedding/lm_head rows split across GPUs (vocab-parallel loss)
    sparse_row_grads: bool = True                     # lazy_adam/rowwise_adam: keep only touched rows' gradients
    loss_chunk_tokens: int = 2048                     # tokens per chunk of the fused loss
    fsdp_config: FsdpConfig = FsdpConfig()

    # --- checkpoints / logging ---
    save_steps: Optional[int] = None
    save_total_limit: Optional[int] = None
    auto_resume_from_checkpoints: bool = True
    resume_from_checkpoint: Optional[str] = None
    logging_steps: int = 1
    profile_step: Optional[int] = None               # profile this step (torch.profiler) -> output_dir/profile/
    wandb_project: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_mode: Optional[str] = None

    @model_validator(mode="after")
    def _check(self):
        if self.fsdp_config.activation_checkpointing:
            raise ValueError("fsdp_config.activation_checkpointing is not supported: use gradient_checkpointing: true "
                             "(per-layer non-reentrant checkpointing, as Axolotl's hf implementation)")
        if self.micro_engram_dense_lr is not None and self.micro_engram_dense_lr <= 0:
            raise ValueError("micro_engram_dense_lr must be > 0 (freeze an engram with micro_engram_frozen instead)")
        if self.liger_rope:
            raise ValueError("liger_rope: Gemma 4 RoPE is already fused with the q/k norms (fused_attn_kernel)")
        kw = self.gradient_checkpointing_kwargs or {}
        if kw.get("use_reentrant"):
            raise ValueError("gradient_checkpointing_kwargs.use_reentrant must be false (frozen inputs get no grads)")
        if self.micro_engram_sharding not in (None, "rows"):
            raise ValueError("micro_engram_sharding must be rows (or unset, which means rows here)")
        return self

    @property
    def large_head_policy(self):
        if self.large_head_attention:
            return self.large_head_attention
        return "auto" if self.flash_attn_d512 else "sdpa"

    def get(self, key, default=None):  # dict-style access some shared code uses
        v = getattr(self, key, default)
        return default if v is None else v


def load_config(path_or_dict):
    """TrainConfig from a YAML path (or a dict). Raises on LoRA keys and unknown keys."""
    if isinstance(path_or_dict, dict):
        raw = dict(path_or_dict)
    else:
        with open(path_or_dict, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    lora = sorted(k for k in raw if k in LORA_KEYS or k.startswith(LORA_PREFIX))
    if lora:
        raise ValueError(
            f"remove {', '.join(lora)}: micro_engram.train loads a 4-bit base directly (load_in_4bit: true), "
            "with no placeholder LoRA, and trains only the engram (no unfrozen_parameters needed)")
    if raw.get("fsdp_version") not in (None, 2, "2"):
        raise ValueError("only FSDP2 is supported (fsdp_version: 2, or leave it out)")
    import os

    first = os.environ.get("RANK", "0") == "0"  # one warning, not one per GPU
    ignored = sorted(k for k in raw if k in IGNORED_KEYS)
    if ignored and first:
        LOG.warning(f"[micro_engram] ignoring Axolotl-only keys: {', '.join(ignored)}")
    loss = sorted(k for k in raw if k in LOSS_KEYS)
    if loss and first:
        LOG.warning(f"[micro_engram] {', '.join(loss)}: not needed, the loss is always fused "
                    f"(chunked, vocab-parallel; no full logits)")
    for k in ignored + loss:
        raw.pop(k)
    if "load_in_8bit" in raw:
        if raw.pop("load_in_8bit"):
            raise ValueError("load_in_8bit is not supported (use load_in_4bit)")
    return TrainConfig(**raw)


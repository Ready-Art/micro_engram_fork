from typing import List, Literal, Optional

from pydantic import BaseModel

Gate = Literal["scalar", "ple", "scalar_sqrt"]


class MicroEngramExtraArgs(BaseModel):
    """One additional engram (micro_engram_extra). Unset sizes/gate copy the main engram's."""
    layer: int                       # decoder layer (0-indexed); one engram per layer
    table_size: Optional[int] = None
    dim: Optional[int] = None
    heads: Optional[int] = None
    gate: Optional[Gate] = None
    seed: Optional[int] = None       # default 10007 * layer (each layer gets its own hash addresses)
    init_from: Optional[str] = None  # start from a trained engram (checkpoint/output folder, config or weights)
    frozen: bool = False             # load it but don't train it (needs init_from)


class MicroEngramArgs(BaseModel):
    micro_engram_layer: int = 0            # inject after this decoder layer (0-indexed)
    micro_engram_table_size: int = 65_536  # rows per hash table
    micro_engram_dim: int = 128            # width of each table row
    micro_engram_heads: int = 4            # hash tables per n-gram order
    micro_engram_save_steps: int = 250     # 0 = only save at the end
    micro_engram_canon_path: Optional[str] = None  # token-merge map from build_canon.py; None = no merging
    # scalar = paper's sigmoid gate; scalar_sqrt = DeepSeek code's signed-sqrt gate with learnable
    # norm scales (recommended for new runs); ple = Gemma 3n PLE-style per-dim gate
    micro_engram_gate: Gate = "scalar"
    micro_engram_save_trainable_only: bool = False  # FSDP2: checkpoints hold only the engram (rebuild with assemble_engram_model.py)
    micro_engram_stats_steps: int = 10  # log engram weight/gate stats every N steps (0 = off)
    # --- several engrams / starting from a trained one ---
    micro_engram_seed: Optional[int] = None  # main engram's hash seed; default 10007 * layer (layer 0 -> 0, as before)
    micro_engram_init_from: Optional[str] = None  # start the main engram from trained weights
    micro_engram_frozen: bool = False  # load the main engram but don't train it (needs init_from)
    micro_engram_extra: Optional[List[MicroEngramExtraArgs]] = None  # more engrams at other layers
    # LR schedule: linear warmup (warmup_steps / warmup_ratio as usual), then the LR is multiplied
    # by this factor at the start of every epoch (0.5 = halve each epoch). Replaces lr_scheduler.
    micro_engram_lr_epoch_decay: Optional[float] = None
    # Engram weight dtype. Unset: same as its layer (bf16). fp32: fp32 master weights (Adam's small
    # steps no longer round away in bf16; ~2x the engram's weight/grad/optimizer memory); under
    # FSDP2 the compute stays bf16 (mixed-precision policy).
    micro_engram_dtype: Optional[Literal["bf16", "fp32"]] = None
    # Diagnostics: record every CUDA allocation for this many optimizer steps, then write
    # <output_dir>/memory_snapshot/rank<R>_step<S>.pickle (open at https://pytorch.org/memory_viz).
    # Slows those steps down; for a short diagnostic run only.
    micro_engram_memory_snapshot: Optional[int] = None
    # Optimizer for the trainable params. Unset: Axolotl's `optimizer`. lazy_adam: AdamW, but a table
    # row whose gradient is zero this step is not touched (no leftover-momentum drift), like
    # SparseAdam; works with FSDP2. Uses learning_rate, adam_beta1/2, adam_epsilon, weight_decay.
    # rowwise_adam (needs micro_engram_sharding: rows): for the tables, lazy updates as above plus ONE
    # fp32 second moment per row instead of one per value (FBGEMM/TorchRec partial row-wise Adam):
    # the tables' optimizer state is ~half of Adam's. Key/value/scales keep the YAML `optimizer`.
    micro_engram_optimizer: Optional[Literal["lazy_adam", "rowwise_adam"]] = None
    # rows: each GPU keeps its own rows of the tables outside FSDP and fetches only the rows a
    # batch uses (all_to_all), instead of FSDP gathering whole tables and reduce-scattering
    # full-size gradients. Needs FSDP. Tables get their own optimizer (AdamW, or lazy with
    # micro_engram_optimizer: lazy_adam) and their own clipping; checkpoints add micro_engram_rows/.
    micro_engram_sharding: Optional[Literal["rows"]] = None

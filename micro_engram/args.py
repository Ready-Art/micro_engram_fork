from typing import Dict, List, Literal, Optional

from pydantic import BaseModel

Gate = Literal["scalar", "ple", "scalar_sqrt"]


class MicroEngramExtraArgs(BaseModel):
    """One additional engram (micro_engram_extra). Unset sizes/gate copy the main engram's."""
    layer: int                       # decoder layer (0-indexed); one engram per layer
    table_size: Optional[int] = None
    table_sizes: Optional[Dict[int, Optional[int]]] = None  # per n-gram order, like micro_engram_table_sizes
    dim: Optional[int] = None
    heads: Optional[int] = None
    gate: Optional[Gate] = None
    seed: Optional[int] = None       # default 10007 * layer (each layer gets its own hash addresses)
    init_from: Optional[str] = None  # start from a trained engram (checkpoint/output folder, config or weights)
    frozen: bool = False             # load it but don't train it (needs init_from)


class MicroEngramArgs(BaseModel):
    micro_engram_layer: int = 0            # inject after this decoder layer (0-indexed)
    micro_engram_table_size: int = 65_536  # rows per hash table
    # Rows per table for each n-gram order, overriding micro_engram_table_size, e.g.
    # {2: 1048576, 3: 3145728, 4: 1048576}. An order set to 0 or null is not used; an order
    # not listed keeps micro_engram_table_size (orders 2 and 3), so {4: N} adds 4-grams.
    micro_engram_table_sizes: Optional[Dict[int, Optional[int]]] = None
    micro_engram_dim: int = 128            # width of each table row
    micro_engram_heads: int = 4            # hash tables per n-gram order
    micro_engram_save_steps: int = 250     # unused by micro_engram.train (kept so old YAMLs load)
    micro_engram_canon_path: Optional[str] = None  # token-merge map from build_canon.py; None = no merging
    # scalar = paper's sigmoid gate; scalar_sqrt = DeepSeek code's signed-sqrt gate with learnable
    # norm scales (recommended for new runs); ple = Gemma 3n PLE-style per-dim gate
    micro_engram_gate: Gate = "scalar"
    micro_engram_save_trainable_only: bool = False  # unused by micro_engram.train (it always saves only the engram)
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
    # Optimizer for the tables. Unset: AdamW (dense gradients). lazy_adam: AdamW, but a table
    # row whose gradient is zero this step is not touched (no leftover-momentum drift), like
    # SparseAdam. Uses learning_rate, adam_beta1/2, adam_epsilon, weight_decay.
    # rowwise_adam: lazy updates as above plus ONE fp32 second moment per row instead of one per value
    # (FBGEMM/TorchRec partial row-wise Adam): the tables' optimizer state is ~half of Adam's.
    # Key/value/scales always use AdamW (micro_engram_dense_lr).
    micro_engram_optimizer: Optional[Literal["lazy_adam", "rowwise_adam"]] = None
    # rows: each GPU keeps its own rows of the tables and fetches only the rows a batch uses
    # (all_to_all). micro_engram.train always does this; the key is accepted for old YAMLs.
    micro_engram_sharding: Optional[Literal["rows"]] = None
    # bf16 engram weights/optimizer states (micro_engram_optimizer lazy_adam / rowwise_adam, and the
    # row-sharded tables): written back with stochastic rounding, so updates smaller than half a
    # bf16 step add up on average instead of rounding away. No effect in fp32.
    micro_engram_stochastic_rounding: bool = True

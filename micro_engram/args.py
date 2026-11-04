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

from typing import Literal, Optional

from pydantic import BaseModel


class MicroEngramArgs(BaseModel):
    micro_engram_layer: int = 0            # inject after this decoder layer (0-indexed)
    micro_engram_table_size: int = 65_536  # rows per hash table
    micro_engram_dim: int = 128            # width of each table row
    micro_engram_heads: int = 4            # hash tables per n-gram order
    micro_engram_save_steps: int = 250     # 0 = only save at the end
    micro_engram_canon_path: Optional[str] = None  # token-merge map from build_canon.py; None = no merging
    # scalar = paper's sigmoid gate; scalar_sqrt = DeepSeek code's signed-sqrt gate with learnable
    # norm scales (recommended for new runs); ple = Gemma 3n PLE-style per-dim gate
    micro_engram_gate: Literal["scalar", "ple", "scalar_sqrt"] = "scalar"
    micro_engram_save_trainable_only: bool = False  # FSDP2: checkpoints hold only the engram (rebuild with assemble_engram_model.py)
    micro_engram_stats_steps: int = 10  # log engram weight/gate stats every N steps (0 = off)

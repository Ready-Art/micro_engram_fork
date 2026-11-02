from typing import Optional

from pydantic import BaseModel


class MicroEngramArgs(BaseModel):
    micro_engram_layer: int = 1            # inject after this decoder layer
    micro_engram_table_size: int = 65_536  # rows per hash table
    micro_engram_dim: int = 128            # width of each table row
    micro_engram_heads: int = 4            # hash tables per n-gram order
    micro_engram_save_steps: int = 250     # 0 = only save at the end
    micro_engram_canon_path: Optional[str] = None  # token-merge map from build_canon.py; None = no merging
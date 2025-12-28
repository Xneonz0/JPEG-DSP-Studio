"""Compression parameters."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class CompressionParams:
    """JPEG-like compression parameters."""
    
    block_size: int = 8
    quality: int = 50
    subsampling_mode: Literal['4:4:4', '4:2:2', '4:2:0'] = '4:2:0'
    use_prefilter: bool = False
    
    def __post_init__(self):
        if not (1 <= self.quality <= 100):
            raise ValueError(f"Quality must be 1-100, got {self.quality}")
        if self.block_size not in [4, 8, 16, 32]:
            raise ValueError(f"Block size must be 4, 8, 16, or 32, got {self.block_size}")

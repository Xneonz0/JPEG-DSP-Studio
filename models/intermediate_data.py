"""Intermediate data for visualization."""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class IntermediateData:
    """Intermediate results for GUI plots and analysis."""
    
    selected_block_idx: tuple = (0, 0)
    selected_block_original: Optional[np.ndarray] = None
    selected_block_shifted: Optional[np.ndarray] = None
    selected_block_dct: Optional[np.ndarray] = None
    selected_block_quantized: Optional[np.ndarray] = None
    selected_block_dequantized: Optional[np.ndarray] = None
    selected_block_reconstructed: Optional[np.ndarray] = None
    
    error_map_y: Optional[np.ndarray] = None
    error_map_rgb: Optional[np.ndarray] = None
    quantized_histogram: Optional[np.ndarray] = None
    all_quantized_coeffs: Optional[np.ndarray] = None

"""Compression result with metrics."""

from dataclasses import dataclass
import numpy as np


@dataclass
class CompressionResult:
    """Results from compression/reconstruction pipeline."""
    
    original_image: np.ndarray
    reconstructed_image: np.ndarray
    
    # Quality metrics
    psnr_y: float
    ssim_y: float
    psnr_rgb: float
    ssim_rgb: float
    
    # Compression stats
    bpp: float
    compression_ratio: float
    nonzero_coeffs: int
    total_coeffs: int
    
    # Runtime
    encode_time_ms: float
    decode_time_ms: float
    
    bitrate_label: str = "Estimated (no entropy coding)"

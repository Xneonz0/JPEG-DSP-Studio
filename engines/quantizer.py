"""Quantization operations."""

import numpy as np
from utils.constants import JPEG_LUMA_Q50


def scale_quant_matrix(base_matrix: np.ndarray, quality: int) -> np.ndarray:
    """Scale quantization matrix by quality factor (1-100)."""
    quality = np.clip(quality, 1, 100)
    
    # JPEG scaling formula
    if quality < 50:
        scale = 5000.0 / quality
    else:
        scale = 200.0 - 2.0 * quality
    
    Q = np.floor((base_matrix * scale + 50.0) / 100.0)
    Q = np.clip(Q, 1, 255)
    return Q.astype(np.float64)


def quantize(dct_coeffs: np.ndarray, Q_matrix: np.ndarray) -> np.ndarray:
    """Quantize DCT coefficients."""
    return np.round(dct_coeffs / Q_matrix).astype(np.int16)


def dequantize(quantized: np.ndarray, Q_matrix: np.ndarray) -> np.ndarray:
    """Dequantize coefficients."""
    return quantized.astype(np.float64) * Q_matrix

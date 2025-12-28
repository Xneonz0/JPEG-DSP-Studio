"""DCT/IDCT operations with level shift."""

import numpy as np
from scipy.fft import dctn, idctn


def dct2(block: np.ndarray) -> np.ndarray:
    """2D DCT-II with orthonormal normalization."""
    return dctn(block, type=2, norm='ortho')


def idct2(coeffs: np.ndarray) -> np.ndarray:
    """2D inverse DCT (Type-III)."""
    return idctn(coeffs, type=2, norm='ortho')


def encode_block(block: np.ndarray) -> np.ndarray:
    """Level shift (-128) then DCT."""
    shifted = block.astype(np.float64) - 128.0
    return dct2(shifted)


def decode_block(coeffs: np.ndarray) -> np.ndarray:
    """IDCT then reverse level shift (+128), clip to [0,255]."""
    spatial = idct2(coeffs)
    unshifted = spatial + 128.0
    return np.clip(unshifted, 0, 255)

"""Color space conversion and chroma subsampling."""

import numpy as np
import cv2
from typing import Literal, Tuple


def rgb_to_ycbcr(rgb: np.ndarray) -> np.ndarray:
    """RGB to YCbCr using ITU-R BT.601."""
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.168736 * R - 0.331264 * G + 0.5 * B + 128.0
    Cr = 0.5 * R - 0.418688 * G - 0.081312 * B + 128.0
    return np.stack([Y, Cb, Cr], axis=-1)


def ycbcr_to_rgb(ycbcr: np.ndarray) -> np.ndarray:
    """YCbCr to RGB using ITU-R BT.601."""
    Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
    R = Y + 1.402 * (Cr - 128.0)
    G = Y - 0.344136 * (Cb - 128.0) - 0.714136 * (Cr - 128.0)
    B = Y + 1.772 * (Cb - 128.0)
    rgb = np.stack([R, G, B], axis=-1)
    return np.clip(rgb, 0, 255)


def subsample_chroma(
    cb: np.ndarray,
    cr: np.ndarray,
    mode: Literal['4:4:4', '4:2:2', '4:2:0'],
    use_prefilter: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample chroma channels according to mode."""
    if mode == '4:4:4':
        return cb.copy(), cr.copy()
    
    # Anti-alias blur before downsampling
    if use_prefilter:
        cb = cv2.GaussianBlur(cb, (3, 3), sigmaX=0.75)
        cr = cv2.GaussianBlur(cr, (3, 3), sigmaX=0.75)
    
    if mode == '4:2:2':
        # Horizontal 2x downsample
        cb_sub = cv2.resize(cb, (cb.shape[1] // 2, cb.shape[0]), interpolation=cv2.INTER_AREA)
        cr_sub = cv2.resize(cr, (cr.shape[1] // 2, cr.shape[0]), interpolation=cv2.INTER_AREA)
    elif mode == '4:2:0':
        # 2x2 downsample
        cb_sub = cv2.resize(cb, (cb.shape[1] // 2, cb.shape[0] // 2), interpolation=cv2.INTER_AREA)
        cr_sub = cv2.resize(cr, (cr.shape[1] // 2, cr.shape[0] // 2), interpolation=cv2.INTER_AREA)
    else:
        raise ValueError(f"Unknown subsampling mode: {mode}")
    
    return cb_sub, cr_sub


def upsample_chroma(
    cb_sub: np.ndarray,
    cr_sub: np.ndarray,
    target_shape: Tuple[int, int],
    method: str = 'bilinear'
) -> Tuple[np.ndarray, np.ndarray]:
    """Upsample chroma channels to target resolution."""
    interp = cv2.INTER_LINEAR if method == 'bilinear' else cv2.INTER_NEAREST
    cb_up = cv2.resize(cb_sub, (target_shape[1], target_shape[0]), interpolation=interp)
    cr_up = cv2.resize(cr_sub, (target_shape[1], target_shape[0]), interpolation=interp)
    return cb_up, cr_up

"""Metrics: PSNR, SSIM, bitrate estimation."""

import time
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from typing import Dict


def compute_psnr_ssim(original_rgb: np.ndarray, reconstructed_rgb: np.ndarray) -> Dict[str, float]:
    """Compute PSNR and SSIM on RGB and Y channel."""
    psnr_rgb = peak_signal_noise_ratio(original_rgb, reconstructed_rgb, data_range=255)
    ssim_rgb = structural_similarity(
        original_rgb, reconstructed_rgb, channel_axis=2, data_range=255
    )
    
    # Y channel (luminance) - BT.601
    original_y = 0.299 * original_rgb[:, :, 0] + 0.587 * original_rgb[:, :, 1] + 0.114 * original_rgb[:, :, 2]
    recon_y = 0.299 * reconstructed_rgb[:, :, 0] + 0.587 * reconstructed_rgb[:, :, 1] + 0.114 * reconstructed_rgb[:, :, 2]
    
    psnr_y = peak_signal_noise_ratio(original_y, recon_y, data_range=255)
    ssim_y = structural_similarity(original_y, recon_y, data_range=255)
    
    return {
        'psnr_rgb': float(psnr_rgb),
        'ssim_rgb': float(ssim_rgb),
        'psnr_y': float(psnr_y),
        'ssim_y': float(ssim_y)
    }


class Timer:
    """Simple timer for encode/decode runtime."""
    
    def __init__(self):
        self.encode_time_ms = 0.0
        self.decode_time_ms = 0.0
    
    def measure_encode(self, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        self.encode_time_ms = (time.perf_counter() - start) * 1000.0
        return result
    
    def measure_decode(self, func, *args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        self.decode_time_ms = (time.perf_counter() - start) * 1000.0
        return result


def estimate_bitrate_no_entropy(
    quantized_coeffs: np.ndarray,
    original_shape: tuple,
    block_size: int = 8
) -> Dict:
    """
    Estimate compressed size WITHOUT entropy coding.
    
    Simple model: for each non-zero coefficient, store position (6 bits)
    and magnitude (variable). Conservative estimate - real JPEG would be smaller.
    """
    h, w = original_shape
    num_pixels = h * w
    original_bits = num_pixels * 3 * 8
    
    padded_h = ((h + block_size - 1) // block_size) * block_size
    padded_w = ((w + block_size - 1) // block_size) * block_size
    num_blocks = (padded_h // block_size) * (padded_w // block_size)
    
    block_overhead_bits = num_blocks * 2
    
    nonzero_mask = quantized_coeffs != 0
    nonzero_coeffs = quantized_coeffs[nonzero_mask]
    
    if len(nonzero_coeffs) > 0:
        position_bits = 6 * len(nonzero_coeffs)
        magnitudes = np.abs(nonzero_coeffs)
        magnitude_bits = np.sum(np.ceil(np.log2(magnitudes + 1)) + 1)
        coeff_bits = position_bits + magnitude_bits
    else:
        coeff_bits = 0
    
    estimated_bits = block_overhead_bits + coeff_bits
    
    return {
        'estimated_bits': int(estimated_bits),
        'bpp': float(estimated_bits / num_pixels),
        'compression_ratio': float(original_bits / max(estimated_bits, 1)),
        'nonzero_count': int(np.sum(nonzero_mask)),
        'total_coeffs': int(quantized_coeffs.size),
        'label': 'Estimated (no entropy coding)'
    }

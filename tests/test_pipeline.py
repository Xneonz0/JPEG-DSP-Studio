"""Tests for compression pipeline."""

import numpy as np
import pytest
from models.compression_params import CompressionParams
from engines.pipeline import compress_reconstruct


def test_quality_psnr_monotonic():
    """PSNR should increase with quality."""
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    qualities = [10, 30, 50, 70, 90]
    psnr_values = []
    
    for q in qualities:
        params = CompressionParams(quality=q, block_size=8, subsampling_mode='4:4:4')
        result, _ = compress_reconstruct(image, params)
        psnr_values.append(result.psnr_y)
    
    for i in range(len(psnr_values) - 1):
        assert psnr_values[i] <= psnr_values[i + 1] + 0.1


def test_perfect_reconstruction_high_quality():
    """Q=100 with 4:4:4 should give PSNR > 45 dB."""
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    params = CompressionParams(quality=100, block_size=8, subsampling_mode='4:4:4')
    result, _ = compress_reconstruct(image, params)
    assert result.psnr_y > 45.0


def test_subsampling_affects_quality():
    """4:4:4 should give better quality than 4:2:0."""
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    result_444, _ = compress_reconstruct(image, CompressionParams(quality=50, subsampling_mode='4:4:4'))
    result_420, _ = compress_reconstruct(image, CompressionParams(quality=50, subsampling_mode='4:2:0'))
    
    assert result_444.psnr_y >= result_420.psnr_y


def test_compression_ratio_increases_with_lower_quality():
    """Lower quality = better compression ratio."""
    image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    
    result_high, _ = compress_reconstruct(image, CompressionParams(quality=90))
    result_low, _ = compress_reconstruct(image, CompressionParams(quality=10))
    
    assert result_low.compression_ratio >= result_high.compression_ratio

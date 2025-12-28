"""Tests for DCT/IDCT operations."""

import numpy as np
import pytest
from engines.dct_engine import dct2, idct2, encode_block, decode_block


def test_dct_idct_invertibility():
    """DCT/IDCT should be perfectly invertible."""
    block = np.random.rand(8, 8) * 255
    shifted = block - 128.0
    dct_coeffs = dct2(shifted)
    recovered = idct2(dct_coeffs) + 128.0
    assert np.allclose(block, recovered, atol=1e-10)


def test_encode_decode_block_invertibility():
    """encode/decode should recover original without quantization."""
    block = np.random.rand(8, 8) * 255
    dct_coeffs = encode_block(block)
    recovered = decode_block(dct_coeffs)
    assert np.allclose(block, recovered, atol=1e-8)


def test_level_shift_reduces_dc():
    """Level shift reduces DC coefficient magnitude."""
    block = np.ones((8, 8)) * 200
    dct_no_shift = dct2(block)
    dct_with_shift = dct2(block - 128.0)
    assert abs(dct_with_shift[0, 0]) < abs(dct_no_shift[0, 0])


def test_energy_preservation():
    """Parseval's theorem: sum(block^2) == sum(dct^2) for ortho norm."""
    block = np.random.rand(8, 8) * 255
    shifted = block - 128.0
    dct_block = dct2(shifted)
    assert np.isclose(np.sum(shifted ** 2), np.sum(dct_block ** 2), rtol=1e-10)


def test_constant_block_dct():
    """Constant block should have only DC coefficient."""
    const_block = np.ones((8, 8)) * 128
    shifted = const_block - 128.0
    dct_block = dct2(shifted)
    assert np.allclose(dct_block[0, 1:], 0, atol=1e-10)
    assert np.allclose(dct_block[1:, :], 0, atol=1e-10)

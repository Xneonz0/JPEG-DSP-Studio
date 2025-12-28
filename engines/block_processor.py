"""Block processing: padding, splitting, merging."""

import numpy as np
from typing import List, Tuple


def pad_to_multiple(channel: np.ndarray, block_size: int) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad channel to multiple of block_size using reflect mode."""
    h, w = channel.shape
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    if pad_h > 0 or pad_w > 0:
        padded = np.pad(channel, ((0, pad_h), (0, pad_w)), mode='reflect')
    else:
        padded = channel.copy()
    return padded, (h, w)


def split_into_blocks(channel: np.ndarray, block_size: int) -> List[Tuple[int, int, np.ndarray]]:
    """Split 2D channel into BxB blocks."""
    h, w = channel.shape
    blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = channel[i:i+block_size, j:j+block_size]
            if block.shape[0] < block_size or block.shape[1] < block_size:
                padded_block = np.zeros((block_size, block_size), dtype=channel.dtype)
                padded_block[:block.shape[0], :block.shape[1]] = block
                block = padded_block
            blocks.append((i, j, block.copy()))
    return blocks


def merge_blocks(
    blocks: List[Tuple[int, int, np.ndarray]],
    shape: Tuple[int, int],
    block_size: int
) -> np.ndarray:
    """Merge blocks back into 2D channel."""
    h, w = shape
    result = np.zeros((h, w), dtype=np.float64)
    for (i, j, block) in blocks:
        end_i = min(i + block_size, h)
        end_j = min(j + block_size, w)
        block_h = end_i - i
        block_w = end_j - j
        result[i:end_i, j:end_j] = block[:block_h, :block_w]
    return result

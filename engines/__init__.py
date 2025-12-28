"""DSP engines - pure computation, no GUI dependencies."""

from .color_space import rgb_to_ycbcr, ycbcr_to_rgb, subsample_chroma, upsample_chroma
from .block_processor import pad_to_multiple, split_into_blocks, merge_blocks
from .dct_engine import dct2, idct2, encode_block, decode_block
from .quantizer import scale_quant_matrix, quantize, dequantize
from utils.constants import JPEG_LUMA_Q50
from .pipeline import compress_reconstruct

__all__ = [
    'rgb_to_ycbcr',
    'ycbcr_to_rgb',
    'subsample_chroma',
    'upsample_chroma',
    'pad_to_multiple',
    'split_into_blocks',
    'merge_blocks',
    'dct2',
    'idct2',
    'encode_block',
    'decode_block',
    'scale_quant_matrix',
    'quantize',
    'dequantize',
    'JPEG_LUMA_Q50',
    'compress_reconstruct',
]


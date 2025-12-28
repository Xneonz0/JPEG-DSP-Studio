"""Main compression/reconstruction pipeline."""

import numpy as np
from typing import Tuple

from models.compression_params import CompressionParams
from models.compression_result import CompressionResult
from models.intermediate_data import IntermediateData
from engines.color_space import rgb_to_ycbcr, ycbcr_to_rgb, subsample_chroma, upsample_chroma
from engines.block_processor import pad_to_multiple, split_into_blocks, merge_blocks
from engines.dct_engine import encode_block, decode_block
from engines.quantizer import scale_quant_matrix, quantize, dequantize
from utils.constants import JPEG_LUMA_Q50
from utils.metrics import compute_psnr_ssim, Timer, estimate_bitrate_no_entropy


def compress_reconstruct(
    image_rgb: np.ndarray,
    params: CompressionParams,
    selected_block_idx: Tuple[int, int] = (0, 0)
) -> Tuple[CompressionResult, IntermediateData]:
    """Run full JPEG-like compression and reconstruction pipeline."""
    timer = Timer()
    original_shape = image_rgb.shape[:2]
    image_float = image_rgb.astype(np.float64)
    
    # === ENCODING ===
    ycbcr = timer.measure_encode(rgb_to_ycbcr, image_float)
    Y, Cb, Cr = ycbcr[:, :, 0], ycbcr[:, :, 1], ycbcr[:, :, 2]
    
    Cb_sub, Cr_sub = subsample_chroma(
        Cb, Cr, params.subsampling_mode, params.use_prefilter
    )
    
    Y_original = Y.copy()
    
    channels_to_process = [
        ('Y', Y, original_shape),
        ('Cb', Cb_sub, Cb_sub.shape),
        ('Cr', Cr_sub, Cr_sub.shape)
    ]
    
    Q_matrix = scale_quant_matrix(JPEG_LUMA_Q50, params.quality)
    all_quantized_coeffs = []
    
    processed_channels = {}
    for channel_name, channel, channel_shape in channels_to_process:
        padded, orig_shape = pad_to_multiple(channel, params.block_size)
        blocks = split_into_blocks(padded, params.block_size)
        
        quantized_blocks = []
        for (i, j, block) in blocks:
            dct_coeffs = encode_block(block)
            quantized = quantize(dct_coeffs, Q_matrix)
            quantized_blocks.append(quantized)
            all_quantized_coeffs.append(quantized.flatten())
        
        processed_channels[channel_name] = {
            'blocks': blocks,
            'quantized_blocks': quantized_blocks,
            'orig_shape': orig_shape,
            'padded_shape': padded.shape
        }
    
    # === DECODING ===
    reconstructed_channels = {}
    
    for channel_name, channel_data in processed_channels.items():
        quantized_blocks = channel_data['quantized_blocks']
        blocks = channel_data['blocks']
        padded_shape = channel_data['padded_shape']
        orig_shape = channel_data['orig_shape']
        
        reconstructed_blocks = []
        for (i, j, _), quantized in zip(blocks, quantized_blocks):
            dequantized = dequantize(quantized, Q_matrix)
            reconstructed_block = decode_block(dequantized)
            reconstructed_blocks.append((i, j, reconstructed_block))
        
        reconstructed_padded = merge_blocks(reconstructed_blocks, padded_shape, params.block_size)
        h, w = orig_shape
        reconstructed_channels[channel_name] = reconstructed_padded[:h, :w]
    
    Y_recon = reconstructed_channels['Y']
    Cb_recon = reconstructed_channels['Cb']
    Cr_recon = reconstructed_channels['Cr']
    
    if params.subsampling_mode != '4:4:4':
        Cb_recon, Cr_recon = timer.measure_decode(
            upsample_chroma, Cb_recon, Cr_recon, original_shape, 'bilinear'
        )
    
    ycbcr_recon = np.stack([Y_recon, Cb_recon, Cr_recon], axis=-1)
    rgb_recon_float = timer.measure_decode(ycbcr_to_rgb, ycbcr_recon)
    rgb_recon = np.clip(rgb_recon_float, 0, 255).astype(np.uint8)
    
    # === METRICS ===
    metrics = compute_psnr_ssim(image_rgb, rgb_recon)
    all_quantized_flat = np.concatenate(all_quantized_coeffs) if all_quantized_coeffs else np.array([])
    bitrate_info = estimate_bitrate_no_entropy(all_quantized_flat, original_shape, params.block_size)
    
    result = CompressionResult(
        original_image=image_rgb,
        reconstructed_image=rgb_recon,
        psnr_y=metrics['psnr_y'],
        ssim_y=metrics['ssim_y'],
        psnr_rgb=metrics['psnr_rgb'],
        ssim_rgb=metrics['ssim_rgb'],
        bpp=bitrate_info['bpp'],
        compression_ratio=bitrate_info['compression_ratio'],
        nonzero_coeffs=bitrate_info['nonzero_count'],
        total_coeffs=bitrate_info['total_coeffs'],
        encode_time_ms=timer.encode_time_ms,
        decode_time_ms=timer.decode_time_ms,
        bitrate_label=bitrate_info['label']
    )
    
    # === INTERMEDIATE DATA ===
    Y_recon_full = Y_recon[:Y_original.shape[0], :Y_original.shape[1]]
    error_map_y = np.abs(Y_original - Y_recon_full)
    error_map_rgb = np.mean(np.abs(image_float - rgb_recon_float), axis=2)
    
    if len(all_quantized_flat) > 0:
        hist, _ = np.histogram(all_quantized_flat, bins=50, range=(-100, 100))
    else:
        hist = np.array([])
    
    selected_block_data = None
    if 'Y' in processed_channels:
        blocks = processed_channels['Y']['blocks']
        quantized_blocks = processed_channels['Y']['quantized_blocks']
        block_row, block_col = selected_block_idx
        
        padded_y = processed_channels['Y']['padded_shape']
        blocks_per_row = padded_y[1] // params.block_size
        
        target_idx = block_row * blocks_per_row + block_col
        if 0 <= target_idx < len(blocks):
            (i, j, orig_block), quantized = blocks[target_idx], quantized_blocks[target_idx]
            shifted = orig_block.astype(np.float64) - 128.0
            dct_coeffs = encode_block(orig_block)
            dequantized = dequantize(quantized, Q_matrix)
            reconstructed_block = decode_block(dequantized)
            selected_block_data = {
                'original': orig_block,
                'shifted': shifted,
                'dct': dct_coeffs,
                'quantized': quantized,
                'dequantized': dequantized,
                'reconstructed': reconstructed_block
            }
    
    intermediate = IntermediateData(
        selected_block_idx=selected_block_idx,
        selected_block_original=selected_block_data['original'] if selected_block_data else None,
        selected_block_shifted=selected_block_data['shifted'] if selected_block_data else None,
        selected_block_dct=selected_block_data['dct'] if selected_block_data else None,
        selected_block_quantized=selected_block_data['quantized'] if selected_block_data else None,
        selected_block_dequantized=selected_block_data['dequantized'] if selected_block_data else None,
        selected_block_reconstructed=selected_block_data['reconstructed'] if selected_block_data else None,
        error_map_y=error_map_y,
        error_map_rgb=error_map_rgb,
        quantized_histogram=hist,
        all_quantized_coeffs=all_quantized_flat
    )
    
    return result, intermediate

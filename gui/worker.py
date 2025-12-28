"""Background workers for compression pipeline."""

import numpy as np
from PySide6.QtCore import QObject, Signal

from models.compression_params import CompressionParams
from engines.pipeline import compress_reconstruct


class CompressionWorker(QObject):
    """Runs compression in background thread."""
    
    finished = Signal(object, object)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, image: np.ndarray, params: CompressionParams, selected_block_idx=(0, 0)):
        super().__init__()
        self.image = image
        self.params = params
        self.selected_block_idx = selected_block_idx
    
    def run(self):
        try:
            h, w = self.image.shape[:2]
            self.progress.emit(f"Processing ({w}×{h})...")
            self.progress.emit(f"DCT & quantization (Q={self.params.quality})...")
            
            result, intermediate = compress_reconstruct(
                self.image, self.params, self.selected_block_idx
            )
            
            self.progress.emit("Computing metrics...")
            self.finished.emit(result, intermediate)
        except Exception as e:
            self.error.emit(str(e))


class BatchSweepWorker(QObject):
    """Runs quality sweep in background thread."""
    
    finished = Signal(list)
    error = Signal(str)
    progress = Signal(int, int)
    
    def __init__(self, image: np.ndarray, base_params: CompressionParams,
                 quality_start: int = 10, quality_end: int = 90, quality_step: int = 10):
        super().__init__()
        self.image = image
        self.base_params = base_params
        self.quality_start = quality_start
        self.quality_end = quality_end
        self.quality_step = quality_step
    
    def run(self):
        try:
            results = []
            qualities = list(range(self.quality_start, self.quality_end + 1, self.quality_step))
            total = len(qualities)
            
            for i, quality in enumerate(qualities):
                params = CompressionParams(
                    block_size=self.base_params.block_size,
                    quality=quality,
                    subsampling_mode=self.base_params.subsampling_mode,
                    use_prefilter=self.base_params.use_prefilter
                )
                result, _ = compress_reconstruct(self.image, params)
                results.append((quality, result))
                self.progress.emit(i + 1, total)
            
            self.finished.emit(results)
        except Exception as e:
            self.error.emit(str(e))

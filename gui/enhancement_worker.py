"""Background worker for image enhancement."""

import time
import cv2
import numpy as np
from dataclasses import dataclass
from PySide6.QtCore import QObject, Signal

from enhancement.device_detector import detect_device, TORCH_AVAILABLE
from enhancement.target_scaler import compute_scale_factor, should_skip_sr, resize_to_target, lanczos_to_target
from enhancement.realesrgan_upscaler import RealESRGANUpscaler, REALESRGAN_AVAILABLE


PDF_THUMBNAIL_SIZE = 512


def create_thumbnail(image: np.ndarray, max_size: int = PDF_THUMBNAIL_SIZE) -> np.ndarray:
    """Create downscaled thumbnail for PDF/display."""
    h, w = image.shape[:2]
    if w <= max_size and h <= max_size:
        return image.copy()
    
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


@dataclass
class EnhancementResult:
    """Result of enhancement operation."""
    
    output_image: np.ndarray
    input_thumbnail: np.ndarray
    output_thumbnail: np.ndarray
    
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    target_width: int
    target_height: int
    
    device: str
    gpu_name: str | None
    method: str
    mode: str
    
    fp16: bool
    tile_size: int
    target_preset: str
    
    runtime_seconds: float
    sr_skipped: bool
    used_gpu: bool


class EnhancementWorker(QObject):
    """Runs enhancement in background thread."""
    
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, image: np.ndarray, target_width: int, target_height: int,
                 mode: str = "fit", fp16: bool = True, tile_size: int = 512, target_preset: str = ""):
        super().__init__()
        self.image = image
        self.target_width = target_width
        self.target_height = target_height
        self.mode = mode
        self.fp16 = fp16
        self.tile_size = tile_size
        self.target_preset = target_preset
    
    def run(self):
        try:
            start_time = time.perf_counter()
            h, w = self.image.shape[:2]
            
            self.progress.emit("Detecting device...")
            device, gpu_name, reason = detect_device()
            
            if device == "cuda" and REALESRGAN_AVAILABLE:
                from enhancement.realesrgan_upscaler import MODEL_PATH
                if not MODEL_PATH.exists():
                    self.progress.emit("Downloading Real-ESRGAN weights (~64MB)...")
            
            scale = compute_scale_factor(w, h, self.target_width, self.target_height, self.mode)
            skip_sr = should_skip_sr(scale)
            use_gpu = device == "cuda" and REALESRGAN_AVAILABLE and not skip_sr
            
            if skip_sr:
                self.progress.emit(f"Scale {scale:.2f}x < 1.1x, direct resize...")
                output = resize_to_target(self.image, self.target_width, self.target_height, self.mode)
                method = "Direct resize"
            elif use_gpu:
                if not RealESRGANUpscaler._warmed_up:
                    self.progress.emit(f"GPU warm-up on {gpu_name}...")
                
                self.progress.emit(f"Real-ESRGAN ×4 on {gpu_name}...")
                sr_output = RealESRGANUpscaler.enhance(
                    self.image, device=device, fp16=self.fp16, tile_size=self.tile_size
                )
                
                self.progress.emit(f"Resize to {self.target_width}×{self.target_height}...")
                output = resize_to_target(sr_output, self.target_width, self.target_height, self.mode)
                method = "Real-ESRGAN x4"
            else:
                self.progress.emit(f"CPU Lanczos to {self.target_width}×{self.target_height}...")
                output = lanczos_to_target(self.image, self.target_width, self.target_height, self.mode)
                method = "CPU Lanczos"
            
            runtime = time.perf_counter() - start_time
            out_h, out_w = output.shape[:2]
            
            result = EnhancementResult(
                output_image=output,
                input_thumbnail=create_thumbnail(self.image),
                output_thumbnail=create_thumbnail(output),
                input_width=w,
                input_height=h,
                output_width=out_w,
                output_height=out_h,
                target_width=self.target_width,
                target_height=self.target_height,
                device=device,
                gpu_name=gpu_name,
                method=method,
                mode=self.mode,
                fp16=self.fp16,
                tile_size=self.tile_size,
                target_preset=self.target_preset,
                runtime_seconds=runtime,
                sr_skipped=skip_sr,
                used_gpu=use_gpu
            )
            
            self.progress.emit(f"Done in {runtime:.2f}s")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))

"""Real-ESRGAN upscaler with model caching."""

import cv2
import numpy as np
from pathlib import Path

from enhancement.device_detector import TORCH_AVAILABLE

REALESRGAN_AVAILABLE = False

if TORCH_AVAILABLE:
    try:
        import torch
        from realesrgan import RealESRGANer
        from basicsr.archs.rrdbnet_arch import RRDBNet
        REALESRGAN_AVAILABLE = True
    except ImportError:
        torch = None
        RealESRGANer = None
        RRDBNet = None
else:
    torch = None
    RealESRGANer = None
    RRDBNet = None

CACHE_DIR = Path.home() / ".cache" / "realesrgan"
MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_PATH = CACHE_DIR / "RealESRGAN_x4plus.pth"


def ensure_model() -> Path:
    """Download model if not present."""
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is not installed")
    
    if MODEL_PATH.exists():
        return MODEL_PATH
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.hub.download_url_to_file(MODEL_URL, str(MODEL_PATH))
    
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Failed to download model to {MODEL_PATH}")
    
    return MODEL_PATH


class RealESRGANUpscaler:
    """Singleton wrapper for Real-ESRGAN."""
    
    _instance = None
    _warmed_up = False
    _current_config = None
    
    @classmethod
    def is_available(cls) -> bool:
        return REALESRGAN_AVAILABLE and TORCH_AVAILABLE
    
    @classmethod
    def get_upsampler(cls, device: str = "cuda", fp16: bool = True, tile_size: int = 512):
        if not cls.is_available():
            raise RuntimeError("Real-ESRGAN not available. Install: pip install realesrgan basicsr")
        
        new_config = {"device": device, "fp16": fp16 and device == "cuda", "tile_size": tile_size}
        
        if cls._instance is not None and cls._current_config == new_config:
            return cls._instance
        
        model_path = ensure_model()
        
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        
        if device == "cuda":
            torch.backends.cudnn.benchmark = True
        
        cls._instance = RealESRGANer(
            scale=4, model_path=str(model_path), model=model,
            tile=tile_size, tile_pad=10, pre_pad=0,
            half=fp16 and device == "cuda", device=device
        )
        cls._current_config = new_config
        
        if device == "cuda" and not cls._warmed_up:
            cls._warmup()
            cls._warmed_up = True
        
        return cls._instance
    
    @classmethod
    def _warmup(cls):
        """GPU warm-up to reduce first-run latency."""
        if cls._instance is None:
            return
        
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            with torch.inference_mode():
                cls._instance.enhance(dummy, outscale=4)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    
    @classmethod
    def enhance(cls, img_rgb: np.ndarray, device: str = "cuda", fp16: bool = True, tile_size: int = 512) -> np.ndarray:
        """Enhance image using Real-ESRGAN. Input/output in RGB."""
        upsampler = cls.get_upsampler(device, fp16, tile_size)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        
        with torch.inference_mode():
            output_bgr, _ = upsampler.enhance(img_bgr, outscale=4)
        
        return cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    
    @classmethod
    def clear_cache(cls):
        cls._instance = None
        cls._current_config = None
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_enhancement_method(device: str) -> str:
    if device == "cuda" and REALESRGAN_AVAILABLE:
        return "Real-ESRGAN x4"
    return "CPU Lanczos"


def get_availability_message() -> tuple[bool, str]:
    if not TORCH_AVAILABLE:
        return (False, "GPU unavailable (PyTorch not installed)")
    if not REALESRGAN_AVAILABLE:
        return (False, "GPU unavailable (Real-ESRGAN not installed)")
    try:
        if not torch.cuda.is_available():
            return (False, "GPU unavailable (No CUDA)")
    except Exception:
        return (False, "GPU unavailable (CUDA error)")
    return (True, "GPU available (Real-ESRGAN x4)")

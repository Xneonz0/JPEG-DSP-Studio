"""Enhancement module for GPU-accelerated image upscaling."""

from enhancement.device_detector import (
    detect_device,
    get_device_display_string,
    TORCH_AVAILABLE,
)
from enhancement.target_scaler import (
    TARGET_RESOLUTIONS,
    compute_fit,
    compute_fill,
    compute_scale_factor,
    resize_to_target,
    center_crop,
    should_skip_sr,
    lanczos_upscale,
    lanczos_to_target,
)
from enhancement.realesrgan_upscaler import (
    RealESRGANUpscaler,
    REALESRGAN_AVAILABLE,
    get_enhancement_method,
    get_availability_message,
)

__all__ = [
    # Device detection
    "detect_device",
    "get_device_display_string",
    "TORCH_AVAILABLE",
    # Target scaling
    "TARGET_RESOLUTIONS",
    "compute_fit",
    "compute_fill",
    "compute_scale_factor",
    "resize_to_target",
    "center_crop",
    "should_skip_sr",
    "lanczos_upscale",
    "lanczos_to_target",
    # Real-ESRGAN
    "RealESRGANUpscaler",
    "REALESRGAN_AVAILABLE",
    "get_enhancement_method",
    "get_availability_message",
]


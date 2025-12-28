"""Target resolution scaling with Fit/Fill modes."""

import cv2
import numpy as np

TARGET_RESOLUTIONS = [
    (1920, 1080, "1080p (1920×1080)"),
    (2560, 1440, "1440p (2560×1440)"),
    (3840, 2160, "4K (3840×2160)"),
]


def compute_fit(w: int, h: int, tw: int, th: int) -> tuple[int, int]:
    """Compute dimensions to fit within target, preserving aspect ratio."""
    scale = min(tw / w, th / h)
    return (round(w * scale), round(h * scale))


def compute_fill(w: int, h: int, tw: int, th: int) -> tuple[int, int, int, int]:
    """Compute dimensions to fill target completely (may crop)."""
    scale = max(tw / w, th / h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    crop_x = (new_w - tw) // 2
    crop_y = (new_h - th) // 2
    return (new_w, new_h, crop_x, crop_y)


def compute_scale_factor(w: int, h: int, tw: int, th: int, mode: str = "fit") -> float:
    """Compute scale factor needed to reach target resolution."""
    if mode == "fill":
        return max(tw / w, th / h)
    return min(tw / w, th / h)


def should_skip_sr(scale: float, threshold: float = 1.1) -> bool:
    """Skip SR if scale is below threshold."""
    return scale < threshold


def center_crop(img: np.ndarray, crop_w: int, crop_h: int) -> np.ndarray:
    """Center crop image to specified dimensions."""
    h, w = img.shape[:2]
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    x = (w - crop_w) // 2
    y = (h - crop_h) // 2
    return img[y:y + crop_h, x:x + crop_w]


def resize_to_target(img: np.ndarray, target_w: int, target_h: int, mode: str = "fit") -> np.ndarray:
    """Resize image to target using smart interpolation."""
    h, w = img.shape[:2]
    
    if mode == "fill":
        new_w, new_h, _, _ = compute_fill(w, h, target_w, target_h)
        interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        return center_crop(resized, target_w, target_h)
    else:
        new_w, new_h = compute_fit(w, h, target_w, target_h)
        interp = cv2.INTER_AREA if (new_w < w or new_h < h) else cv2.INTER_CUBIC
        return cv2.resize(img, (new_w, new_h), interpolation=interp)


def lanczos_upscale(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Upscale using Lanczos (CPU fallback)."""
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)


def lanczos_to_target(img: np.ndarray, target_w: int, target_h: int, mode: str = "fit") -> np.ndarray:
    """Scale to target using Lanczos."""
    h, w = img.shape[:2]
    
    if mode == "fill":
        new_w, new_h, _, _ = compute_fill(w, h, target_w, target_h)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return center_crop(resized, target_w, target_h)
    else:
        new_w, new_h = compute_fit(w, h, target_w, target_h)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

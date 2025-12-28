"""Image I/O using OpenCV."""

import cv2
import numpy as np


def load_image(path: str) -> np.ndarray:
    """Load image as RGB uint8."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image from {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(image: np.ndarray, path: str) -> None:
    """Save RGB image."""
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

"""Synthetic test image generators for compression demos."""

import numpy as np


def generate_colored_checkerboard(size: int = 512) -> np.ndarray:
    """High-contrast checkerboard - shows blocking and chroma aliasing."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    block_size = 32
    
    for i in range(0, size, block_size):
        for j in range(0, size, block_size):
            block_idx = (i // block_size + j // block_size) % 2
            if block_idx == 0:
                img[i:i+block_size, j:j+block_size] = [30, 30, 30]
            else:
                img[i:i+block_size, j:j+block_size] = [220, 220, 220]
    
    return img


def generate_thin_stripes(size: int = 512, stripe_width: int = 4) -> np.ndarray:
    """Fine vertical stripes - shows aliasing from subsampling."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    for j in range(size):
        if (j // stripe_width) % 2 == 0:
            img[:, j] = [200, 60, 60]
        else:
            img[:, j] = [60, 180, 200]
    
    return img


def generate_gradient(size: int = 512) -> np.ndarray:
    """Smooth diagonal gradient - reveals banding from quantization."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    
    for i in range(size):
        for j in range(size):
            t = (i + j) / (2 * size - 2)
            img[i, j] = [
                40 + t * 180,
                60 + t * 140,
                120 + t * 100
            ]
    
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_text_edges(size: int = 512) -> np.ndarray:
    """Sharp geometric shapes - shows ringing and edge artifacts."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 245
    
    margin = size // 10
    bar_height = size // 16
    
    # Horizontal bars of varying thickness
    y = margin
    for thickness in [bar_height, bar_height // 2, bar_height // 4, 2]:
        img[y:y + thickness, margin:size - margin] = [25, 25, 25]
        y += thickness + margin // 2
    
    # Vertical bars
    x = margin
    for thickness in [bar_height, bar_height // 2, bar_height // 4, 2]:
        img[size // 2 + margin:size - margin, x:x + thickness] = [25, 25, 25]
        x += thickness + margin // 2
    
    # Diagonal line
    for i in range(size // 4):
        y_pos = size // 2 + margin + i
        x_pos = size // 2 + i
        if y_pos < size - margin and x_pos < size - margin:
            img[y_pos:y_pos + 3, x_pos:x_pos + 3] = [25, 25, 25]
    
    return img


def generate_chroma_stripes(size: int = 512) -> np.ndarray:
    """Saturated color bars - shows chroma bleeding and subsampling effects."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    colors = [
        [180, 40, 40],    # Red
        [40, 160, 40],    # Green
        [40, 80, 180],    # Blue
        [180, 180, 40],   # Yellow
        [180, 40, 180],   # Magenta
        [40, 180, 180],   # Cyan
        [200, 120, 40],   # Orange
        [120, 40, 180],   # Purple
    ]
    
    stripe_width = size // len(colors)
    
    for i, color in enumerate(colors):
        x_start = i * stripe_width
        x_end = (i + 1) * stripe_width if i < len(colors) - 1 else size
        img[:, x_start:x_end] = color
    
    return img


def generate_photo(size: int = 512) -> np.ndarray:
    """Natural scene with sky, mountains, and ground - realistic test image."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    
    horizon = int(size * 0.45)
    mountain_base = int(size * 0.55)
    
    # Sky gradient
    for i in range(horizon):
        t = i / horizon
        img[i, :] = [
            180 - t * 60,   # R: light to medium blue
            210 - t * 80,   # G
            240 - t * 40    # B
        ]
    
    # Mountains
    np.random.seed(123)
    mountain_heights = np.zeros(size)
    for freq in [8, 16, 32, 64]:
        phase = np.random.rand() * 2 * np.pi
        amplitude = (size * 0.15) / (freq / 8)
        mountain_heights += amplitude * np.sin(np.linspace(0, freq * np.pi, size) + phase)
    
    mountain_heights = mountain_heights - mountain_heights.min()
    mountain_heights = mountain_heights / mountain_heights.max() * (mountain_base - horizon - 20)
    
    for j in range(size):
        peak = int(horizon + 20 + mountain_heights[j])
        for i in range(horizon, mountain_base):
            if i < peak:
                depth = (i - horizon) / (peak - horizon)
                img[i, j] = [70 + depth * 30, 80 + depth * 20, 100 + depth * 10]
            else:
                img[i, j] = [90, 95, 85]
    
    # Ground with texture
    for i in range(mountain_base, size):
        t = (i - mountain_base) / (size - mountain_base)
        for j in range(size):
            noise = np.random.rand() * 15 - 7.5
            img[i, j] = [
                60 + t * 40 + noise,
                100 + t * 30 + noise,
                50 + t * 20 + noise
            ]
    
    # Sun glow
    sun_x, sun_y = size // 4, size // 6
    sun_radius = size // 10
    for i in range(max(0, sun_y - sun_radius * 2), min(horizon, sun_y + sun_radius * 2)):
        for j in range(max(0, sun_x - sun_radius * 2), min(size, sun_x + sun_radius * 2)):
            dist = np.sqrt((i - sun_y) ** 2 + (j - sun_x) ** 2)
            if dist < sun_radius * 1.5:
                glow = max(0, 1 - (dist / (sun_radius * 1.5)) ** 2)
                img[i, j] = img[i, j] * (1 - glow * 0.7) + np.array([255, 240, 200]) * glow * 0.7
    
    return np.clip(img, 0, 255).astype(np.uint8)


def generate_demo_image(key: str) -> np.ndarray | None:
    """Generate demo image by key."""
    generators = {
        "photo": lambda: generate_photo(512),
        "text_edges": lambda: generate_text_edges(512),
        "gradient": lambda: generate_gradient(512),
        "checkerboard": lambda: generate_colored_checkerboard(512),
        "chroma_stripes": lambda: generate_chroma_stripes(512),
    }
    
    if key in generators:
        return generators[key]()
    
    return None

"""Device detection for torch/CUDA."""

TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None


def detect_device() -> tuple[str, str | None, str]:
    """Detect compute device. Returns (device, gpu_name, reason)."""
    if not TORCH_AVAILABLE:
        return ("cpu", None, "PyTorch not installed")
    
    try:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return ("cuda", gpu_name, "CUDA GPU detected")
        return ("cpu", None, "No CUDA GPU available")
    except Exception as e:
        return ("cpu", None, f"CUDA detection failed: {e}")


def get_device_display_string() -> str:
    """Get formatted device string for UI."""
    device, gpu_name, reason = detect_device()
    if device == "cuda" and gpu_name:
        return f"{gpu_name} (CUDA)"
    return f"CPU ({reason})"

"""System information collection for System Info dialog."""

import sys
import platform
import re
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Try to import optional dependencies
PSUTIL_AVAILABLE = False
PYNVML_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    pass

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pass


def _get_windows_os_name() -> tuple[str, int]:
    """
    Get correct Windows OS name (10 vs 11) based on build number.
    
    Returns:
        (os_display_string, build_number)
    """
    try:
        # Get build number
        build = 0
        try:
            # Preferred method: sys.getwindowsversion()
            winver = sys.getwindowsversion()
            build = winver.build
        except AttributeError:
            # Fallback: parse platform.version()
            version_str = platform.version()
            # Format is typically "10.0.22631"
            parts = version_str.split(".")
            if len(parts) >= 3:
                build = int(parts[2])
        
        # Windows 11 starts at build 22000
        if build >= 22000:
            os_name = "Windows 11"
        else:
            os_name = "Windows 10"
        
        return (f"{os_name} (build {build})", build)
    except Exception:
        # Fallback
        return (f"Windows (build unknown)", 0)


def _get_cpu_model_windows() -> str:
    """
    Get CPU model name on Windows using multiple methods.
    
    Priority:
    1. Windows Registry (fast, no admin required)
    2. PowerShell CIM query
    3. platform.processor() fallback
    """
    cpu_model = "Unknown CPU"
    
    # Method 1: Windows Registry
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        cpu_model, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        winreg.CloseKey(key)
        
        # Clean up the string
        cpu_model = _normalize_cpu_string(cpu_model)
        if cpu_model and cpu_model != "Unknown CPU":
            return cpu_model
    except Exception:
        pass
    
    # Method 2: PowerShell CIM (slower but reliable)
    try:
        result = subprocess.run(
            [
                "powershell", "-NoProfile", "-Command",
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)"
            ],
            capture_output=True,
            text=True,
            timeout=5,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.returncode == 0 and result.stdout.strip():
            cpu_model = _normalize_cpu_string(result.stdout.strip())
            if cpu_model and cpu_model != "Unknown CPU":
                return cpu_model
    except Exception:
        pass
    
    # Method 3: platform.processor() fallback
    try:
        cpu_model = platform.processor()
        if cpu_model:
            cpu_model = _normalize_cpu_string(cpu_model)
    except Exception:
        pass
    
    return cpu_model or "Unknown CPU"


def _get_cpu_model_linux() -> str:
    """Get CPU model name on Linux."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    return _normalize_cpu_string(line.split(":")[1].strip())
    except Exception:
        pass
    
    # Fallback
    try:
        return _normalize_cpu_string(platform.processor()) or "Unknown CPU"
    except Exception:
        return "Unknown CPU"


def _get_cpu_model_mac() -> str:
    """Get CPU model name on macOS."""
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            return _normalize_cpu_string(result.stdout.strip())
    except Exception:
        pass
    
    # Fallback
    try:
        return _normalize_cpu_string(platform.processor()) or "Unknown CPU"
    except Exception:
        return "Unknown CPU"


def _normalize_cpu_string(s: str) -> str:
    """Normalize CPU model string (strip, collapse spaces, etc.)."""
    if not s:
        return "Unknown CPU"
    
    # Strip whitespace
    s = s.strip()
    
    # Collapse multiple spaces
    s = re.sub(r'\s+', ' ', s)
    
    # Remove some common redundant parts (optional)
    # s = s.replace("(R)", "®").replace("(TM)", "™")
    
    return s if s else "Unknown CPU"


def get_cpu_model() -> str:
    """Get CPU model name (cross-platform)."""
    system = platform.system()
    
    if system == "Windows":
        return _get_cpu_model_windows()
    elif system == "Linux":
        return _get_cpu_model_linux()
    elif system == "Darwin":
        return _get_cpu_model_mac()
    else:
        try:
            return _normalize_cpu_string(platform.processor()) or "Unknown CPU"
        except Exception:
            return "Unknown CPU"


def get_os_display() -> str:
    """Get OS display string (cross-platform, with correct Windows 10/11 detection)."""
    system = platform.system()
    
    if system == "Windows":
        os_display, _ = _get_windows_os_name()
        return os_display
    elif system == "Darwin":
        try:
            mac_ver = platform.mac_ver()[0]
            return f"macOS {mac_ver}"
        except Exception:
            return "macOS"
    elif system == "Linux":
        try:
            # Try to get distro info
            import distro
            return f"{distro.name()} {distro.version()}"
        except ImportError:
            return f"Linux {platform.release()}"
    else:
        return f"{system} {platform.release()}"


@dataclass
class StaticSystemInfo:
    """Static system information (collected once)."""
    # OS
    os_name: str = "N/A"
    os_release: str = "N/A"
    os_version: str = "N/A"
    os_display: str = "N/A"
    
    # Python
    python_version: str = "N/A"
    python_requirement: str = "3.11+"
    
    # CPU
    cpu_model: str = "N/A"
    cpu_cores_physical: int = 0
    cpu_cores_logical: int = 0
    
    # RAM
    ram_total_bytes: int = 0
    ram_total_display: str = "N/A"
    
    # Disk
    disk_path: str = "N/A"
    disk_total_bytes: int = 0
    disk_total_display: str = "N/A"
    
    # GPU
    gpu_name: str = "No CUDA GPU detected"
    cuda_available: bool = False
    
    # Software stack
    torch_installed: bool = False
    torch_version: str = "N/A"
    realesrgan_installed: bool = False
    basicsr_installed: bool = False
    weights_status: str = "Unknown"
    weights_path: str = "N/A"


@dataclass
class LiveSystemInfo:
    """Live system information (updated periodically)."""
    # CPU
    cpu_percent: float = 0.0
    
    # RAM
    ram_used_bytes: int = 0
    ram_used_display: str = "N/A"
    ram_percent: float = 0.0
    
    # Disk
    disk_free_bytes: int = 0
    disk_free_display: str = "N/A"
    disk_used_percent: float = 0.0
    
    # GPU (via pynvml)
    gpu_util_percent: float = -1  # -1 means N/A
    gpu_memory_used_bytes: int = 0
    gpu_memory_total_bytes: int = 0
    gpu_memory_display: str = "N/A"
    gpu_temp_celsius: int = -1  # -1 means N/A
    gpu_available: bool = False


def format_bytes(bytes_val: int, precision: int = 1) -> str:
    """Format bytes to human-readable string (GB/MB)."""
    if bytes_val <= 0:
        return "N/A"
    
    gb = bytes_val / (1024 ** 3)
    if gb >= 1:
        return f"{gb:.{precision}f} GB"
    
    mb = bytes_val / (1024 ** 2)
    return f"{mb:.{precision}f} MB"


def get_static_info() -> StaticSystemInfo:
    """Collect static system information (call once, cache result)."""
    info = StaticSystemInfo()
    
    # OS info (with correct Windows 10/11 detection)
    try:
        info.os_name = platform.system()
        info.os_release = platform.release()
        info.os_version = platform.version()
        info.os_display = get_os_display()
    except Exception:
        pass
    
    # Python info
    try:
        info.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    except Exception:
        pass
    
    # CPU info (with proper model detection)
    try:
        info.cpu_model = get_cpu_model()
    except Exception:
        info.cpu_model = "Unknown CPU"
    
    if PSUTIL_AVAILABLE:
        try:
            info.cpu_cores_physical = psutil.cpu_count(logical=False) or 0
            info.cpu_cores_logical = psutil.cpu_count(logical=True) or 0
        except Exception:
            pass
        
        # RAM info
        try:
            mem = psutil.virtual_memory()
            info.ram_total_bytes = mem.total
            info.ram_total_display = format_bytes(mem.total)
        except Exception:
            pass
        
        # Disk info
        try:
            # Use the path where the app is running
            app_path = Path(__file__).parent.parent
            if platform.system() == "Windows":
                # Get drive letter
                info.disk_path = str(app_path.drive) + "\\" if app_path.drive else "C:\\"
            else:
                info.disk_path = "/"
            
            disk = psutil.disk_usage(info.disk_path)
            info.disk_total_bytes = disk.total
            info.disk_total_display = format_bytes(disk.total)
        except Exception:
            pass
    
    # GPU info via torch
    try:
        from enhancement.device_detector import TORCH_AVAILABLE
        info.torch_installed = TORCH_AVAILABLE
        
        if TORCH_AVAILABLE:
            import torch
            info.torch_version = torch.__version__
            info.cuda_available = torch.cuda.is_available()
            
            if torch.cuda.is_available():
                info.gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    
    # Real-ESRGAN / basicsr
    try:
        from enhancement.device_detector import REALESRGAN_AVAILABLE
        info.realesrgan_installed = REALESRGAN_AVAILABLE
    except Exception:
        pass
    
    try:
        import basicsr
        info.basicsr_installed = True
    except ImportError:
        info.basicsr_installed = False
    except Exception:
        pass
    
    # Weights status
    try:
        from enhancement.realesrgan_upscaler import MODEL_PATH
        if MODEL_PATH.exists():
            size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            info.weights_status = f"Cached ({size_mb:.0f} MB)"
            info.weights_path = str(MODEL_PATH)
        else:
            info.weights_status = "Not downloaded"
            info.weights_path = str(MODEL_PATH)
    except Exception:
        info.weights_status = "Unknown"
    
    return info


def get_live_info(static_info: StaticSystemInfo) -> LiveSystemInfo:
    """Collect live system information (call periodically)."""
    info = LiveSystemInfo()
    
    if PSUTIL_AVAILABLE:
        # CPU usage (non-blocking)
        try:
            info.cpu_percent = psutil.cpu_percent(interval=None)
        except Exception:
            pass
        
        # RAM usage
        try:
            mem = psutil.virtual_memory()
            info.ram_used_bytes = mem.used
            info.ram_used_display = format_bytes(mem.used)
            info.ram_percent = mem.percent
        except Exception:
            pass
        
        # Disk usage
        try:
            disk = psutil.disk_usage(static_info.disk_path)
            info.disk_free_bytes = disk.free
            info.disk_free_display = format_bytes(disk.free)
            info.disk_used_percent = disk.percent
        except Exception:
            pass
    
    # GPU stats via pynvml
    if PYNVML_AVAILABLE and static_info.cuda_available:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info.gpu_util_percent = util.gpu
                info.gpu_available = True
            except Exception:
                pass
            
            # Memory
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info.gpu_memory_used_bytes = mem.used
                info.gpu_memory_total_bytes = mem.total
                info.gpu_memory_display = f"{format_bytes(mem.used, 1)} / {format_bytes(mem.total, 1)}"
                info.gpu_available = True
            except Exception:
                pass
            
            # Temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                info.gpu_temp_celsius = temp
                info.gpu_available = True
            except Exception:
                pass
            
            pynvml.nvmlShutdown()
        except Exception:
            pass
    
    return info


def generate_snapshot_text(static: StaticSystemInfo, live: LiveSystemInfo) -> str:
    """Generate a formatted text snapshot for clipboard."""
    lines = []
    
    # Header
    lines.append("JPEG-DSP Studio v1.0")
    lines.append(f"OS: {static.os_display} ({static.os_version})")
    lines.append(f"Python: {static.python_version} (requires {static.python_requirement})")
    
    # CPU
    cpu_line = f"CPU: {static.cpu_model}"
    if static.cpu_cores_physical > 0:
        cpu_line += f" | Cores: {static.cpu_cores_physical}"
    if static.cpu_cores_logical > 0:
        cpu_line += f" | Threads: {static.cpu_cores_logical}"
    cpu_line += f" | Load: {live.cpu_percent:.0f}%"
    lines.append(cpu_line)
    
    # RAM
    if static.ram_total_bytes > 0:
        lines.append(f"RAM: {live.ram_used_display} / {static.ram_total_display} ({live.ram_percent:.0f}%)")
    else:
        lines.append("RAM: N/A")
    
    # Disk
    if static.disk_total_bytes > 0:
        lines.append(f"Disk ({static.disk_path}): Free {live.disk_free_display} / {static.disk_total_display}")
    else:
        lines.append("Disk: N/A")
    
    # GPU
    gpu_line = f"GPU: {static.gpu_name}"
    gpu_line += f" | CUDA: {'Yes' if static.cuda_available else 'No'}"
    lines.append(gpu_line)
    
    # GPU live stats
    if live.gpu_available:
        gpu_stats = []
        if live.gpu_util_percent >= 0:
            gpu_stats.append(f"Util: {live.gpu_util_percent:.0f}%")
        if live.gpu_memory_total_bytes > 0:
            gpu_stats.append(f"VRAM: {live.gpu_memory_display}")
        if live.gpu_temp_celsius >= 0:
            gpu_stats.append(f"Temp: {live.gpu_temp_celsius}°C")
        if gpu_stats:
            lines.append("GPU Stats: " + " | ".join(gpu_stats))
    else:
        lines.append("GPU Stats: N/A (pynvml not installed)")
    
    # Software
    torch_str = f"v{static.torch_version}" if static.torch_installed else "Not installed"
    esrgan_str = "Installed" if static.realesrgan_installed else "Not installed"
    basicsr_str = "Installed" if static.basicsr_installed else "Not installed"
    lines.append(f"PyTorch: {torch_str} | Real-ESRGAN: {esrgan_str} | basicsr: {basicsr_str}")
    
    # Weights
    lines.append(f"Weights: {static.weights_status}")
    if static.weights_path != "N/A" and static.weights_status.startswith("Cached"):
        lines.append(f"  Path: {static.weights_path}")
    
    return "\n".join(lines)


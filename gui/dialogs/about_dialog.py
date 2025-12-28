"""Enhanced About dialog with system info copy and app icon."""

import sys
import platform
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextBrowser, QApplication, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QIcon

# App metadata
APP_VERSION = "1.0"
APP_NAME = "JPEG-DSP Studio"
PYTHON_REQUIREMENT = "3.11+"


class AboutDialog(QDialog):
    """
    Enhanced About dialog with:
    - App icon in header
    - System information display
    - Copy System Info button
    - AI Acceleration status
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle(f"About {APP_NAME}")
        self.setMinimumSize(500, 480)
        self.setMaximumSize(600, 600)
        
        self._system_info = self._gather_system_info()
        self._init_ui()
        self._apply_style()
    
    def _gather_system_info(self) -> dict:
        """Gather all system information."""
        info = {
            "app_version": APP_VERSION,
            "app_name": APP_NAME,
            "os": f"{platform.system()} {platform.release()}",
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "torch_available": False,
            "torch_version": None,
            "cuda_available": False,
            "gpu_name": None,
            "realesrgan_available": False,
            "weights_path": None,
            "weights_status": "Not downloaded",
        }
        
        try:
            from enhancement.device_detector import detect_device, TORCH_AVAILABLE
            from enhancement.realesrgan_upscaler import REALESRGAN_AVAILABLE, MODEL_PATH
            
            info["torch_available"] = TORCH_AVAILABLE
            info["realesrgan_available"] = REALESRGAN_AVAILABLE
            
            if TORCH_AVAILABLE:
                import torch
                info["torch_version"] = torch.__version__
                info["cuda_available"] = torch.cuda.is_available()
                
                if torch.cuda.is_available():
                    info["gpu_name"] = torch.cuda.get_device_name(0)
            
            if REALESRGAN_AVAILABLE:
                info["weights_path"] = str(MODEL_PATH)
                if MODEL_PATH.exists():
                    info["weights_status"] = f"Cached ({MODEL_PATH.stat().st_size // (1024*1024)}MB)"
                else:
                    info["weights_status"] = "Not downloaded"
        except Exception:
            pass
        
        return info
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header with icon and title
        header = QHBoxLayout()
        header.setSpacing(15)
        
        # App icon
        icon_label = QLabel()
        icon_paths = [
            Path(__file__).parent.parent / "icon.png",
            Path(__file__).parent.parent.parent / "icon.png",
            Path(__file__).parent.parent / "assets" / "icon.png",
            Path(__file__).parent.parent / "assets" / "icon.ico",
        ]
        icon_found = False
        for icon_path in icon_paths:
            if icon_path.exists():
                pixmap = QPixmap(str(icon_path)).scaled(
                    64, 64, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                icon_label.setPixmap(pixmap)
                icon_found = True
                break
        
        if not icon_found:
            # Fallback icon
            icon_label.setText("🖼️")
            icon_label.setStyleSheet("font-size: 48px;")
        
        header.addWidget(icon_label)
        
        # Title and version
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        
        title = QLabel(f"<b style='font-size: 18px;'>{APP_NAME}</b>")
        title_layout.addWidget(title)
        
        version = QLabel(f"Version {APP_VERSION}")
        version.setStyleSheet("color: #888; font-size: 12px;")
        title_layout.addWidget(version)
        
        subtitle = QLabel("JPEG-Inspired Image Compression System")
        subtitle.setStyleSheet("color: #aaa; font-size: 11px;")
        title_layout.addWidget(subtitle)
        
        title_layout.addStretch()
        header.addLayout(title_layout)
        header.addStretch()
        
        layout.addLayout(header)
        
        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setStyleSheet("background-color: #555;")
        layout.addWidget(sep1)
        
        # Content browser
        content = QTextBrowser()
        content.setOpenExternalLinks(True)
        content.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e1e;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        
        # Build content HTML
        info = self._system_info
        
        # AI Acceleration status
        ai_status_items = []
        if info["torch_available"]:
            ai_status_items.append(f"PyTorch {info['torch_version']}")
            if info["cuda_available"] and info["gpu_name"]:
                ai_status_items.append(f"<span style='color: #4ade80;'>✓ CUDA ({info['gpu_name']})</span>")
            else:
                ai_status_items.append("<span style='color: #888;'>✗ CUDA not available</span>")
        else:
            ai_status_items.append("<span style='color: #888;'>✗ PyTorch not installed</span>")
        
        if info["realesrgan_available"]:
            ai_status_items.append("<span style='color: #4ade80;'>✓ Real-ESRGAN installed</span>")
        else:
            ai_status_items.append("<span style='color: #888;'>✗ Real-ESRGAN not installed</span>")
        
        ai_status_html = "<br>".join(ai_status_items)
        
        content_html = f"""
        <p><b>Features:</b></p>
        <ul style="margin-top: 5px;">
            <li>DCT-based compression with adjustable quality</li>
            <li>Chroma subsampling (4:4:4, 4:2:2, 4:2:0)</li>
            <li>Anti-aliasing prefilter for chroma</li>
            <li>Real-ESRGAN ×4 AI enhancement (auto when CUDA available)</li>
            <li>CPU Lanczos upscaling (always available)</li>
        </ul>
        
        <hr style="border-color: #444;">
        
        <p><b>Python:</b> {info['python']} (requires {PYTHON_REQUIREMENT})</p>
        
        <hr style="border-color: #444;">
        
        <p><b>AI Acceleration (Auto-detected):</b></p>
        <p style="margin-left: 10px;">{ai_status_html}</p>
        <p style="color: #666; font-size: 10px;">Requires: torch, realesrgan, basicsr</p>
        
        <hr style="border-color: #444;">
        
        <p style="color: #888; text-align: center;"><i>DSP Course Project</i></p>
        """
        
        content.setHtml(content_html)
        layout.addWidget(content)
        
        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)
        
        copy_btn = QPushButton("📋 Copy System Info")
        copy_btn.setToolTip("Copy system information to clipboard")
        copy_btn.clicked.connect(self._copy_system_info)
        btn_layout.addWidget(copy_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setDefault(True)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
    
    def _copy_system_info(self):
        """Copy system information to clipboard."""
        info = self._system_info
        
        lines = [
            f"{info['app_name']} v{info['app_version']}",
            f"OS: {info['os']}",
            f"Python: {info['python']}",
            "",
            "AI Acceleration:",
        ]
        
        if info["torch_available"]:
            lines.append(f"  PyTorch: {info['torch_version']}")
            lines.append(f"  CUDA: {'Yes' if info['cuda_available'] else 'No'}")
            if info["gpu_name"]:
                lines.append(f"  GPU: {info['gpu_name']}")
        else:
            lines.append("  PyTorch: Not installed")
        
        lines.append(f"  Real-ESRGAN: {'Installed' if info['realesrgan_available'] else 'Not installed'}")
        lines.append(f"  Weights: {info['weights_status']}")
        
        if info["weights_path"]:
            lines.append(f"  Weights Path: {info['weights_path']}")
        
        clipboard_text = "\n".join(lines)
        
        clipboard = QApplication.clipboard()
        clipboard.setText(clipboard_text)
        
        # Show brief feedback by updating button
        sender = self.sender()
        if sender:
            original_text = sender.text()
            sender.setText("✓ Copied!")
            sender.setEnabled(False)
            
            from PySide6.QtCore import QTimer
            QTimer.singleShot(1500, lambda: self._reset_copy_button(sender, original_text))
    
    def _reset_copy_button(self, btn, text):
        """Reset copy button text after feedback."""
        btn.setText(text)
        btn.setEnabled(True)
    
    def _apply_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px 16px;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #333;
            }
            QPushButton:disabled {
                background-color: #3a5a3a;
                color: #8f8;
            }
        """)


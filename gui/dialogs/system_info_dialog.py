"""System Information dialog with live stats and modern card layout."""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QFrame, QProgressBar, QApplication,
    QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QFontMetrics

from gui.system_info import (
    get_static_info, get_live_info, generate_snapshot_text,
    StaticSystemInfo, LiveSystemInfo, format_bytes,
    PSUTIL_AVAILABLE, PYNVML_AVAILABLE
)
from gui.widgets.toast import ToastNotification


def elide_text(text: str, max_chars: int) -> str:
    """Truncate text with ellipsis if too long."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - 1] + "…"


class StatCard(QFrame):
    """Compact modern card for displaying a stat with progress bar."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._init_ui()
        self._apply_style()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(4)
        
        # Title row
        title_row = QHBoxLayout()
        title_row.setSpacing(6)
        
        self._title_label = QLabel(self._title.upper())
        self._title_label.setStyleSheet(
            "color: #4a9eff; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )
        title_row.addWidget(self._title_label)
        title_row.addStretch()
        
        layout.addLayout(title_row)
        
        # Value row: big number + unit
        value_row = QHBoxLayout()
        value_row.setSpacing(4)
        value_row.setAlignment(Qt.AlignmentFlag.AlignBaseline)
        
        self._value_label = QLabel("—")
        font = QFont()
        font.setPointSize(22)
        font.setBold(True)
        self._value_label.setFont(font)
        self._value_label.setStyleSheet("color: #e0e0e0;")
        value_row.addWidget(self._value_label)
        
        self._unit_label = QLabel("")
        self._unit_label.setStyleSheet("color: #888; font-size: 12px; margin-bottom: 4px;")
        value_row.addWidget(self._unit_label)
        value_row.addStretch()
        
        layout.addLayout(value_row)
        
        # Progress bar
        self._progress = QProgressBar()
        self._progress.setFixedHeight(4)
        self._progress.setTextVisible(False)
        self._progress.setStyleSheet("""
            QProgressBar {
                background-color: #333;
                border: none;
                border-radius: 2px;
            }
            QProgressBar::chunk {
                background-color: #4ade80;
                border-radius: 2px;
            }
        """)
        layout.addWidget(self._progress)
        
        # Subtitle (model/description) - single line with ellipsis
        self._subtitle_label = QLabel("")
        self._subtitle_label.setStyleSheet("color: #999; font-size: 10px;")
        self._subtitle_label.setWordWrap(False)
        layout.addWidget(self._subtitle_label)
        
        # Detail line
        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet("color: #666; font-size: 9px;")
        layout.addWidget(self._detail_label)
        
        # Extra info line (for GPU stats unavailable message)
        self._extra_label = QLabel("")
        self._extra_label.setStyleSheet("color: #666; font-size: 9px; font-style: italic;")
        self._extra_label.setWordWrap(True)
        self._extra_label.setVisible(False)
        layout.addWidget(self._extra_label)
    
    def _apply_style(self):
        self.setStyleSheet("""
            StatCard {
                background-color: #282828;
                border: 1px solid #3a3a3a;
                border-radius: 12px;
            }
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(115)
    
    def set_value(self, value: str, unit: str = ""):
        self._value_label.setText(value)
        self._unit_label.setText(unit)
        self._value_label.setVisible(True)
    
    def hide_value(self):
        """Hide the big value display (for GPU N/A state)."""
        self._value_label.setVisible(False)
        self._unit_label.setVisible(False)
    
    def set_subtitle(self, text: str, max_chars: int = 38):
        """Set subtitle with auto-ellipsis."""
        self._subtitle_label.setText(elide_text(text, max_chars))
        self._subtitle_label.setToolTip(text if len(text) > max_chars else "")
        self._subtitle_label.setVisible(bool(text))
    
    def set_detail(self, text: str):
        self._detail_label.setText(text)
        self._detail_label.setVisible(bool(text))
    
    def set_extra_info(self, text: str, tooltip: str = ""):
        """Set extra info line (e.g., pynvml missing message)."""
        self._extra_label.setText(text)
        self._extra_label.setVisible(bool(text))
        if tooltip:
            self._extra_label.setToolTip(tooltip)
    
    def set_progress(self, value: float):
        self._progress.setValue(int(max(0, min(100, value))))
    
    def hide_progress(self):
        """Hide progress bar."""
        self._progress.setVisible(False)
    
    def show_progress(self):
        """Show progress bar."""
        self._progress.setVisible(True)
    
    def set_progress_color(self, percent: float, invert: bool = False):
        """Set progress bar color based on percentage (green/yellow/red)."""
        if invert:
            # For disk: low free = bad
            if percent < 15:
                color = "#ef4444"  # red
            elif percent < 30:
                color = "#fbbf24"  # yellow
            else:
                color = "#4ade80"  # green
        else:
            # For CPU/RAM/GPU: high usage = warning
            if percent < 60:
                color = "#4ade80"  # green
            elif percent < 85:
                color = "#fbbf24"  # yellow
            else:
                color = "#ef4444"  # red
        
        self._progress.setStyleSheet(f"""
            QProgressBar {{
                background-color: #333;
                border: none;
                border-radius: 2px;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 2px;
            }}
        """)


class SoftwareCard(QFrame):
    """Compact card for software stack information."""
    
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self._title = title
        self._rows = []
        self._init_ui()
        self._apply_style()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(3)
        
        # Title
        title_label = QLabel(self._title.upper())
        title_label.setStyleSheet(
            "color: #4a9eff; font-size: 10px; font-weight: bold; letter-spacing: 1px;"
        )
        layout.addWidget(title_label)
        
        layout.addSpacing(4)
        
        # Content grid
        self._grid = QGridLayout()
        self._grid.setSpacing(3)
        self._grid.setColumnStretch(1, 1)
        layout.addLayout(self._grid)
        
        layout.addStretch()
    
    def _apply_style(self):
        self.setStyleSheet("""
            SoftwareCard {
                background-color: #282828;
                border: 1px solid #3a3a3a;
                border-radius: 12px;
            }
        """)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(115)
    
    def add_row(self, label: str, value: str, is_ok: bool = True):
        """Add a label: value row."""
        row = self._grid.rowCount()
        
        lbl = QLabel(label)
        lbl.setStyleSheet("color: #888; font-size: 10px;")
        self._grid.addWidget(lbl, row, 0, Qt.AlignmentFlag.AlignLeft)
        
        val = QLabel(value)
        color = "#4ade80" if is_ok else "#888"
        val.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: 500;")
        self._grid.addWidget(val, row, 1, Qt.AlignmentFlag.AlignLeft)
        
        self._rows.append((lbl, val))


class SystemInfoDialog(QDialog):
    """
    Modern System Information dialog with live stats.
    
    Features:
    - Compact 2-row dashboard layout
    - CPU, RAM, Disk, GPU stats with progress bars
    - Software stack status
    - Copy Snapshot button
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("System Information")
        self.setMinimumWidth(580)
        self.setMaximumWidth(700)
        
        # Collect static info once
        self._static_info = get_static_info()
        self._live_info = LiveSystemInfo()
        
        # Track GPU state
        self._gpu_has_live_stats = False
        
        self._init_ui()
        self._apply_style()
        self._populate_static_info()
        
        # Start live update timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._update_live_info)
        self._timer.start(1000)
        
        # Initial live update
        self._update_live_info()
        
        # Adjust size to content
        self.adjustSize()
        self.setFixedHeight(self.sizeHint().height())
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        
        # Header row
        header = QHBoxLayout()
        header.setSpacing(8)
        
        # Title + subtitle
        title_col = QVBoxLayout()
        title_col.setSpacing(2)
        
        title = QLabel("System Information")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #e0e0e0;")
        title_col.addWidget(title)
        
        subtitle = QLabel("Live stats · Updates every 1s")
        subtitle.setStyleSheet("font-size: 10px; color: #666;")
        title_col.addWidget(subtitle)
        
        header.addLayout(title_col)
        header.addStretch()
        
        # Copy button in header
        self._copy_btn = QPushButton("📋 Copy Snapshot")
        self._copy_btn.setToolTip("Copy all system info to clipboard")
        self._copy_btn.clicked.connect(self._copy_snapshot)
        self._copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                border: 1px solid #3b82f6;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)
        header.addWidget(self._copy_btn)
        
        layout.addLayout(header)
        
        # Row 1: CPU | Memory | Disk
        row1 = QHBoxLayout()
        row1.setSpacing(10)
        
        self._cpu_card = StatCard("CPU")
        row1.addWidget(self._cpu_card)
        
        self._ram_card = StatCard("Memory")
        row1.addWidget(self._ram_card)
        
        self._disk_card = StatCard("Disk")
        row1.addWidget(self._disk_card)
        
        layout.addLayout(row1)
        
        # Row 2: GPU | Software Stack
        row2 = QHBoxLayout()
        row2.setSpacing(10)
        
        self._gpu_card = StatCard("GPU")
        row2.addWidget(self._gpu_card, 1)
        
        self._software_card = SoftwareCard("Software Stack")
        row2.addWidget(self._software_card, 1)
        
        layout.addLayout(row2)
        
        # Footer (minimal)
        footer = QHBoxLayout()
        footer.setSpacing(8)
        
        # Dependency warning (if any)
        if not PSUTIL_AVAILABLE:
            warn = QLabel("⚠ psutil not installed — some stats unavailable")
            warn.setStyleSheet("color: #fbbf24; font-size: 10px;")
            warn.setToolTip("Install psutil for CPU/RAM/Disk stats")
            footer.addWidget(warn)
        
        footer.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 6px 16px;
                font-size: 11px;
                color: #e0e0e0;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
        """)
        footer.addWidget(close_btn)
        
        layout.addLayout(footer)
    
    def _apply_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
            QLabel {
                color: #e0e0e0;
            }
        """)
    
    def _populate_static_info(self):
        """Populate cards with static information."""
        info = self._static_info
        
        # CPU - single line with ellipsis
        self._cpu_card.set_subtitle(info.cpu_model, max_chars=36)
        if info.cpu_cores_physical > 0:
            self._cpu_card.set_detail(f"{info.cpu_cores_physical} cores · {info.cpu_cores_logical} threads")
        
        # RAM
        self._ram_card.set_detail(f"Total: {info.ram_total_display}")
        
        # Disk
        self._disk_card.set_subtitle(info.disk_path, max_chars=20)
        self._disk_card.set_detail(f"Total: {info.disk_total_display}")
        
        # GPU
        self._gpu_card.set_subtitle(info.gpu_name, max_chars=34)
        
        cuda_text = "CUDA: Yes" if info.cuda_available else "CUDA: No"
        
        if info.cuda_available and not PYNVML_AVAILABLE:
            # GPU available but no live stats - hide big value, show compact stats line
            self._gpu_card.hide_value()
            self._gpu_card.set_detail(cuda_text)
            self._gpu_card.set_extra_info(
                "Util: —  VRAM: —  Temp: —\nInstall pynvml for live stats",
                "Install pynvml to enable GPU utilization, VRAM, and temperature monitoring"
            )
            self._gpu_card.hide_progress()
            self._gpu_has_live_stats = False
        elif not info.cuda_available:
            # No CUDA GPU
            self._gpu_card.hide_value()
            self._gpu_card.set_detail(cuda_text)
            self._gpu_card.hide_progress()
            self._gpu_has_live_stats = False
        else:
            # GPU with live stats available
            self._gpu_card.set_detail(cuda_text)
            self._gpu_has_live_stats = True
        
        # Software - use corrected os_display with explicit status text
        self._software_card.add_row("OS:", info.os_display, True)
        self._software_card.add_row("Python:", f"{info.python_version}", True)
        self._software_card.add_row(
            "PyTorch:",
            f"v{info.torch_version}" if info.torch_installed else "Not installed",
            info.torch_installed
        )
        self._software_card.add_row(
            "Real-ESRGAN:",
            "Installed" if info.realesrgan_installed else "Not installed",
            info.realesrgan_installed
        )
        self._software_card.add_row(
            "Weights:",
            info.weights_status if info.weights_status.startswith("Cached") else "Not downloaded",
            info.weights_status.startswith("Cached")
        )
    
    def _update_live_info(self):
        """Update live statistics."""
        self._live_info = get_live_info(self._static_info)
        live = self._live_info
        
        # CPU
        self._cpu_card.set_value(f"{live.cpu_percent:.0f}", "%")
        self._cpu_card.set_progress(live.cpu_percent)
        self._cpu_card.set_progress_color(live.cpu_percent)
        
        # RAM
        self._ram_card.set_value(f"{live.ram_percent:.0f}", "%")
        self._ram_card.set_subtitle(f"{live.ram_used_display} / {self._static_info.ram_total_display}")
        self._ram_card.set_progress(live.ram_percent)
        self._ram_card.set_progress_color(live.ram_percent)
        
        # Disk - show free space with percentage
        free_percent = 100 - live.disk_used_percent
        free_val = live.disk_free_display.replace(" GB", "").replace(" MB", "")
        free_unit = "GB free" if "GB" in live.disk_free_display else "MB free"
        self._disk_card.set_value(free_val, free_unit)
        self._disk_card.set_subtitle(f"Free: {free_percent:.0f}%")
        self._disk_card.set_progress(free_percent)
        self._disk_card.set_progress_color(free_percent, invert=True)
        
        # GPU - only update if live stats available
        if live.gpu_available and self._gpu_has_live_stats:
            if live.gpu_util_percent >= 0:
                self._gpu_card.set_value(f"{live.gpu_util_percent:.0f}", "%")
                self._gpu_card.show_progress()
                self._gpu_card.set_progress(live.gpu_util_percent)
                self._gpu_card.set_progress_color(live.gpu_util_percent)
            
            # Build detail line with live stats
            details = ["CUDA: Yes"]
            if live.gpu_memory_total_bytes > 0:
                details.append(f"VRAM: {live.gpu_memory_display}")
            if live.gpu_temp_celsius >= 0:
                details.append(f"Temp: {live.gpu_temp_celsius}°C")
            
            self._gpu_card.set_detail(" · ".join(details))
        elif not self._gpu_has_live_stats:
            # Keep static display, no big value
            self._gpu_card.set_value("—", "")
    
    def _copy_snapshot(self):
        """Copy system snapshot to clipboard."""
        text = generate_snapshot_text(self._static_info, self._live_info)
        
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        
        # Toast notification
        ToastNotification.show_toast(self, "System snapshot copied", "success")
        
        # Button feedback
        original = self._copy_btn.text()
        self._copy_btn.setText("✓ Copied!")
        self._copy_btn.setEnabled(False)
        self._copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #16a34a;
                border: 1px solid #22c55e;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
                color: white;
            }
        """)
        
        QTimer.singleShot(1500, lambda: self._reset_copy_btn(original))
    
    def _reset_copy_btn(self, text: str):
        self._copy_btn.setText(text)
        self._copy_btn.setEnabled(True)
        self._copy_btn.setStyleSheet("""
            QPushButton {
                background-color: #2563eb;
                border: 1px solid #3b82f6;
                border-radius: 6px;
                padding: 6px 12px;
                font-size: 11px;
                font-weight: bold;
                color: white;
            }
            QPushButton:hover {
                background-color: #3b82f6;
            }
            QPushButton:pressed {
                background-color: #1d4ed8;
            }
        """)
    
    def closeEvent(self, event):
        """Stop timer on close."""
        self._timer.stop()
        super().closeEvent(event)

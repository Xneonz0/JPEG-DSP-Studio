"""Enhancement Tab - GPU-accelerated image upscaling with Real-ESRGAN."""

import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QPushButton, QLabel, QComboBox, QCheckBox,
    QGroupBox, QSplitter, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QThread

from enhancement.device_detector import detect_device, get_device_display_string
from enhancement.target_scaler import TARGET_RESOLUTIONS
from enhancement.realesrgan_upscaler import (
    REALESRGAN_AVAILABLE,
    get_availability_message,
    MODEL_PATH,
)
from gui.widgets.image_viewer import ImageViewerWithControls
from gui.widgets.collapsible_group import CollapsibleGroupBox
from gui.widgets.toast import ToastNotification
from gui.widgets.styled_combobox import style_combobox
from gui.enhancement_worker import EnhancementWorker, EnhancementResult
from utils.image_io import load_image, save_image


# Tile size options
TILE_SIZE_OPTIONS = [
    (0, "Auto (512)"),
    (256, "256 (Low VRAM)"),
    (512, "512 (Default)"),
    (1024, "1024 (High VRAM)"),
]


class EnhancementTab(QWidget):
    """
    Enhancement tab for GPU-accelerated image upscaling.
    
    Features:
    - Load image or use reconstructed from compression tab
    - Target resolution selection (1080p, 1440p, 4K)
    - Fit/Fill mode
    - GPU acceleration with Real-ESRGAN
    - CPU Lanczos fallback
    - FP16 and tile size options
    """
    
    def __init__(self, compression_tab=None, parent=None):
        super().__init__(parent)
        
        # Reference to compression tab for "Use Reconstructed" feature
        self._compression_tab = compression_tab
        
        # State
        self._image = None
        self._image_path = None
        self._result = None
        self._thread = None
        self._worker = None
        self._source_mode = "loaded"  # "loaded" or "reconstructed"
        
        self._init_ui()
        self._update_device_info()
    
    def _init_ui(self):
        """Initialize the UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        
        # Main horizontal splitter
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # ========== LEFT: Control Panel ==========
        control_panel = self._create_control_panel()
        main_splitter.addWidget(control_panel)
        
        # ========== CENTER: Image Viewers ==========
        viewer_container = self._create_viewer_area()
        main_splitter.addWidget(viewer_container)
        
        # ========== RIGHT: Metrics Panel ==========
        metrics_panel = self._create_metrics_panel()
        main_splitter.addWidget(metrics_panel)
        
        # Splitter proportions
        main_splitter.setStretchFactor(0, 0)  # Control panel - fixed
        main_splitter.setStretchFactor(1, 1)  # Viewers - stretch
        main_splitter.setStretchFactor(2, 0)  # Metrics - fixed
        main_splitter.setSizes([250, 600, 250])
        
        main_layout.addWidget(main_splitter)
    
    def _create_control_panel(self) -> QWidget:
        """Create left control panel with collapsible sections."""
        panel = QWidget()
        panel.setMinimumWidth(200)
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(4)
        
        # === Source Selection (Collapsible) ===
        source_group = CollapsibleGroupBox("Source Image", settings_key="enh_source")
        
        # Load button row
        load_row = QWidget()
        load_layout = QHBoxLayout(load_row)
        load_layout.setContentsMargins(0, 0, 0, 0)
        load_layout.setSpacing(4)
        
        self._load_btn = QPushButton("Load Image...")
        self._load_btn.clicked.connect(self._on_load_image)
        load_layout.addWidget(self._load_btn)
        
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setToolTip("Clear image and reset session")
        self._clear_btn.clicked.connect(self.clear_enhancement_session)
        self._clear_btn.setEnabled(False)
        self._clear_btn.setMaximumWidth(60)
        self._clear_btn.setStyleSheet("""
            QPushButton { background-color: #3a2a2a; border-color: #554444; }
            QPushButton:hover { background-color: #4a3a3a; }
            QPushButton:disabled { background-color: #2a2a2a; color: #555; }
        """)
        load_layout.addWidget(self._clear_btn)
        
        source_group.add_widget(load_row)
        
        self._use_recon_btn = QPushButton("Use Reconstructed")
        self._use_recon_btn.clicked.connect(self._on_use_reconstructed)
        self._use_recon_btn.setToolTip("Use reconstructed image from Compression tab")
        source_group.add_widget(self._use_recon_btn)
        
        self._source_label = QLabel("No image loaded")
        self._source_label.setWordWrap(True)
        self._source_label.setStyleSheet("color: #888; font-size: 11px;")
        source_group.add_widget(self._source_label)
        
        layout.addWidget(source_group)
        
        # === Target Resolution (Collapsible) ===
        target_group = CollapsibleGroupBox("Target Resolution", settings_key="enh_target")
        target_container = QWidget()
        target_layout = QFormLayout(target_container)
        target_layout.setContentsMargins(0, 0, 0, 0)
        target_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        target_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        self._target_combo = QComboBox()
        for w, h, label in TARGET_RESOLUTIONS:
            self._target_combo.addItem(label, (w, h))
        self._target_combo.setCurrentIndex(2)  # Default: 4K
        style_combobox(self._target_combo)
        target_layout.addRow("Resolution:", self._target_combo)
        
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Fit", "Fill"])
        self._mode_combo.setToolTip(
            "Fit: Scale to fit within target (may have letterboxing)\n"
            "Fill: Scale to fill target (may crop edges)"
        )
        style_combobox(self._mode_combo)
        target_layout.addRow("Mode:", self._mode_combo)
        
        target_group.add_widget(target_container)
        layout.addWidget(target_group)
        
        # === System Info (Collapsible) ===
        system_group = CollapsibleGroupBox("System", settings_key="enh_system")
        system_container = QWidget()
        system_layout = QFormLayout(system_container)
        system_layout.setContentsMargins(0, 0, 0, 0)
        system_layout.setSpacing(4)
        system_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        system_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        self._sys_device_label = QLabel("Detecting...")
        self._sys_device_label.setStyleSheet("font-size: 10px;")
        system_layout.addRow("Device:", self._sys_device_label)
        
        self._sys_torch_label = QLabel("...")
        self._sys_torch_label.setStyleSheet("font-size: 10px;")
        system_layout.addRow("PyTorch:", self._sys_torch_label)
        
        self._sys_esrgan_label = QLabel("...")
        self._sys_esrgan_label.setStyleSheet("font-size: 10px;")
        system_layout.addRow("Real-ESRGAN:", self._sys_esrgan_label)
        
        self._sys_weights_label = QLabel("...")
        self._sys_weights_label.setStyleSheet("font-size: 10px;")
        system_layout.addRow("Weights:", self._sys_weights_label)
        
        system_group.add_widget(system_container)
        layout.addWidget(system_group)
        
        # === Processing Settings (Collapsible) ===
        settings_group = CollapsibleGroupBox("Processing", settings_key="enh_processing",
                                             initially_expanded=False)
        settings_container = QWidget()
        settings_layout = QFormLayout(settings_container)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        settings_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        self._fp16_check = QCheckBox("FP16 Autocast")
        self._fp16_check.setChecked(True)
        self._fp16_check.setToolTip(
            "Use FP16 for 2x memory reduction on GPU.\n"
            "Slightly faster with minimal quality loss."
        )
        settings_layout.addRow("", self._fp16_check)
        
        self._tile_combo = QComboBox()
        for size, label in TILE_SIZE_OPTIONS:
            self._tile_combo.addItem(label, size)
        self._tile_combo.setCurrentIndex(0)  # Auto
        self._tile_combo.setToolTip(
            "Tile size for processing.\n"
            "Smaller = less VRAM, slower.\n"
            "Larger = more VRAM, faster."
        )
        style_combobox(self._tile_combo)
        settings_layout.addRow("Tile Size:", self._tile_combo)
        
        settings_group.add_widget(settings_container)
        layout.addWidget(settings_group)
        
        # === GPU Availability Warning ===
        self._warning_label = QLabel("")
        self._warning_label.setWordWrap(True)
        self._warning_label.setStyleSheet(
            "color: #ffaa00; "
            "background-color: #3a3000; "
            "border: 1px solid #554400; "
            "border-radius: 4px; "
            "padding: 8px; "
            "font-size: 11px;"
        )
        self._warning_label.setVisible(False)
        layout.addWidget(self._warning_label)
        
        # === Actions (Collapsible) ===
        actions_group = CollapsibleGroupBox("Actions", settings_key="enh_actions")
        
        self._enhance_btn = QPushButton("Enhance")
        self._enhance_btn.clicked.connect(self._on_enhance)
        self._enhance_btn.setEnabled(False)
        self._enhance_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #22c55e, stop:1 #16a34a);
                border: 1px solid #34d669;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                padding: 12px 18px;
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #34d669, stop:1 #22c55e);
                border-color: #4ae080;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #128a3d, stop:1 #16a34a);
                padding-top: 13px;
                padding-bottom: 11px;
            }
            QPushButton:disabled {
                background: #2a2a2a;
                color: #555;
                border-color: #383838;
            }
        """)
        actions_group.add_widget(self._enhance_btn)
        
        self._export_btn = QPushButton("Export Result")
        self._export_btn.clicked.connect(self._on_export)
        self._export_btn.setEnabled(False)
        actions_group.add_widget(self._export_btn)
        
        layout.addWidget(actions_group)
        
        # === Status ===
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888;")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        
        layout.addStretch()
        return panel
    
    def _create_viewer_area(self) -> QWidget:
        """Create center area with before/after image viewers."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Input image viewer with controls
        self._input_viewer = ImageViewerWithControls(label="Input")
        self._input_viewer.imageDropped.connect(self._on_image_dropped)
        self._input_viewer.clearClicked.connect(self.clear_enhancement_session)
        
        # Output image viewer with A/B toggle
        output_container = QWidget()
        output_layout = QVBoxLayout(output_container)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(2)
        
        self._output_viewer = ImageViewerWithControls(label="Output")
        self._output_viewer.clearClicked.connect(self.clear_enhancement_session)
        self._output_viewer.imageDropped.connect(self._on_image_dropped)
        
        # A/B toggle button
        ab_container = QWidget()
        ab_layout = QHBoxLayout(ab_container)
        ab_layout.setContentsMargins(0, 0, 0, 0)
        ab_layout.addStretch()
        
        self._ab_toggle = QPushButton("A/B")
        self._ab_toggle.setCheckable(True)
        self._ab_toggle.setMaximumWidth(50)
        self._ab_toggle.setToolTip("Quick compare: Toggle to show input in output viewer")
        self._ab_toggle.toggled.connect(self._on_ab_toggle)
        self._ab_toggle.setStyleSheet("""
            QPushButton:checked {
                background-color: #4a9eff;
                border-color: #3a8eef;
            }
        """)
        ab_layout.addWidget(self._ab_toggle)
        
        output_layout.addWidget(ab_container)
        output_layout.addWidget(self._output_viewer)
        
        # Link viewers for synchronized zoom/pan
        self._input_viewer.viewChanged.connect(
            lambda: self._output_viewer.sync_view(self._input_viewer)
        )
        self._output_viewer.viewChanged.connect(
            lambda: self._input_viewer.sync_view(self._output_viewer)
        )
        
        layout.addWidget(self._input_viewer)
        layout.addWidget(output_container)
        
        return container
    
    def _on_image_dropped(self, file_path: str):
        """Handle image dropped on viewer."""
        try:
            self._image = load_image(file_path)
            self._image_path = Path(file_path)
            self._source_mode = "loaded"
            
            h, w = self._image.shape[:2]
            self._source_label.setText(f"Loaded: {self._image_path.name}\n{w} × {h}")
            
            self._input_viewer.set_image(self._image)
            self._output_viewer.clear_image()
            
            self._enhance_btn.setEnabled(True)
            self._export_btn.setEnabled(False)
            self._clear_btn.setEnabled(True)
            self._result = None
            
            self._clear_metrics()
            self._status_label.setText("Image loaded. Click 'Enhance' to process.")
            
            ToastNotification.show_toast(self, f"Loaded: {self._image_path.name}", "success")
            
        except Exception as e:
            ToastNotification.show_toast(self, f"Failed to load: {e}", "error")
    
    def _on_ab_toggle(self, checked: bool):
        """Handle A/B compare toggle."""
        if self._result is None or self._image is None:
            self._ab_toggle.setChecked(False)
            return
        
        if checked:
            self._output_viewer.swap_pixmap(self._image)
            self._output_viewer.set_label("Input (A/B)")
        else:
            self._output_viewer.swap_pixmap(self._result.output_image)
            self._output_viewer.set_label("Output")
    
    def _create_metrics_panel(self) -> QWidget:
        """Create right metrics panel."""
        panel = QWidget()
        panel.setMinimumWidth(200)
        panel.setMaximumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        
        # Dimensions group
        dims_group = QGroupBox("Dimensions")
        dims_layout = QFormLayout(dims_group)
        
        self._input_dims_label = QLabel("—")
        self._target_dims_label = QLabel("—")
        self._output_dims_label = QLabel("—")
        
        dims_layout.addRow("Input:", self._input_dims_label)
        dims_layout.addRow("Target:", self._target_dims_label)
        dims_layout.addRow("Output:", self._output_dims_label)
        
        layout.addWidget(dims_group)
        
        # Processing group
        proc_group = QGroupBox("Processing")
        proc_layout = QFormLayout(proc_group)
        
        self._mode_display_label = QLabel("—")
        self._device_display_label = QLabel("—")
        self._method_label = QLabel("—")
        self._runtime_label = QLabel("—")
        
        proc_layout.addRow("Mode:", self._mode_display_label)
        proc_layout.addRow("Device:", self._device_display_label)
        proc_layout.addRow("Method:", self._method_label)
        proc_layout.addRow("Runtime:", self._runtime_label)
        
        layout.addWidget(proc_group)
        
        # Style for monospace labels
        for label in [
            self._input_dims_label, self._target_dims_label, 
            self._output_dims_label, self._runtime_label
        ]:
            label.setStyleSheet("font-family: 'Consolas', 'Monaco', monospace;")
        
        layout.addStretch()
        return panel
    
    def _update_device_info(self):
        """Update device display and warning labels."""
        from enhancement.device_detector import TORCH_AVAILABLE
        
        device, gpu_name, reason = detect_device()
        
        # Update System group labels
        if device == "cuda" and gpu_name:
            self._sys_device_label.setText(f"<span style='color:#51cf66;'>{gpu_name}</span>")
        else:
            self._sys_device_label.setText(f"<span style='color:#ffaa00;'>CPU</span>")
        
        if TORCH_AVAILABLE:
            import torch
            cuda_str = "CUDA" if torch.cuda.is_available() else "CPU only"
            self._sys_torch_label.setText(f"<span style='color:#51cf66;'>v{torch.__version__}</span> ({cuda_str})")
        else:
            self._sys_torch_label.setText("<span style='color:#ff6b6b;'>Not installed</span>")
        
        if REALESRGAN_AVAILABLE:
            self._sys_esrgan_label.setText("<span style='color:#51cf66;'>Installed</span>")
        else:
            self._sys_esrgan_label.setText("<span style='color:#ff6b6b;'>Not installed</span>")
        
        if MODEL_PATH.exists():
            size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
            self._sys_weights_label.setText(f"<span style='color:#51cf66;'>Cached</span> ({size_mb:.0f}MB)")
        else:
            self._sys_weights_label.setText("<span style='color:#888;'>Not downloaded</span>")
        
        is_gpu_available, message = get_availability_message()
        
        if not is_gpu_available:
            self._warning_label.setText(message)
            self._warning_label.setVisible(True)
            self._fp16_check.setEnabled(False)
            self._fp16_check.setChecked(False)
            self._tile_combo.setEnabled(False)
        else:
            self._warning_label.setVisible(False)
            self._fp16_check.setEnabled(True)
            self._tile_combo.setEnabled(True)
    
    def _on_load_image(self):
        """Handle load image button click."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tiff);;All Files (*)"
        )
        
        if file_path:
            try:
                self._image = load_image(file_path)
                self._image_path = Path(file_path)
                self._source_mode = "loaded"
                
                h, w = self._image.shape[:2]
                self._source_label.setText(f"Loaded: {self._image_path.name}\n{w} × {h}")
                
                self._input_viewer.set_image(self._image)
                self._output_viewer.clear_image()
                
                self._enhance_btn.setEnabled(True)
                self._export_btn.setEnabled(False)
                self._clear_btn.setEnabled(True)
                self._result = None
                
                self._clear_metrics()
                self._status_label.setText("Image loaded. Click 'Enhance' to process.")
                
                ToastNotification.show_toast(self, f"Loaded: {self._image_path.name}", "success")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")
                ToastNotification.show_toast(self, f"Load failed: {e}", "error")
    
    def _on_use_reconstructed(self):
        """Use reconstructed image from compression tab."""
        if self._compression_tab is None:
            QMessageBox.warning(
                self, "Not Available",
                "Compression tab reference not available."
            )
            return
        
        if not hasattr(self._compression_tab, '_result') or self._compression_tab._result is None:
            QMessageBox.warning(
                self, "No Image",
                "No reconstructed image available.\n"
                "Run compression first in the Compression Lab tab."
            )
            return
        
        self._image = self._compression_tab._result.reconstructed_image.copy()
        self._image_path = None
        self._source_mode = "reconstructed"
        
        h, w = self._image.shape[:2]
        self._source_label.setText(f"From Compression Tab\n{w} × {h}")
        
        self._input_viewer.set_image(self._image)
        self._output_viewer.clear_image()
        
        self._enhance_btn.setEnabled(True)
        self._export_btn.setEnabled(False)
        self._clear_btn.setEnabled(True)
        self._result = None
        
        self._clear_metrics()
        self._status_label.setText("Using reconstructed image. Click 'Enhance' to process.")
        
        ToastNotification.show_toast(self, "Using reconstructed image", "info")
    
    def _on_enhance(self):
        """Run enhancement."""
        if self._image is None:
            return
        
        self._set_running(True)
        
        target_w, target_h = self._target_combo.currentData()
        target_preset = self._target_combo.currentText()
        mode = self._mode_combo.currentText().lower()
        fp16 = self._fp16_check.isChecked()
        tile_size = self._tile_combo.currentData()
        
        if tile_size == 0:
            tile_size = 512
        
        self._status_label.setText("Starting enhancement...")
        
        self._thread = QThread()
        self._worker = EnhancementWorker(
            image=self._image,
            target_width=target_w,
            target_height=target_h,
            mode=mode,
            fp16=fp16,
            tile_size=tile_size,
            target_preset=target_preset
        )
        self._worker.moveToThread(self._thread)
        
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_enhancement_finished)
        self._worker.error.connect(self._on_enhancement_error)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        
        self._thread.start()
    
    def _on_enhancement_finished(self, result: EnhancementResult):
        """Handle enhancement completion."""
        self._result = result
        
        self._output_viewer.set_image(result.output_image)
        
        # Update context badge with enhancement method
        self._output_viewer.set_context_info(result.method)
        
        self._update_metrics(result)
        
        self._set_running(False)
        self._export_btn.setEnabled(True)
        
        self._status_label.setText(f"Enhancement complete in {result.runtime_seconds:.2f}s")
        
        ToastNotification.show_toast(
            self, 
            f"Enhancement complete ({result.runtime_seconds:.2f}s)", 
            "success"
        )
    
    def _on_enhancement_error(self, error_msg: str):
        """Handle enhancement error."""
        self._set_running(False)
        QMessageBox.critical(self, "Enhancement Error", error_msg)
        self._status_label.setText(f"Error: {error_msg}")
        ToastNotification.show_toast(self, f"Error: {error_msg}", "error")
    
    def _on_progress(self, message: str):
        """Handle progress updates."""
        self._status_label.setText(message)
    
    def _update_metrics(self, result: EnhancementResult):
        """Update metrics panel with result data."""
        self._input_dims_label.setText(f"{result.input_width} × {result.input_height}")
        self._target_dims_label.setText(f"{result.target_width} × {result.target_height}")
        self._output_dims_label.setText(f"{result.output_width} × {result.output_height}")
        
        self._mode_display_label.setText(result.mode.capitalize())
        
        if result.gpu_name:
            self._device_display_label.setText(
                f"<span style='color: #51cf66;'>{result.device}</span> ({result.gpu_name})"
            )
        else:
            self._device_display_label.setText(
                f"<span style='color: #ffaa00;'>{result.device}</span>"
            )
        
        if result.used_gpu:
            self._method_label.setText(f"<span style='color: #51cf66;'>{result.method}</span>")
        elif result.sr_skipped:
            self._method_label.setText(f"<span style='color: #888;'>{result.method}</span>")
        else:
            self._method_label.setText(f"<span style='color: #ffaa00;'>{result.method}</span>")
        
        self._runtime_label.setText(f"{result.runtime_seconds:.2f}s")
    
    def _clear_metrics(self):
        """Clear all metrics."""
        for label in [
            self._input_dims_label, self._target_dims_label,
            self._output_dims_label, self._mode_display_label,
            self._device_display_label, self._method_label,
            self._runtime_label
        ]:
            label.setText("—")
    
    def _on_export(self):
        """Export enhanced image."""
        if self._result is None:
            return
        
        if self._image_path:
            default_name = f"{self._image_path.stem}_enhanced.png"
        else:
            default_name = "enhanced.png"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Enhanced Image",
            default_name,
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                save_image(self._result.output_image, file_path)
                self._status_label.setText(f"Saved: {Path(file_path).name}")
                ToastNotification.show_toast(self, f"Exported: {Path(file_path).name}", "success")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save:\n{e}")
                ToastNotification.show_toast(self, f"Export failed: {e}", "error")
    
    def _set_running(self, running: bool):
        """Enable/disable UI during processing."""
        self._enhance_btn.setEnabled(not running and self._image is not None)
        self._load_btn.setEnabled(not running)
        self._use_recon_btn.setEnabled(not running)
        self._export_btn.setEnabled(not running and self._result is not None)
        self._target_combo.setEnabled(not running)
        self._mode_combo.setEnabled(not running)
        
        is_gpu_available, _ = get_availability_message()
        self._fp16_check.setEnabled(not running and is_gpu_available)
        self._tile_combo.setEnabled(not running and is_gpu_available)
    
    def _cleanup_thread(self):
        """Clean up thread after completion."""
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
    
    def clear_enhancement_session(self):
        """Clear all images, results, and reset session state."""
        # Check if worker is running
        if self._thread is not None and self._thread.isRunning():
            ToastNotification.show_toast(self, "Wait for processing to complete", "warning")
            return
        
        # Clear images
        self._image = None
        self._image_path = None
        self._input_viewer.clear_image()
        self._output_viewer.clear_image()
        
        # Clear results
        self._result = None
        self._source_mode = "loaded"
        
        # Reset metrics
        self._clear_metrics()
        
        # Disable action buttons
        self._enhance_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)
        
        # Reset A/B toggle
        self._ab_toggle.setChecked(False)
        
        # Reset source label
        self._source_label.setText("No image loaded")
        
        # Update status
        self._status_label.setText("")
        
        ToastNotification.show_toast(self, "Cleared enhancement", "info")

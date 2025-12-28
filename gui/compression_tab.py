"""Compression Lab tab - main interface for JPEG-like compression."""

import cv2
import numpy as np
from pathlib import Path
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QFormLayout,
    QPushButton, QLabel, QSlider, QComboBox, QCheckBox,
    QGroupBox, QSplitter, QTabWidget, QFileDialog, QMessageBox,
    QProgressBar, QStackedWidget
)
from PySide6.QtCore import Qt, QThread

from models.compression_params import CompressionParams
from models.compression_result import CompressionResult
from models.intermediate_data import IntermediateData
from utils.image_io import load_image, save_image
from gui.widgets.image_viewer import ImageViewerWithControls
from gui.widgets.mpl_canvas import MplCanvas
from gui.widgets.metrics_panel import MetricsPanel
from gui.widgets.quant_matrix_widget import QuantMatrixWidget
from gui.widgets.collapsible_group import CollapsibleGroupBox
from gui.widgets.toast import ToastNotification
from gui.widgets.plot_placeholder import PlotPlaceholder
from gui.widgets.styled_combobox import style_combobox
from gui.worker import CompressionWorker, BatchSweepWorker
from gui.dialogs.aliasing_demo_dialog import AliasingDemoDialog
from gui.dialogs.report_exporter import ReportExportWorker


# Preview resolution options (width, height, label)
PREVIEW_RESOLUTIONS = [
    (960, 540, "960×540 (Fast)"),
    (1280, 720, "1280×720 (Balanced)"),
    (1920, 1080, "1920×1080 (High)"),
]


class CompressionTab(QWidget):
    """
    Main compression lab interface.
    
    Layout:
    - Left: Control panel with collapsible sections
    - Center: Original / Reconstructed image viewers
    - Right: Metrics panel
    - Bottom: Plot area with tabs
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State
        self._image = None
        self._image_path = None
        self._result = None
        self._intermediate = None
        self._selected_block = (0, 0)
        self._thread = None
        self._worker = None
        
        # Batch sweep results for rate-distortion marker
        self._batch_results = None  # List of (quality, result) tuples
        
        # Preview mode state
        self._preview_image = None  # Downscaled image for preview mode
        self._is_preview_result = False  # Whether current result is from preview
        self._pending_report_path = None  # For full-res export path
        
        # Track if plots have data
        self._has_plot_data = False
        
        self._init_ui()
    
    def _init_ui(self):
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
        self._metrics_panel = MetricsPanel()
        self._metrics_panel.setMinimumWidth(200)
        self._metrics_panel.setMaximumWidth(280)
        main_splitter.addWidget(self._metrics_panel)
        
        # Splitter proportions
        main_splitter.setStretchFactor(0, 0)  # Control panel - fixed
        main_splitter.setStretchFactor(1, 1)  # Viewers - stretch
        main_splitter.setStretchFactor(2, 0)  # Metrics - fixed
        main_splitter.setSizes([250, 600, 250])
        
        main_layout.addWidget(main_splitter, stretch=1)
        
        # ========== BOTTOM: Plot Area ==========
        plot_area = self._create_plot_area()
        main_layout.addWidget(plot_area)
    
    def _create_control_panel(self) -> QWidget:
        """Create left control panel with collapsible sections."""
        panel = QWidget()
        panel.setMinimumWidth(200)
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(4)
        
        # === Image Loading (Collapsible) ===
        image_group = CollapsibleGroupBox("Image", settings_key="comp_image")
        
        # Load/Clear buttons in horizontal layout
        load_clear_container = QWidget()
        load_clear_layout = QHBoxLayout(load_clear_container)
        load_clear_layout.setContentsMargins(0, 0, 0, 0)
        load_clear_layout.setSpacing(4)
        
        self._load_btn = QPushButton("Load Image")
        self._load_btn.clicked.connect(self._on_load_image)
        load_clear_layout.addWidget(self._load_btn)
        
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setToolTip("Clear image and reset session")
        self._clear_btn.clicked.connect(self.clear_compression_session)
        self._clear_btn.setEnabled(False)
        self._clear_btn.setMaximumWidth(60)
        self._clear_btn.setStyleSheet("""
            QPushButton { background-color: #3a2a2a; border-color: #554444; }
            QPushButton:hover { background-color: #4a3a3a; }
            QPushButton:disabled { background-color: #2a2a2a; color: #555; }
        """)
        load_clear_layout.addWidget(self._clear_btn)
        
        image_group.add_widget(load_clear_container)
        
        self._image_info_label = QLabel("No image loaded")
        self._image_info_label.setWordWrap(True)
        self._image_info_label.setStyleSheet("color: #888; font-size: 11px;")
        image_group.add_widget(self._image_info_label)
        
        layout.addWidget(image_group)
        
        # === Parameters (Collapsible) ===
        params_group = CollapsibleGroupBox("Parameters", settings_key="comp_params")
        params_container = QWidget()
        params_layout = QFormLayout(params_container)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        params_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        # Block size
        self._block_size_combo = QComboBox()
        self._block_size_combo.addItems(["8", "16"])
        self._block_size_combo.setCurrentText("8")
        style_combobox(self._block_size_combo)
        params_layout.addRow("Block Size:", self._block_size_combo)
        
        # Quality slider
        quality_container = QWidget()
        quality_layout = QHBoxLayout(quality_container)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        
        self._quality_slider = QSlider(Qt.Orientation.Horizontal)
        self._quality_slider.setRange(1, 100)
        self._quality_slider.setValue(50)
        self._quality_slider.valueChanged.connect(self._on_quality_changed)
        
        self._quality_label = QLabel("50")
        self._quality_label.setMinimumWidth(30)
        
        quality_layout.addWidget(self._quality_slider)
        quality_layout.addWidget(self._quality_label)
        params_layout.addRow("Quality:", quality_container)
        
        # Subsampling
        self._subsampling_combo = QComboBox()
        self._subsampling_combo.addItems(["4:4:4", "4:2:2", "4:2:0"])
        self._subsampling_combo.setCurrentText("4:2:0")
        style_combobox(self._subsampling_combo)
        params_layout.addRow("Subsampling:", self._subsampling_combo)
        
        # Prefilter
        self._prefilter_check = QCheckBox("Anti-alias prefilter")
        self._prefilter_check.setToolTip("Apply Gaussian blur before chroma downsampling")
        params_layout.addRow("", self._prefilter_check)
        
        params_group.add_widget(params_container)
        layout.addWidget(params_group)
        
        # === Preview Mode (Collapsible) ===
        preview_group = CollapsibleGroupBox("Preview Mode", settings_key="comp_preview", 
                                            initially_expanded=False)
        preview_container = QWidget()
        preview_layout = QFormLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        preview_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        self._preview_check = QCheckBox("Enable Preview Mode")
        self._preview_check.setToolTip(
            "Process a downscaled version for faster preview.\n"
            "Recommended for 4K images. Exports always use full resolution."
        )
        self._preview_check.stateChanged.connect(self._on_preview_mode_changed)
        preview_layout.addRow("", self._preview_check)
        
        self._preview_res_combo = QComboBox()
        for w, h, label in PREVIEW_RESOLUTIONS:
            self._preview_res_combo.addItem(label, (w, h))
        self._preview_res_combo.setCurrentIndex(1)  # Default: 1280x720
        self._preview_res_combo.setEnabled(False)
        self._preview_res_combo.setToolTip("Target resolution for preview processing")
        style_combobox(self._preview_res_combo)
        preview_layout.addRow("Resolution:", self._preview_res_combo)
        
        preview_group.add_widget(preview_container)
        layout.addWidget(preview_group)
        
        # === Quantization Matrix Preview (Collapsible) ===
        quant_group = CollapsibleGroupBox("Quant Matrix", settings_key="comp_quant",
                                          initially_expanded=False)
        self._quant_matrix_widget = QuantMatrixWidget()
        quant_group.add_widget(self._quant_matrix_widget)
        layout.addWidget(quant_group)
        
        # === Actions (Collapsible) ===
        actions_group = CollapsibleGroupBox("Actions", settings_key="comp_actions")
        
        self._run_btn = QPushButton("▶ Run Compression")
        self._run_btn.clicked.connect(self._on_run)
        self._run_btn.setEnabled(False)
        self._run_btn.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3b82f6, stop:1 #2563eb);
                border: 1px solid #4a90f7;
                border-radius: 6px;
                font-weight: bold;
                font-size: 12px;
                padding: 12px 18px;
                color: white;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a90f7, stop:1 #3b82f6);
                border-color: #5a9eff;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2050d0, stop:1 #2563eb);
                padding-top: 13px;
                padding-bottom: 11px;
            }
            QPushButton:disabled {
                background: #2a2a2a;
                color: #555;
                border-color: #383838;
            }
        """)
        actions_group.add_widget(self._run_btn)
        
        self._batch_btn = QPushButton("Batch Sweep (Q=10..90)")
        self._batch_btn.clicked.connect(self._on_batch_sweep)
        self._batch_btn.setEnabled(False)
        actions_group.add_widget(self._batch_btn)
        
        self._export_btn = QPushButton("Export Reconstructed")
        self._export_btn.clicked.connect(self._on_export)
        self._export_btn.setEnabled(False)
        actions_group.add_widget(self._export_btn)
        
        self._aliasing_btn = QPushButton("Aliasing Demo...")
        self._aliasing_btn.clicked.connect(self._on_aliasing_demo)
        self._aliasing_btn.setToolTip("Demonstrate chroma aliasing with/without prefilter")
        actions_group.add_widget(self._aliasing_btn)
        
        self._export_report_btn = QPushButton("Export Report (PDF)")
        self._export_report_btn.clicked.connect(self._on_export_report)
        self._export_report_btn.setEnabled(False)
        self._export_report_btn.setToolTip("Export full PDF report with metrics and plots")
        actions_group.add_widget(self._export_report_btn)
        
        layout.addWidget(actions_group)
        
        # === Progress ===
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)
        
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #888;")
        layout.addWidget(self._status_label)
        
        layout.addStretch()
        return panel
    
    def _create_viewer_area(self) -> QWidget:
        """Create center area with two linked image viewers."""
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        
        # Original image viewer with controls
        self._original_viewer = ImageViewerWithControls(label="Original")
        self._original_viewer.imageDropped.connect(self._on_image_dropped)
        self._original_viewer.clearClicked.connect(self.clear_compression_session)
        
        # Reconstructed image viewer with A/B toggle
        recon_container = QWidget()
        recon_layout = QVBoxLayout(recon_container)
        recon_layout.setContentsMargins(0, 0, 0, 0)
        recon_layout.setSpacing(2)
        
        self._recon_viewer = ImageViewerWithControls(label="Reconstructed")
        self._recon_viewer.viewer().blockClicked.connect(self._on_block_clicked)
        self._recon_viewer.clearClicked.connect(self.clear_compression_session)
        self._recon_viewer.imageDropped.connect(self._on_image_dropped)
        
        # A/B toggle button in header area
        ab_container = QWidget()
        ab_layout = QHBoxLayout(ab_container)
        ab_layout.setContentsMargins(0, 0, 0, 0)
        ab_layout.addStretch()
        
        self._ab_toggle = QPushButton("A/B")
        self._ab_toggle.setCheckable(True)
        self._ab_toggle.setMaximumWidth(50)
        self._ab_toggle.setToolTip("Quick compare: Toggle to show original in output viewer")
        self._ab_toggle.toggled.connect(self._on_ab_toggle)
        self._ab_toggle.setStyleSheet("""
            QPushButton:checked {
                background-color: #4a9eff;
                border-color: #3a8eef;
            }
        """)
        ab_layout.addWidget(self._ab_toggle)
        
        recon_layout.addWidget(ab_container)
        recon_layout.addWidget(self._recon_viewer)
        
        # Each viewer zooms independently (no sync)
        
        layout.addWidget(self._original_viewer)
        layout.addWidget(recon_container)
        
        return container
    
    def _on_image_dropped(self, file_path: str):
        """Handle image dropped on viewer."""
        try:
            self._image = load_image(file_path)
            self._image_path = Path(file_path)
            h, w = self._image.shape[:2]
            
            self._image_info_label.setText(f"{self._image_path.name}\n{w} × {h}")
            self._original_viewer.set_image(self._image)
            
            self._update_preview_image()
            self._run_btn.setEnabled(True)
            self._batch_btn.setEnabled(True)
            self._clear_btn.setEnabled(True)
            
            self._recon_viewer.clear_image()
            self._metrics_panel.clear_metrics()
            self._result = None
            self._intermediate = None
            self._export_btn.setEnabled(False)
            
            self._status_label.setText("Image loaded. Click 'Run Compression'.")
            self._update_statusbar(f"Loaded: {self._image_path.name} ({w}x{h})")
            
            self._batch_results = None
            self._is_preview_result = False
            self._has_plot_data = False
            self._show_plot_placeholder()
            
            # Toast notification
            ToastNotification.show_toast(self, f"Loaded: {self._image_path.name}", "success")
            
        except Exception as e:
            ToastNotification.show_toast(self, f"Failed to load: {e}", "error")
    
    def _on_ab_toggle(self, checked: bool):
        """Handle A/B compare toggle."""
        if self._result is None or self._image is None:
            self._ab_toggle.setChecked(False)
            return
        
        display_image, _ = self._get_processing_image()
        
        if checked:
            self._recon_viewer.swap_pixmap(display_image)
            self._recon_viewer.set_label("Original (A/B)")
        else:
            self._recon_viewer.swap_pixmap(self._result.reconstructed_image)
            self._recon_viewer.set_label("Reconstructed")
    
    def _create_plot_area(self) -> QWidget:
        """Create bottom plot area with tabs and placeholder."""
        container = QGroupBox("Analysis Plots")
        container.setMinimumHeight(220)
        container.setMaximumHeight(320)
        layout = QVBoxLayout(container)
        
        # Stacked widget for placeholder vs plots
        self._plot_stack = QStackedWidget()
        
        # Placeholder (index 0)
        self._plot_placeholder = PlotPlaceholder(
            "Run compression to generate analysis plots",
            "📊"
        )
        self._plot_stack.addWidget(self._plot_placeholder)
        
        # Plot tabs (index 1)
        self._plot_tabs = QTabWidget()
        
        self._dct_canvas = MplCanvas(width=4, height=3)
        self._plot_tabs.addTab(self._dct_canvas, "DCT Coefficients")
        
        self._hist_canvas = MplCanvas(width=4, height=3)
        self._plot_tabs.addTab(self._hist_canvas, "Coefficient Histogram")
        
        self._error_canvas = MplCanvas(width=4, height=3)
        self._plot_tabs.addTab(self._error_canvas, "Error Map (Y)")
        
        self._curve_canvas = MplCanvas(width=4, height=3)
        self._plot_tabs.addTab(self._curve_canvas, "Rate-Distortion")
        
        self._plot_stack.addWidget(self._plot_tabs)
        
        layout.addWidget(self._plot_stack)
        
        # Show placeholder initially
        self._plot_stack.setCurrentIndex(0)
        
        return container
    
    def _show_plot_placeholder(self):
        """Show plot placeholder."""
        self._plot_stack.setCurrentIndex(0)
    
    def _show_plots(self):
        """Show actual plots."""
        self._plot_stack.setCurrentIndex(1)
    
    def _on_quality_changed(self, value):
        """Update quality label and quant matrix when slider changes."""
        self._quality_label.setText(str(value))
        self._quant_matrix_widget.set_quality(value)
        self._update_rd_marker()
    
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
                h, w = self._image.shape[:2]
                
                self._image_info_label.setText(f"{self._image_path.name}\n{w} × {h}")
                self._original_viewer.set_image(self._image)
                
                self._update_preview_image()
                self._run_btn.setEnabled(True)
                self._batch_btn.setEnabled(True)
                self._clear_btn.setEnabled(True)
                
                self._recon_viewer.clear_image()
                self._metrics_panel.clear_metrics()
                self._result = None
                self._intermediate = None
                self._export_btn.setEnabled(False)
                
                self._status_label.setText("Image loaded. Click 'Run Compression'.")
                self._update_statusbar(f"Loaded: {self._image_path.name} ({w}x{h})")
                
                self._batch_results = None
                self._is_preview_result = False
                self._has_plot_data = False
                self._show_plot_placeholder()
                
                # Toast notification
                ToastNotification.show_toast(self, f"Loaded: {self._image_path.name}", "success")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image:\n{e}")
                ToastNotification.show_toast(self, f"Load failed: {e}", "error")
    
    def _get_params(self) -> CompressionParams:
        """Get current compression parameters from UI."""
        return CompressionParams(
            block_size=int(self._block_size_combo.currentText()),
            quality=self._quality_slider.value(),
            subsampling_mode=self._subsampling_combo.currentText(),
            use_prefilter=self._prefilter_check.isChecked()
        )
    
    def _on_preview_mode_changed(self, state):
        """Handle preview mode checkbox state change."""
        enabled = state == Qt.CheckState.Checked.value
        self._preview_res_combo.setEnabled(enabled)
        self._update_preview_image()
        
        self._selected_block = (0, 0)
        self._original_viewer.set_selected_block(0, 0)
        self._recon_viewer.set_selected_block(0, 0)
        
        if self._result is not None:
            self._recon_viewer.clear_image()
            self._metrics_panel.clear_metrics()
            self._result = None
            self._intermediate = None
            self._export_btn.setEnabled(False)
            self._status_label.setText("Preview mode changed. Click 'Run Compression'.")
    
    def _update_preview_image(self):
        """Create or update the preview (downscaled) image."""
        if self._image is None:
            self._preview_image = None
            return
        
        h, w = self._image.shape[:2]
        target_w, target_h = self._preview_res_combo.currentData()
        
        if w <= target_w and h <= target_h:
            self._preview_image = self._image.copy()
            return
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        self._preview_image = cv2.resize(
            self._image, (new_w, new_h), 
            interpolation=cv2.INTER_AREA
        )
    
    def _get_processing_image(self) -> tuple:
        """Get the image to process based on preview mode."""
        if self._preview_check.isChecked() and self._preview_image is not None:
            return self._preview_image, True
        return self._image, False
    
    def _on_run(self):
        """Run compression with current parameters."""
        if self._image is None:
            return
        
        self._set_running(True)
        
        params = self._get_params()
        image_to_process, is_preview = self._get_processing_image()
        self._is_preview_result = is_preview
        
        self._original_viewer.set_block_size(params.block_size)
        self._recon_viewer.set_block_size(params.block_size)
        
        if is_preview:
            ph, pw = image_to_process.shape[:2]
            self._status_label.setText(f"Processing preview ({pw}×{ph})...")
        else:
            self._status_label.setText("Processing full resolution...")
        
        self._thread = QThread()
        self._worker = CompressionWorker(image_to_process, params, self._selected_block)
        self._worker.moveToThread(self._thread)
        
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_compression_finished)
        self._worker.error.connect(self._on_compression_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        
        self._thread.start()
    
    def _on_compression_finished(self, result: CompressionResult, intermediate: IntermediateData):
        """Handle compression completion."""
        self._result = result
        self._intermediate = intermediate
        
        self._recon_viewer.set_image(result.reconstructed_image)
        self._recon_viewer.set_selected_block(*self._selected_block)
        
        # Update context badge with compression params
        params = self._get_params()
        context_str = f"Q={params.quality} {params.subsampling_mode}"
        if params.use_prefilter:
            context_str += " +filter"
        self._recon_viewer.set_context_info(context_str)
        
        self._metrics_panel.update_metrics(result, params)
        
        if self._image is not None:
            h, w = self._image.shape[:2]
            rh, rw = result.reconstructed_image.shape[:2]
            self._metrics_panel.update_processing_info(
                is_preview=self._is_preview_result,
                input_size=(w, h),
                processed_size=(rw, rh)
            )
        
        self._update_plots()
        self._has_plot_data = True
        self._show_plots()
        
        self._export_btn.setEnabled(True)
        self._export_report_btn.setEnabled(True)
        self._set_running(False)
        total_time = result.encode_time_ms + result.decode_time_ms
        
        mode_str = "preview" if self._is_preview_result else "full"
        self._status_label.setText(f"Done ({mode_str}) in {total_time:.1f}ms")
        self._update_statusbar(f"Compression complete ({mode_str}, {total_time:.1f}ms)")
        
        # Toast notification
        ToastNotification.show_toast(
            self, 
            f"Compression complete ({total_time:.1f}ms)", 
            "success"
        )
    
    def _on_compression_error(self, error_msg: str):
        """Handle compression error."""
        self._set_running(False)
        QMessageBox.critical(self, "Compression Error", error_msg)
        ToastNotification.show_toast(self, f"Error: {error_msg}", "error")
    
    def _on_block_clicked(self, block_row: int, block_col: int):
        """Handle block selection on reconstructed image."""
        self._selected_block = (block_row, block_col)
        self._original_viewer.set_selected_block(block_row, block_col)
        
        if self._image is not None:
            self._on_run()
    
    def _update_plots(self):
        """Update all analysis plots with current intermediate data."""
        if self._intermediate is None:
            return
        
        if self._intermediate.selected_block_dct is not None:
            self._dct_canvas.plot_heatmap(
                self._intermediate.selected_block_dct,
                title=f"DCT Coefficients (Block {self._selected_block})",
                log_scale=True
            )
        
        if self._intermediate.all_quantized_coeffs is not None:
            self._hist_canvas.plot_histogram(
                self._intermediate.all_quantized_coeffs,
                title="Quantized Coefficient Distribution",
                bins=50
            )
        
        if self._intermediate.error_map_y is not None:
            self._error_canvas.plot_error_map(
                self._intermediate.error_map_y,
                title="Reconstruction Error (Y channel, 10x amplified)"
            )
    
    def _on_batch_sweep(self):
        """Run batch quality sweep."""
        if self._image is None:
            return
        
        self._set_running(True)
        self._progress_bar.setVisible(True)
        self._progress_bar.setValue(0)
        
        params = self._get_params()
        image_to_process, is_preview = self._get_processing_image()
        
        if is_preview:
            ph, pw = image_to_process.shape[:2]
            self._status_label.setText(f"Batch sweep (preview {pw}×{ph})...")
        
        self._thread = QThread()
        self._worker = BatchSweepWorker(image_to_process, params)
        self._worker.moveToThread(self._thread)
        
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_batch_finished)
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.error.connect(self._on_compression_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)
        
        self._thread.start()
    
    def _on_batch_progress(self, current: int, total: int):
        """Update progress bar during batch sweep."""
        self._progress_bar.setMaximum(total)
        self._progress_bar.setValue(current)
        self._status_label.setText(f"Processing quality {current}/{total}...")
    
    def _on_batch_finished(self, results: list):
        """Handle batch sweep completion."""
        self._progress_bar.setVisible(False)
        self._set_running(False)
        
        self._batch_results = results
        self._plot_rd_curves()
        self._has_plot_data = True
        self._show_plots()
        self._plot_tabs.setCurrentWidget(self._curve_canvas)
        
        total_time = sum(r[1].encode_time_ms + r[1].decode_time_ms for r in results)
        self._status_label.setText(f"Batch sweep complete. {len(results)} levels in {total_time:.0f}ms.")
        self._update_statusbar(f"Batch sweep complete ({len(results)} levels)")
        
        # Toast notification
        ToastNotification.show_toast(
            self, 
            f"Batch sweep complete ({len(results)} levels)", 
            "success"
        )
    
    def _plot_rd_curves(self):
        """Plot rate-distortion curves with current quality marker."""
        if self._batch_results is None:
            self._curve_canvas.fig.clear()
            ax = self._curve_canvas.fig.add_subplot(111)
            ax.set_facecolor('#1e1e1e')
            ax.text(0.5, 0.5, "Run 'Batch Sweep' first to see\nrate-distortion curves",
                    ha='center', va='center', color='#888', fontsize=12,
                    transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            self._curve_canvas.draw()
            return
        
        qualities = [r[0] for r in self._batch_results]
        bpp_values = [r[1].bpp for r in self._batch_results]
        psnr_y_values = [r[1].psnr_y for r in self._batch_results]
        
        self._curve_canvas.fig.clear()
        ax = self._curve_canvas.fig.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        
        ax.plot(bpp_values, psnr_y_values, 'o-', color='#4a9eff', 
                linewidth=2, markersize=6, label='PSNR (Y)')
        
        current_quality = self._quality_slider.value()
        closest_idx = min(range(len(qualities)), 
                          key=lambda i: abs(qualities[i] - current_quality))
        
        if current_quality in qualities:
            idx = qualities.index(current_quality)
            marker_bpp = bpp_values[idx]
            marker_psnr = psnr_y_values[idx]
        else:
            marker_bpp = bpp_values[closest_idx]
            marker_psnr = psnr_y_values[closest_idx]
        
        ax.scatter([marker_bpp], [marker_psnr], s=150, c='#ff6b6b', 
                   marker='*', zorder=5, label=f'Current Q={current_quality}')
        
        ax.annotate(f'Q={current_quality}', (marker_bpp, marker_psnr),
                    textcoords='offset points', xytext=(10, 10),
                    color='#ff6b6b', fontsize=9)
        
        ax.set_title("Rate-Distortion Curve", color='white', fontsize=10)
        ax.set_xlabel("BPP (bits per pixel)", color='white', fontsize=9)
        ax.set_ylabel("PSNR (dB)", color='white', fontsize=9)
        ax.tick_params(colors='white')
        ax.legend(facecolor='#2b2b2b', edgecolor='#555555', labelcolor='white')
        ax.grid(True, alpha=0.3, color='#555555')
        for spine in ax.spines.values():
            spine.set_color('#555555')
        
        self._curve_canvas.fig.tight_layout()
        self._curve_canvas.draw()
    
    def _update_rd_marker(self):
        """Update rate-distortion marker when quality slider changes."""
        if self._batch_results is not None:
            self._plot_rd_curves()
    
    def _on_aliasing_demo(self):
        """Open aliasing demonstration dialog."""
        dialog = AliasingDemoDialog(self, loaded_image=self._image)
        dialog.applySettings.connect(self._on_apply_aliasing_settings)
        dialog.exec()
    
    def _on_apply_aliasing_settings(self, subsampling_mode: str, use_prefilter: bool):
        """Apply settings from aliasing demo to main pipeline."""
        idx = self._subsampling_combo.findText(subsampling_mode)
        if idx >= 0:
            self._subsampling_combo.setCurrentIndex(idx)
        self._prefilter_check.setChecked(use_prefilter)
        self._status_label.setText(f"Applied: {subsampling_mode}, prefilter={'ON' if use_prefilter else 'OFF'}")
    
    def _on_export(self):
        """Export reconstructed image."""
        if self._result is None:
            return
        
        default_name = "reconstructed.png"
        if self._image_path:
            default_name = f"{self._image_path.stem}_reconstructed.png"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Reconstructed Image",
            default_name,
            "PNG (*.png);;JPEG (*.jpg);;All Files (*)"
        )
        
        if file_path:
            try:
                save_image(self._result.reconstructed_image, file_path)
                self._status_label.setText(f"Saved: {Path(file_path).name}")
                ToastNotification.show_toast(self, f"Exported: {Path(file_path).name}", "success")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to save:\n{e}")
                ToastNotification.show_toast(self, f"Export failed: {e}", "error")
    
    def _on_export_report(self):
        """Export PDF report with metrics, plots, and analysis."""
        if self._result is None:
            return
        
        if self._is_preview_result:
            reply = QMessageBox.information(
                self, "Full Resolution Export",
                "Export runs at Full Resolution.\n\n"
                "The PDF report will be generated using the full-resolution image "
                "for accurate metrics. This may take longer than preview mode.",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
            )
            if reply == QMessageBox.StandardButton.Cancel:
                return
        
        default_name = "compression_report.pdf"
        if self._image_path:
            default_name = f"{self._image_path.stem}_report.pdf"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF Report",
            default_name,
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if not file_path:
            return
        
        self._set_running(True)
        
        if self._is_preview_result:
            self._status_label.setText("Running full-res compression for export...")
            self._pending_report_path = file_path
            self._run_fullres_for_export()
            return
        
        self._status_label.setText("Generating PDF report...")
        
        self._report_thread = QThread()
        self._report_worker = ReportExportWorker(
            output_path=file_path,
            result=self._result,
            params=self._get_params(),
            intermediate=self._intermediate,
            batch_results=self._batch_results,
            enhancement_result=self._get_enhancement_result(),
            image_path=self._image_path
        )
        self._report_worker.moveToThread(self._report_thread)
        
        self._report_thread.started.connect(self._report_worker.run)
        self._report_worker.finished.connect(self._on_report_finished)
        self._report_worker.progress.connect(self._on_report_progress)
        self._report_worker.error.connect(self._on_report_error)
        self._report_worker.finished.connect(self._report_thread.quit)
        self._report_worker.error.connect(self._report_thread.quit)
        self._report_thread.finished.connect(self._cleanup_report_thread)
        
        self._report_thread.start()
    
    def _on_report_progress(self, message: str):
        """Update status during report generation."""
        self._status_label.setText(message)
    
    def _on_report_finished(self, output_path: str):
        """Handle report generation completion."""
        self._set_running(False)
        self._status_label.setText(f"Report saved: {Path(output_path).name}")
        ToastNotification.show_toast(self, f"Report saved: {Path(output_path).name}", "success")
        
        QMessageBox.information(
            self, "Report Exported",
            f"PDF report successfully exported to:\n\n{output_path}"
        )
    
    def _on_report_error(self, error_msg: str):
        """Handle report generation error."""
        self._set_running(False)
        QMessageBox.critical(self, "Report Error", f"Failed to generate report:\n{error_msg}")
        ToastNotification.show_toast(self, f"Report failed: {error_msg}", "error")
    
    def _cleanup_report_thread(self):
        """Clean up report thread after completion."""
        if hasattr(self, '_report_thread') and self._report_thread:
            self._report_thread.deleteLater()
            self._report_thread = None
        if hasattr(self, '_report_worker') and self._report_worker:
            self._report_worker.deleteLater()
            self._report_worker = None
    
    def _run_fullres_for_export(self):
        """Run full-resolution compression for PDF export."""
        params = self._get_params()
        
        self._fullres_thread = QThread()
        self._fullres_worker = CompressionWorker(self._image, params, (0, 0))
        self._fullres_worker.moveToThread(self._fullres_thread)
        
        self._fullres_thread.started.connect(self._fullres_worker.run)
        self._fullres_worker.finished.connect(self._on_fullres_finished)
        self._fullres_worker.error.connect(self._on_fullres_error)
        self._fullres_worker.finished.connect(self._fullres_thread.quit)
        self._fullres_worker.error.connect(self._fullres_thread.quit)
        self._fullres_thread.finished.connect(self._cleanup_fullres_thread)
        
        self._fullres_thread.start()
    
    def _on_fullres_finished(self, result: CompressionResult, intermediate: IntermediateData):
        """Handle full-res compression completion, then generate PDF."""
        self._status_label.setText("Generating PDF report...")
        
        file_path = self._pending_report_path
        self._pending_report_path = None
        
        self._report_thread = QThread()
        self._report_worker = ReportExportWorker(
            output_path=file_path,
            result=result,
            params=self._get_params(),
            intermediate=intermediate,
            batch_results=self._batch_results,
            enhancement_result=self._get_enhancement_result(),
            image_path=self._image_path
        )
        self._report_worker.moveToThread(self._report_thread)
        
        self._report_thread.started.connect(self._report_worker.run)
        self._report_worker.finished.connect(self._on_report_finished)
        self._report_worker.progress.connect(self._on_report_progress)
        self._report_worker.error.connect(self._on_report_error)
        self._report_worker.finished.connect(self._report_thread.quit)
        self._report_worker.error.connect(self._report_thread.quit)
        self._report_thread.finished.connect(self._cleanup_report_thread)
        
        self._report_thread.start()
    
    def _on_fullres_error(self, error_msg: str):
        """Handle full-res compression error."""
        self._set_running(False)
        self._pending_report_path = None
        QMessageBox.critical(self, "Export Error", f"Full-resolution processing failed:\n{error_msg}")
    
    def _cleanup_fullres_thread(self):
        """Clean up full-res thread after completion."""
        if hasattr(self, '_fullres_thread') and self._fullres_thread:
            self._fullres_thread.deleteLater()
            self._fullres_thread = None
        if hasattr(self, '_fullres_worker') and self._fullres_worker:
            self._fullres_worker.deleteLater()
            self._fullres_worker = None
    
    def _set_running(self, running: bool):
        """Enable/disable UI during processing."""
        self._run_btn.setEnabled(not running and self._image is not None)
        self._batch_btn.setEnabled(not running and self._image is not None)
        self._load_btn.setEnabled(not running)
        self._export_btn.setEnabled(not running and self._result is not None)
        self._export_report_btn.setEnabled(not running and self._result is not None)
        
        if running:
            self._status_label.setText("Processing...")
    
    def _cleanup_thread(self):
        """Clean up thread after completion."""
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
    
    def clear_compression_session(self):
        """Clear all images, results, and reset session state."""
        # Check if worker is running
        if self._thread is not None and self._thread.isRunning():
            ToastNotification.show_toast(self, "Wait for processing to complete", "warning")
            return
        
        # Clear images
        self._image = None
        self._image_path = None
        self._preview_image = None
        self._original_viewer.clear_image()
        self._recon_viewer.clear_image()
        
        # Clear results
        self._result = None
        self._intermediate = None
        self._batch_results = None
        self._selected_block = (0, 0)
        self._is_preview_result = False
        self._has_plot_data = False
        
        # Reset metrics
        self._metrics_panel.clear_metrics()
        
        # Reset plots
        self._show_plot_placeholder()
        
        # Disable action buttons
        self._run_btn.setEnabled(False)
        self._batch_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._export_report_btn.setEnabled(False)
        self._clear_btn.setEnabled(False)
        
        # Reset A/B toggle
        self._ab_toggle.setChecked(False)
        
        # Reset info label
        self._image_info_label.setText("No image loaded")
        
        # Update status
        self._status_label.setText("")
        self._update_statusbar("Session cleared")
        
        ToastNotification.show_toast(self, "Cleared session", "info")
    
    def _update_statusbar(self, message: str):
        """Update main window status bar if available."""
        main_window = self.window()
        if hasattr(main_window, 'statusBar'):
            main_window.statusBar().showMessage(message)
    
    def _get_enhancement_result(self):
        """Get enhancement result from enhancement tab if available."""
        main_window = self.window()
        if hasattr(main_window, '_enhancement_tab'):
            enhancement_tab = main_window._enhancement_tab
            if hasattr(enhancement_tab, '_result') and enhancement_tab._result is not None:
                return enhancement_tab._result
        return None

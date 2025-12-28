"""Metrics display panel widget."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLabel
)
from PySide6.QtCore import Qt

from models.compression_result import CompressionResult


class MetricsPanel(QWidget):
    """
    Panel displaying compression metrics in a readable format.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Helper to configure form layouts
        def setup_form(form_layout):
            form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
            form_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        
        # Quality Metrics (Y channel - primary)
        quality_group = QGroupBox("Quality (Y Channel)")
        quality_layout = QFormLayout(quality_group)
        setup_form(quality_layout)
        self._psnr_y_label = QLabel("—")
        self._ssim_y_label = QLabel("—")
        quality_layout.addRow("PSNR:", self._psnr_y_label)
        quality_layout.addRow("SSIM:", self._ssim_y_label)
        layout.addWidget(quality_group)
        
        # Quality Metrics (RGB - secondary)
        rgb_group = QGroupBox("Quality (RGB)")
        rgb_layout = QFormLayout(rgb_group)
        setup_form(rgb_layout)
        self._psnr_rgb_label = QLabel("—")
        self._ssim_rgb_label = QLabel("—")
        rgb_layout.addRow("PSNR:", self._psnr_rgb_label)
        rgb_layout.addRow("SSIM:", self._ssim_rgb_label)
        layout.addWidget(rgb_group)
        
        # Compression Stats
        comp_group = QGroupBox("Compression")
        comp_layout = QFormLayout(comp_group)
        setup_form(comp_layout)
        self._bpp_label = QLabel("—")
        self._ratio_label = QLabel("—")
        self._nonzero_label = QLabel("—")
        self._bitrate_type_label = QLabel("—")
        comp_layout.addRow("BPP:", self._bpp_label)
        comp_layout.addRow("Ratio:", self._ratio_label)
        comp_layout.addRow("Non-zero:", self._nonzero_label)
        comp_layout.addRow("Type:", self._bitrate_type_label)
        layout.addWidget(comp_group)
        
        # Runtime
        runtime_group = QGroupBox("Runtime")
        runtime_layout = QFormLayout(runtime_group)
        setup_form(runtime_layout)
        self._encode_time_label = QLabel("—")
        self._decode_time_label = QLabel("—")
        runtime_layout.addRow("Encode:", self._encode_time_label)
        runtime_layout.addRow("Decode:", self._decode_time_label)
        layout.addWidget(runtime_group)
        
        # Parameters
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout(params_group)
        setup_form(params_layout)
        self._subsampling_label = QLabel("—")
        self._prefilter_label = QLabel("—")
        self._block_size_label = QLabel("—")
        self._quality_label = QLabel("—")
        params_layout.addRow("Subsampling:", self._subsampling_label)
        params_layout.addRow("Prefilter:", self._prefilter_label)
        params_layout.addRow("Block size:", self._block_size_label)
        params_layout.addRow("Quality:", self._quality_label)
        layout.addWidget(params_group)
        
        # Processing Info (for Preview Mode)
        processing_group = QGroupBox("Processing")
        processing_layout = QFormLayout(processing_group)
        setup_form(processing_layout)
        self._mode_label = QLabel("—")
        self._input_size_label = QLabel("—")
        self._processed_size_label = QLabel("—")
        processing_layout.addRow("Mode:", self._mode_label)
        processing_layout.addRow("Input:", self._input_size_label)
        processing_layout.addRow("Processed:", self._processed_size_label)
        layout.addWidget(processing_group)
        
        # Stretch at bottom
        layout.addStretch()
        
        # Style
        self.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }
            QLabel {
                font-family: 'Consolas', 'Monaco', monospace;
            }
        """)
    
    def update_metrics(self, result: CompressionResult, params=None):
        """Update displayed metrics from CompressionResult."""
        # Quality (Y)
        self._psnr_y_label.setText(f"{result.psnr_y:.2f} dB")
        self._ssim_y_label.setText(f"{result.ssim_y:.4f}")
        
        # Quality (RGB)
        self._psnr_rgb_label.setText(f"{result.psnr_rgb:.2f} dB")
        self._ssim_rgb_label.setText(f"{result.ssim_rgb:.4f}")
        
        # Compression
        self._bpp_label.setText(f"{result.bpp:.3f}")
        self._ratio_label.setText(f"{result.compression_ratio:.2f}:1")
        self._nonzero_label.setText(f"{result.nonzero_coeffs:,} / {result.total_coeffs:,}")
        self._bitrate_type_label.setText(result.bitrate_label)
        
        # Runtime
        self._encode_time_label.setText(f"{result.encode_time_ms:.2f} ms")
        self._decode_time_label.setText(f"{result.decode_time_ms:.2f} ms")
        
        # Parameters
        if params:
            self._subsampling_label.setText(params.subsampling_mode)
            self._prefilter_label.setText("Yes" if params.use_prefilter else "No")
            self._block_size_label.setText(f"{params.block_size}x{params.block_size}")
            self._quality_label.setText(str(params.quality))
    
    def update_processing_info(self, is_preview: bool, input_size: tuple, processed_size: tuple):
        """
        Update processing mode display.
        
        Args:
            is_preview: True if running in preview mode
            input_size: Original image size (w, h)
            processed_size: Size of image actually processed (w, h)
        """
        if is_preview:
            self._mode_label.setText("<span style='color: #ffaa00;'>Preview</span>")
            self._mode_label.setToolTip("Running on downscaled image for faster preview")
        else:
            self._mode_label.setText("<span style='color: #51cf66;'>Full</span>")
            self._mode_label.setToolTip("Running on full resolution image")
        
        self._input_size_label.setText(f"{input_size[0]}×{input_size[1]}")
        self._processed_size_label.setText(f"{processed_size[0]}×{processed_size[1]}")
    
    def clear_processing_info(self):
        """Clear processing info display."""
        self._mode_label.setText("—")
        self._input_size_label.setText("—")
        self._processed_size_label.setText("—")
    
    def clear_metrics(self):
        """Clear all displayed metrics."""
        for label in [
            self._psnr_y_label, self._ssim_y_label,
            self._psnr_rgb_label, self._ssim_rgb_label,
            self._bpp_label, self._ratio_label, self._nonzero_label,
            self._bitrate_type_label,
            self._encode_time_label, self._decode_time_label,
            self._subsampling_label, self._prefilter_label,
            self._block_size_label, self._quality_label
        ]:
            label.setText("—")
        self.clear_processing_info()


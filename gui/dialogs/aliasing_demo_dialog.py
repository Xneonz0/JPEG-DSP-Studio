"""Aliasing demonstration dialog with enhanced visibility and loaded image support."""

import numpy as np
import cv2
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QGridLayout, QComboBox, QRadioButton, QButtonGroup,
    QSpinBox, QFormLayout, QFrame
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QPixmap, QImage

from models.compression_params import CompressionParams
from engines.pipeline import compress_reconstruct
from gui.widgets.styled_combobox import style_combobox
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def generate_equiluminance_stripes(size: int = 256) -> np.ndarray:
    """
    Generate red/cyan 1-pixel vertical stripes with MATCHED LUMINANCE.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    red = np.array([220, 40, 60], dtype=np.uint8)
    cyan = np.array([30, 220, 210], dtype=np.uint8)
    
    for j in range(size):
        if j % 2 == 0:
            img[:, j] = red
        else:
            img[:, j] = cyan
    
    return img


def generate_chroma_checkerboard(size: int = 256) -> np.ndarray:
    """Generate 2x2 pixel checkerboard with high chroma contrast."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    red = np.array([230, 50, 60], dtype=np.uint8)
    cyan = np.array([50, 220, 220], dtype=np.uint8)
    
    for i in range(size):
        for j in range(size):
            if ((i // 2) + (j // 2)) % 2 == 0:
                img[i, j] = red
            else:
                img[i, j] = cyan
    
    return img


def generate_1px_checkerboard(size: int = 256) -> np.ndarray:
    """Generate 1x1 pixel checkerboard - maximum spatial frequency."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    red = np.array([240, 40, 50], dtype=np.uint8)
    cyan = np.array([40, 240, 230], dtype=np.uint8)
    
    for i in range(size):
        for j in range(size):
            if (i + j) % 2 == 0:
                img[i, j] = red
            else:
                img[i, j] = cyan
    
    return img


def compute_metrics(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute PSNR and SSIM for Y and RGB channels."""
    # Convert to YCbCr for Y-channel metrics
    orig_ycbcr = cv2.cvtColor(original, cv2.COLOR_RGB2YCrCb)
    recon_ycbcr = cv2.cvtColor(reconstructed, cv2.COLOR_RGB2YCrCb)
    
    orig_y = orig_ycbcr[:, :, 0]
    recon_y = recon_ycbcr[:, :, 0]
    
    return {
        'psnr_y': psnr(orig_y, recon_y, data_range=255),
        'ssim_y': ssim(orig_y, recon_y, data_range=255),
        'psnr_rgb': psnr(original, reconstructed, data_range=255),
        'ssim_rgb': ssim(original, reconstructed, data_range=255, channel_axis=2),
    }


class AliasingDemoWorker(QObject):
    """Worker for aliasing comparison."""
    
    finished = Signal(object)  # dict with all results
    progress = Signal(str)
    error = Signal(str)
    
    def __init__(self, image: np.ndarray, quality: int = 50):
        super().__init__()
        self.image = image
        self.quality = quality
    
    def run(self):
        try:
            self.progress.emit("Processing without prefilter (true decimation)...")
            recon_no_pf = self._process_with_explicit_subsample(prefilter=False)
            metrics_no_pf = compute_metrics(self.image, recon_no_pf)
            
            self.progress.emit("Processing with Gaussian prefilter...")
            recon_pf = self._process_with_explicit_subsample(prefilter=True)
            metrics_pf = compute_metrics(self.image, recon_pf)
            
            # Compute amplified difference maps
            diff_no_pf = self._compute_difference(self.image, recon_no_pf)
            diff_pf = self._compute_difference(self.image, recon_pf)
            
            self.finished.emit({
                'original': self.image,
                'recon_no_pf': recon_no_pf,
                'recon_pf': recon_pf,
                'diff_no_pf': diff_no_pf,
                'diff_pf': diff_pf,
                'metrics_no_pf': metrics_no_pf,
                'metrics_pf': metrics_pf,
            })
        except Exception as e:
            import traceback
            self.error.emit(f"{str(e)}\n{traceback.format_exc()}")
    
    def _process_with_explicit_subsample(self, prefilter: bool) -> np.ndarray:
        """Process image with explicit control over subsampling."""
        img = self.image.astype(np.float32)
        
        ycbcr = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        Y = ycbcr[:, :, 0]
        Cr = ycbcr[:, :, 1]
        Cb = ycbcr[:, :, 2]
        
        h, w = Y.shape
        
        if prefilter:
            Cb_blurred = cv2.GaussianBlur(Cb, (5, 5), 0.8)
            Cr_blurred = cv2.GaussianBlur(Cr, (5, 5), 0.8)
            Cb_sub = Cb_blurred[::2, ::2]
            Cr_sub = Cr_blurred[::2, ::2]
        else:
            Cb_sub = Cb[::2, ::2]
            Cr_sub = Cr[::2, ::2]
        
        Cb_up = cv2.resize(Cb_sub, (w, h), interpolation=cv2.INTER_LINEAR)
        Cr_up = cv2.resize(Cr_sub, (w, h), interpolation=cv2.INTER_LINEAR)
        
        ycbcr_recon = np.stack([Y, Cr_up, Cb_up], axis=-1)
        rgb_recon = cv2.cvtColor(ycbcr_recon.astype(np.float32), cv2.COLOR_YCrCb2RGB)
        rgb_recon = np.clip(rgb_recon, 0, 255).astype(np.uint8)
        
        params = CompressionParams(
            quality=self.quality,
            block_size=8,
            subsampling_mode='4:4:4',
            use_prefilter=False
        )
        result, _ = compress_reconstruct(rgb_recon, params)
        
        return result.reconstructed_image
    
    def _compute_difference(self, original: np.ndarray, reconstructed: np.ndarray) -> np.ndarray:
        """Compute amplified absolute difference (10x)."""
        diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))
        diff = np.clip(diff * 10, 0, 255).astype(np.uint8)
        return diff


class AliasingDemoDialog(QDialog):
    """
    Dialog demonstrating chroma aliasing artifacts and prefilter effect.
    
    Features:
    - Synthetic patterns (default) or current loaded image
    - ROI cropping for loaded images
    - PSNR/SSIM metrics comparison
    - Apply settings to main pipeline button
    """
    
    # Signal to apply settings to main pipeline
    applySettings = Signal(str, bool)  # (subsampling_mode, use_prefilter)
    
    def __init__(self, parent=None, loaded_image: np.ndarray = None):
        super().__init__(parent)
        self.setWindowTitle("Chroma Aliasing Demonstration")
        self.setMinimumSize(1150, 750)
        
        self._loaded_image = loaded_image
        self._thread = None
        self._worker = None
        self._last_results = None
        
        self._init_ui()
        self._apply_style()
        self._update_source_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Description
        desc = QLabel(
            "<b>Chroma Subsampling Aliasing Demonstration</b><br>"
            "Compare 4:2:0 subsampling <span style='color: #ff6b6b;'>without prefilter</span> vs "
            "<span style='color: #51cf66;'>with prefilter</span>."
        )
        desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)
        
        # === Source Selection ===
        source_group = QGroupBox("Image Source")
        source_layout = QVBoxLayout(source_group)
        
        # Radio buttons
        radio_layout = QHBoxLayout()
        self._source_group = QButtonGroup(self)
        
        self._synthetic_radio = QRadioButton("Synthetic Pattern")
        self._synthetic_radio.setChecked(True)
        self._source_group.addButton(self._synthetic_radio, 0)
        radio_layout.addWidget(self._synthetic_radio)
        
        self._loaded_radio = QRadioButton("Current Loaded Image")
        self._source_group.addButton(self._loaded_radio, 1)
        radio_layout.addWidget(self._loaded_radio)
        
        if self._loaded_image is None:
            self._loaded_radio.setEnabled(False)
            self._loaded_radio.setToolTip("No image loaded in main window")
        else:
            h, w = self._loaded_image.shape[:2]
            self._loaded_radio.setText(f"Current Loaded Image ({w}×{h})")
        
        radio_layout.addStretch()
        source_layout.addLayout(radio_layout)
        
        # Synthetic pattern controls
        self._synthetic_frame = QFrame()
        synthetic_layout = QHBoxLayout(self._synthetic_frame)
        synthetic_layout.setContentsMargins(0, 0, 0, 0)
        synthetic_layout.addWidget(QLabel("Pattern:"))
        self._pattern_combo = QComboBox()
        self._pattern_combo.addItems([
            "1-Pixel Red/Cyan Stripes (worst case)",
            "1-Pixel Checkerboard (max frequency)",
            "2×2 Checkerboard"
        ])
        style_combobox(self._pattern_combo)
        synthetic_layout.addWidget(self._pattern_combo)
        synthetic_layout.addStretch()
        source_layout.addWidget(self._synthetic_frame)
        
        # Loaded image ROI controls
        self._loaded_frame = QFrame()
        loaded_layout = QFormLayout(self._loaded_frame)
        loaded_layout.setContentsMargins(0, 0, 0, 0)
        
        roi_layout = QHBoxLayout()
        self._roi_size_spin = QSpinBox()
        self._roi_size_spin.setRange(128, 1024)
        self._roi_size_spin.setValue(512)
        self._roi_size_spin.setSingleStep(64)
        self._roi_size_spin.setToolTip("Size of ROI to crop from loaded image")
        roi_layout.addWidget(self._roi_size_spin)
        roi_layout.addWidget(QLabel("px"))
        roi_layout.addStretch()
        loaded_layout.addRow("ROI Size:", roi_layout)
        
        offset_layout = QHBoxLayout()
        self._roi_x_spin = QSpinBox()
        self._roi_x_spin.setRange(0, 10000)
        self._roi_x_spin.setValue(0)
        self._roi_x_spin.setToolTip("X offset from center (0 = center crop)")
        offset_layout.addWidget(QLabel("X:"))
        offset_layout.addWidget(self._roi_x_spin)
        self._roi_y_spin = QSpinBox()
        self._roi_y_spin.setRange(0, 10000)
        self._roi_y_spin.setValue(0)
        self._roi_y_spin.setToolTip("Y offset from center (0 = center crop)")
        offset_layout.addWidget(QLabel("Y:"))
        offset_layout.addWidget(self._roi_y_spin)
        offset_layout.addStretch()
        loaded_layout.addRow("Offset:", offset_layout)
        
        self._loaded_frame.setVisible(False)
        source_layout.addWidget(self._loaded_frame)
        
        layout.addWidget(source_group)
        
        # Connect radio buttons
        self._source_group.buttonClicked.connect(self._update_source_ui)
        
        # === Run Button ===
        run_layout = QHBoxLayout()
        run_layout.addStretch()
        self._run_btn = QPushButton("▶ Run Comparison")
        self._run_btn.clicked.connect(self._run_demo)
        self._run_btn.setStyleSheet("font-weight: bold; padding: 10px 20px;")
        run_layout.addWidget(self._run_btn)
        run_layout.addStretch()
        layout.addLayout(run_layout)
        
        # Progress
        self._progress_label = QLabel("")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setStyleSheet("color: #888;")
        layout.addWidget(self._progress_label)
        
        # === Image Comparison Area ===
        comparison_group = QGroupBox("Comparison Results")
        comparison_layout = QGridLayout(comparison_group)
        
        labels = [
            ("Original / ROI", "#e0e0e0"),
            ("No Prefilter\n(True Decimation)", "#ff6b6b"),
            ("With Prefilter\n(Gaussian + Decimate)", "#51cf66"),
            ("Diff: No Prefilter\n(10× amplified)", "#ffaa00"),
            ("Diff: Prefilter\n(10× amplified)", "#00aaff")
        ]
        
        self._image_labels = []
        
        for col, (text, color) in enumerate(labels):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet(f"font-weight: bold; color: {color};")
            comparison_layout.addWidget(lbl, 0, col)
            
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setMinimumSize(200, 200)
            img_lbl.setStyleSheet("border: 1px solid #555; background-color: #1a1a1a;")
            comparison_layout.addWidget(img_lbl, 1, col)
            self._image_labels.append(img_lbl)
        
        layout.addWidget(comparison_group)
        
        # === Metrics Display ===
        metrics_group = QGroupBox("Quality Metrics Comparison")
        metrics_layout = QGridLayout(metrics_group)
        
        # Headers
        metrics_layout.addWidget(QLabel(""), 0, 0)
        no_pf_header = QLabel("<b>Without Prefilter</b>")
        no_pf_header.setStyleSheet("color: #ff6b6b;")
        metrics_layout.addWidget(no_pf_header, 0, 1)
        pf_header = QLabel("<b>With Prefilter</b>")
        pf_header.setStyleSheet("color: #51cf66;")
        metrics_layout.addWidget(pf_header, 0, 2)
        metrics_layout.addWidget(QLabel("<b>Δ (Prefilter - No)</b>"), 0, 3)
        
        # Metric rows
        self._psnr_y_labels = []
        self._ssim_y_labels = []
        self._psnr_rgb_labels = []
        self._ssim_rgb_labels = []
        
        for row, (name, label_list) in enumerate([
            ("PSNR (Y)", self._psnr_y_labels),
            ("SSIM (Y)", self._ssim_y_labels),
            ("PSNR (RGB)", self._psnr_rgb_labels),
            ("SSIM (RGB)", self._ssim_rgb_labels),
        ], start=1):
            metrics_layout.addWidget(QLabel(name), row, 0)
            for col in range(3):
                lbl = QLabel("—")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet("font-family: monospace;")
                metrics_layout.addWidget(lbl, row, col + 1)
                label_list.append(lbl)
        
        layout.addWidget(metrics_group)
        
        # === Bottom Buttons ===
        button_layout = QHBoxLayout()
        
        self._apply_btn = QPushButton("Apply Settings to Main Pipeline")
        self._apply_btn.setToolTip(
            "Copy the recommended settings (4:2:0 with prefilter) to the main compression tab"
        )
        self._apply_btn.clicked.connect(self._on_apply_settings)
        button_layout.addWidget(self._apply_btn)
        
        button_layout.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        button_layout.addWidget(close_btn)
        
        layout.addLayout(button_layout)
    
    def _update_source_ui(self):
        """Show/hide controls based on source selection."""
        is_synthetic = self._synthetic_radio.isChecked()
        self._synthetic_frame.setVisible(is_synthetic)
        self._loaded_frame.setVisible(not is_synthetic)
    
    def _get_input_image(self) -> np.ndarray:
        """Get the input image based on source selection."""
        if self._synthetic_radio.isChecked():
            pattern_idx = self._pattern_combo.currentIndex()
            if pattern_idx == 0:
                return generate_equiluminance_stripes(256)
            elif pattern_idx == 1:
                return generate_1px_checkerboard(256)
            else:
                return generate_chroma_checkerboard(256)
        else:
            # Extract ROI from loaded image
            if self._loaded_image is None:
                raise ValueError("No image loaded")
            
            h, w = self._loaded_image.shape[:2]
            roi_size = self._roi_size_spin.value()
            offset_x = self._roi_x_spin.value()
            offset_y = self._roi_y_spin.value()
            
            # Center crop with offset
            cx = w // 2 + offset_x
            cy = h // 2 + offset_y
            
            x1 = max(0, cx - roi_size // 2)
            y1 = max(0, cy - roi_size // 2)
            x2 = min(w, x1 + roi_size)
            y2 = min(h, y1 + roi_size)
            
            # Adjust if we hit the edge
            if x2 - x1 < roi_size and x1 > 0:
                x1 = max(0, x2 - roi_size)
            if y2 - y1 < roi_size and y1 > 0:
                y1 = max(0, y2 - roi_size)
            
            roi = self._loaded_image[y1:y2, x1:x2].copy()
            return roi
    
    def _run_demo(self):
        """Run the aliasing demonstration."""
        self._set_running(True)
        self._progress_label.setStyleSheet("color: #888;")
        
        try:
            image = self._get_input_image()
        except Exception as e:
            self._progress_label.setText(f"Error: {e}")
            self._progress_label.setStyleSheet("color: #ff6b6b;")
            self._set_running(False)
            return
        
        # Create worker and thread
        self._thread = QThread()
        self._worker = AliasingDemoWorker(image, quality=50)
        self._worker.moveToThread(self._thread)
        
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_finished)
        self._worker.progress.connect(self._on_progress)
        self._worker.error.connect(self._on_error)
        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        
        self._thread.start()
    
    def _on_progress(self, msg: str):
        self._progress_label.setText(msg)
    
    def _on_finished(self, results: dict):
        """Handle completion of aliasing demo."""
        self._set_running(False)
        self._last_results = results
        self._progress_label.setText("✓ Complete!")
        self._progress_label.setStyleSheet("color: #51cf66;")
        
        # Display images
        images = [
            results['original'],
            results['recon_no_pf'],
            results['recon_pf'],
            results['diff_no_pf'],
            results['diff_pf']
        ]
        for label, img in zip(self._image_labels, images):
            self._set_image(label, img)
        
        # Display metrics
        m_no_pf = results['metrics_no_pf']
        m_pf = results['metrics_pf']
        
        # PSNR Y
        self._psnr_y_labels[0].setText(f"{m_no_pf['psnr_y']:.2f} dB")
        self._psnr_y_labels[1].setText(f"{m_pf['psnr_y']:.2f} dB")
        delta = m_pf['psnr_y'] - m_no_pf['psnr_y']
        self._psnr_y_labels[2].setText(f"{delta:+.2f} dB")
        self._psnr_y_labels[2].setStyleSheet(
            f"color: {'#51cf66' if delta >= 0 else '#ff6b6b'}; font-family: monospace;"
        )
        
        # SSIM Y
        self._ssim_y_labels[0].setText(f"{m_no_pf['ssim_y']:.4f}")
        self._ssim_y_labels[1].setText(f"{m_pf['ssim_y']:.4f}")
        delta = m_pf['ssim_y'] - m_no_pf['ssim_y']
        self._ssim_y_labels[2].setText(f"{delta:+.4f}")
        self._ssim_y_labels[2].setStyleSheet(
            f"color: {'#51cf66' if delta >= 0 else '#ff6b6b'}; font-family: monospace;"
        )
        
        # PSNR RGB
        self._psnr_rgb_labels[0].setText(f"{m_no_pf['psnr_rgb']:.2f} dB")
        self._psnr_rgb_labels[1].setText(f"{m_pf['psnr_rgb']:.2f} dB")
        delta = m_pf['psnr_rgb'] - m_no_pf['psnr_rgb']
        self._psnr_rgb_labels[2].setText(f"{delta:+.2f} dB")
        self._psnr_rgb_labels[2].setStyleSheet(
            f"color: {'#51cf66' if delta >= 0 else '#ff6b6b'}; font-family: monospace;"
        )
        
        # SSIM RGB
        self._ssim_rgb_labels[0].setText(f"{m_no_pf['ssim_rgb']:.4f}")
        self._ssim_rgb_labels[1].setText(f"{m_pf['ssim_rgb']:.4f}")
        delta = m_pf['ssim_rgb'] - m_no_pf['ssim_rgb']
        self._ssim_rgb_labels[2].setText(f"{delta:+.4f}")
        self._ssim_rgb_labels[2].setStyleSheet(
            f"color: {'#51cf66' if delta >= 0 else '#ff6b6b'}; font-family: monospace;"
        )
    
    def _on_error(self, msg: str):
        self._set_running(False)
        self._progress_label.setText(f"Error: {msg}")
        self._progress_label.setStyleSheet("color: #ff6b6b;")
    
    def _set_image(self, label: QLabel, image: np.ndarray):
        """Set QLabel to display numpy RGB image."""
        h, w = image.shape[:2]
        c = image.shape[2] if len(image.shape) > 2 else 1
        if c == 3:
            bytes_per_line = 3 * w
            qimage = QImage(image.data.tobytes(), w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:
            qimage = QImage(image.data.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
        
        pixmap = QPixmap.fromImage(qimage).scaled(
            200, 200, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation
        )
        label.setPixmap(pixmap)
    
    def _set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._pattern_combo.setEnabled(not running)
        self._synthetic_radio.setEnabled(not running)
        self._loaded_radio.setEnabled(not running and self._loaded_image is not None)
        self._roi_size_spin.setEnabled(not running)
        self._roi_x_spin.setEnabled(not running)
        self._roi_y_spin.setEnabled(not running)
    
    def _on_apply_settings(self):
        """Apply recommended settings to main pipeline."""
        # Emit signal with 4:2:0 + prefilter enabled (recommended based on demo)
        self.applySettings.emit('4:2:0', True)
        self._progress_label.setText("Settings applied: 4:2:0 with prefilter")
        self._progress_label.setStyleSheet("color: #4a9eff;")
    
    def _cleanup(self):
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
    
    def _apply_style(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #2b2b2b;
                color: #e0e0e0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #555;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:disabled {
                background-color: #2a2a2a;
                color: #666;
            }
            QComboBox, QSpinBox {
                background-color: #3a3a3a;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QRadioButton {
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
            }
        """)

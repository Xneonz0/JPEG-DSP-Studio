"""Main application window."""

import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QMainWindow, QTabWidget, QWidget, QVBoxLayout, QLabel, QMenuBar, QStatusBar,
    QFileDialog, QMessageBox, QMenu
)
from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QIcon

from gui.compression_tab import CompressionTab
from gui.enhancement_tab import EnhancementTab

# App metadata
APP_VERSION = "1.0"
APP_NAME = "JPEG-DSP Studio"
PYTHON_REQUIREMENT = "3.11+"


class MainWindow(QMainWindow):
    """
    Main application window with tab interface.
    
    Tabs:
    - Compression Lab: JPEG-like compression with DCT
    - Enhancement: GPU-accelerated upscaling (Real-ESRGAN / Lanczos)
    """
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle(f"{APP_NAME}: JPEG-Inspired Image Compression")
        self.setMinimumSize(1200, 800)
        
        # Settings for remembering last export folder
        self._settings = QSettings("DSPLab", "DSPLab")
        
        self._init_ui()
        self._init_menu()
        self._init_statusbar()
        
        # Apply dark theme and icon
        self._apply_dark_theme()
        self._set_app_icon()
    
    def _init_ui(self):
        """Initialize main UI with tabs."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget
        self._tab_widget = QTabWidget()
        layout.addWidget(self._tab_widget)
        
        # Compression Lab tab
        self._compression_tab = CompressionTab()
        self._tab_widget.addTab(self._compression_tab, "Compression Lab")
        
        # Enhancement tab (Real-ESRGAN / Lanczos upscaling)
        self._enhancement_tab = EnhancementTab(compression_tab=self._compression_tab)
        self._tab_widget.addTab(self._enhancement_tab, "Enhancement")
    
    
    def _init_menu(self):
        """Initialize menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        new_session_action = QAction("&New Session", self)
        new_session_action.setShortcut("Ctrl+N")
        new_session_action.setToolTip("Clear all images and reset both tabs")
        new_session_action.triggered.connect(self._on_new_session)
        file_menu.addAction(new_session_action)
        
        file_menu.addSeparator()
        
        load_action = QAction("&Load Image", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._compression_tab._on_load_image)
        file_menu.addAction(load_action)
        
        export_action = QAction("&Export Reconstructed", self)
        export_action.setShortcut("Ctrl+S")
        export_action.triggered.connect(self._compression_tab._on_export)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        export_all_action = QAction("Export &Everything...", self)
        export_all_action.setShortcut("Ctrl+Shift+E")
        export_all_action.triggered.connect(self._on_export_everything)
        file_menu.addAction(export_all_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Demo menu
        demo_menu = menubar.addMenu("&Demo")
        
        demo_submenu = demo_menu.addMenu("Load Demo Image")
        
        demo_images = [
            ("Photo (Natural)", "photo"),
            ("Text & Edges", "text_edges"),
            ("Gradient", "gradient"),
            ("Checkerboard", "checkerboard"),
            ("Chroma Stripes", "chroma_stripes"),
        ]
        
        for label, key in demo_images:
            action = QAction(label, self)
            action.setData(key)
            action.triggered.connect(lambda checked, k=key: self._load_demo_image(k))
            demo_submenu.addAction(action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        run_action = QAction("&Run Compression", self)
        run_action.setShortcut("F5")
        run_action.triggered.connect(self._compression_tab._on_run)
        view_menu.addAction(run_action)
        
        batch_action = QAction("&Batch Sweep", self)
        batch_action.setShortcut("F6")
        batch_action.triggered.connect(self._compression_tab._on_batch_sweep)
        view_menu.addAction(batch_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        theory_action = QAction("&DSP Theory Reference", self)
        theory_action.setShortcut("F1")
        theory_action.triggered.connect(self._show_theory)
        help_menu.addAction(theory_action)
        
        system_info_action = QAction("&System Info...", self)
        system_info_action.triggered.connect(self._show_system_info)
        help_menu.addAction(system_info_action)
        
        help_menu.addSeparator()
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _init_statusbar(self):
        """Initialize status bar."""
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready. Load an image to begin.")
    
    def _on_new_session(self):
        """Clear both tabs and start a new session."""
        self._compression_tab.clear_compression_session()
        self._enhancement_tab.clear_enhancement_session()
        self._statusbar.showMessage("New session started")
    
    def _show_theory(self):
        """Show DSP theory reference dialog."""
        from gui.dialogs.theory_dialog import TheoryDialog
        dialog = TheoryDialog(self)
        dialog.exec()
    
    def _show_system_info(self):
        """Show system information dialog."""
        from gui.dialogs.system_info_dialog import SystemInfoDialog
        dialog = SystemInfoDialog(self)
        dialog.exec()
    
    def _show_about(self):
        """Show enhanced about dialog."""
        from gui.dialogs.about_dialog import AboutDialog
        dialog = AboutDialog(self)
        dialog.exec()
    
    def _on_export_everything(self):
        """Export all available artifacts to a single folder."""
        # Check if anything to export
        comp_result = self._compression_tab._result
        if comp_result is None:
            QMessageBox.warning(
                self, "Nothing to Export",
                "Run compression first before exporting."
            )
            return
        
        # Get last export folder or default
        last_folder = self._settings.value("last_export_folder", "")
        
        # Ask for output folder
        folder = QFileDialog.getExistingDirectory(
            self, "Select Export Folder", last_folder
        )
        
        if not folder:
            return
        
        # Remember folder
        self._settings.setValue("last_export_folder", folder)
        folder_path = Path(folder)
        
        # Get base name from loaded image
        if self._compression_tab._image_path:
            base_name = self._compression_tab._image_path.stem
        else:
            base_name = "output"
        
        exported = []
        errors = []
        
        # 1. Export reconstructed image
        try:
            from utils.image_io import save_image
            recon_path = folder_path / f"{base_name}_reconstructed.png"
            save_image(comp_result.reconstructed_image, str(recon_path))
            exported.append(f"Reconstructed: {recon_path.name}")
        except Exception as e:
            errors.append(f"Reconstructed: {e}")
        
        # 2. Export enhanced image (if available)
        enh_result = self._enhancement_tab._result if hasattr(self._enhancement_tab, '_result') else None
        if enh_result is not None:
            try:
                from utils.image_io import save_image
                enh_path = folder_path / f"{base_name}_enhanced.png"
                save_image(enh_result.output_image, str(enh_path))
                exported.append(f"Enhanced: {enh_path.name}")
            except Exception as e:
                errors.append(f"Enhanced: {e}")
        
        # 3. Export PDF report
        try:
            from gui.dialogs.report_exporter import ReportExportWorker
            pdf_path = folder_path / f"{base_name}_report.pdf"
            
            # Run synchronously for simplicity
            worker = ReportExportWorker(
                output_path=str(pdf_path),
                result=comp_result,
                params=self._compression_tab._get_params(),
                intermediate=self._compression_tab._intermediate,
                batch_results=self._compression_tab._batch_results,
                enhancement_result=enh_result,
                image_path=self._compression_tab._image_path
            )
            worker._generate_report()
            exported.append(f"PDF Report: {pdf_path.name}")
        except Exception as e:
            errors.append(f"PDF Report: {e}")
        
        # 4. Export batch sweep CSV (if available)
        batch_results = self._compression_tab._batch_results
        if batch_results:
            try:
                csv_path = folder_path / f"{base_name}_batch_sweep.csv"
                self._export_batch_csv(batch_results, csv_path)
                exported.append(f"Batch CSV: {csv_path.name}")
            except Exception as e:
                errors.append(f"Batch CSV: {e}")
        
        # Show summary
        summary = f"Exported to: {folder}\n\n"
        if exported:
            summary += "Successfully exported:\n" + "\n".join(f"  - {e}" for e in exported)
        if errors:
            summary += "\n\nErrors:\n" + "\n".join(f"  - {e}" for e in errors)
        
        if errors:
            QMessageBox.warning(self, "Export Complete (with errors)", summary)
        else:
            QMessageBox.information(self, "Export Complete", summary)
        
        self._statusbar.showMessage(f"Exported {len(exported)} files to {folder}")
    
    def _export_batch_csv(self, batch_results: list, csv_path: Path):
        """Export batch sweep results to CSV."""
        with open(csv_path, 'w') as f:
            f.write("quality,bpp,psnr_y,ssim_y,psnr_rgb,ssim_rgb,compression_ratio\n")
            for quality, result in batch_results:
                f.write(f"{quality},{result.bpp:.4f},{result.psnr_y:.2f},"
                        f"{result.ssim_y:.4f},{result.psnr_rgb:.2f},"
                        f"{result.ssim_rgb:.4f},{result.compression_ratio:.2f}\n")
    
    def _load_demo_image(self, key: str):
        """Load a demo image by key."""
        from utils.test_images import generate_demo_image
        
        # Generate or load demo image
        demo_image = generate_demo_image(key)
        
        if demo_image is None:
            QMessageBox.warning(self, "Demo Error", f"Could not load demo image: {key}")
            return
        
        # Load into compression tab
        self._compression_tab._image = demo_image
        self._compression_tab._image_path = None
        h, w = demo_image.shape[:2]
        
        self._compression_tab._image_info_label.setText(f"Demo: {key}\n{w} × {h}")
        self._compression_tab._original_viewer.set_image(demo_image)
        self._compression_tab._update_preview_image()
        
        # Enable buttons
        self._compression_tab._run_btn.setEnabled(True)
        self._compression_tab._batch_btn.setEnabled(True)
        
        # Clear previous results
        self._compression_tab._recon_viewer.clear_image()
        self._compression_tab._metrics_panel.clear_metrics()
        self._compression_tab._result = None
        self._compression_tab._intermediate = None
        self._compression_tab._export_btn.setEnabled(False)
        self._compression_tab._batch_results = None
        self._compression_tab._is_preview_result = False
        
        # Show tip for demo parameters
        self._statusbar.showMessage(
            f"Loaded demo: {key}. Tip: Try Q=10, 4:2:0, Prefilter OFF to see compression artifacts."
        )
        
        # Switch to compression tab
        self._tab_widget.setCurrentIndex(0)
    
    def _apply_dark_theme(self):
        """Apply premium dark theme stylesheet with enhanced UX."""
        self.setStyleSheet("""
            /* === Base Styles === */
            QMainWindow {
                background-color: #1a1a1a;
            }
            QWidget {
                background-color: #242424;
                color: #e8e8e8;
                font-family: 'Segoe UI', 'SF Pro Display', 'Arial', sans-serif;
                font-size: 12px;
            }
            
            /* === Group Boxes === */
            QGroupBox {
                font-weight: 600;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                margin-top: 14px;
                padding-top: 12px;
                background-color: #2a2a2a;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                color: #999;
            }
            
            /* === Buttons - Premium Feel === */
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #404040, stop:1 #353535);
                border: 1px solid #4a4a4a;
                border-radius: 5px;
                padding: 8px 16px;
                min-height: 22px;
                color: #e8e8e8;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #4a4a4a, stop:1 #404040);
                border-color: #5a5a5a;
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #333333, stop:1 #3a3a3a);
                border-color: #444;
                padding-top: 9px;
                padding-bottom: 7px;
            }
            QPushButton:focus {
                border-color: #4a9eff;
                outline: none;
            }
            QPushButton:disabled {
                background: #2a2a2a;
                color: #555;
                border-color: #383838;
            }
            
            /* === Sliders - Polished === */
            QSlider::groove:horizontal {
                border: none;
                height: 6px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2a2a2a, stop:1 #353535);
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3a7bd5, stop:1 #4a9eff);
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5ab0ff, stop:1 #4a9eff);
                border: 2px solid #3a8eef;
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6ac0ff, stop:1 #5ab0ff);
                border-color: #4a9eff;
            }
            QSlider::handle:horizontal:pressed {
                background: #3a8eef;
            }
            
            /* === ComboBox - Clean & Responsive === */
            QComboBox {
                background-color: #363636;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 5px 10px;
                padding-right: 25px;
                min-height: 20px;
                color: #e8e8e8;
            }
            QComboBox:hover {
                border-color: #5a5a5a;
                background-color: #3a3a3a;
            }
            QComboBox:focus {
                border-color: #4a9eff;
            }
            QComboBox:on {
                background-color: #3a3a3a;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 20px;
                border: none;
                background: transparent;
            }
            QComboBox::down-arrow {
                width: 0;
                height: 0;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #888;
            }
            QComboBox::down-arrow:hover {
                border-top-color: #bbb;
            }
            QComboBox QAbstractItemView {
                background-color: #2d2d2d;
                border: 1px solid #555;
                outline: none;
                selection-background-color: #4a9eff;
                selection-color: #fff;
            }
            QComboBox QAbstractItemView::item {
                height: 26px;
                padding: 4px 8px;
                color: #ddd;
            }
            QComboBox QAbstractItemView::item:hover {
                background-color: #404040;
                color: #fff;
            }
            QComboBox QAbstractItemView::item:selected {
                background-color: #4a9eff;
                color: #fff;
            }
            
            /* === Checkboxes - Premium === */
            QCheckBox {
                spacing: 10px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #4a4a4a;
                border-radius: 4px;
                background-color: #2a2a2a;
            }
            QCheckBox::indicator:hover {
                border-color: #5a5a5a;
                background-color: #333;
            }
            QCheckBox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #5ab0ff, stop:1 #4a9eff);
                border-color: #3a8eef;
            }
            QCheckBox::indicator:checked:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #6ac0ff, stop:1 #5ab0ff);
            }
            
            /* === Tab Widget - Clean === */
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                background-color: #242424;
                top: -1px;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #383838, stop:1 #303030);
                border: 1px solid #3d3d3d;
                border-bottom: none;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                padding: 10px 20px;
                margin-right: 2px;
                color: #aaa;
            }
            QTabBar::tab:selected {
                background: #242424;
                color: #fff;
                border-bottom: 2px solid #4a9eff;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #404040, stop:1 #383838);
                color: #ddd;
            }
            QTabBar::tab:!selected {
                margin-top: 3px;
            }
            
            /* === Progress Bar - Animated Look === */
            QProgressBar {
                border: none;
                border-radius: 4px;
                text-align: center;
                background-color: #2a2a2a;
                height: 8px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #3a7bd5, stop:0.5 #4a9eff, stop:1 #3a7bd5);
                border-radius: 4px;
            }
            
            /* === Menu Bar - Refined === */
            QMenuBar {
                background-color: #1e1e1e;
                border-bottom: 1px solid #333;
                padding: 2px 0;
            }
            QMenuBar::item {
                padding: 6px 12px;
                border-radius: 4px;
                margin: 2px;
            }
            QMenuBar::item:selected {
                background-color: #3a3a3a;
            }
            QMenu {
                background-color: #2a2a2a;
                border: 1px solid #3d3d3d;
                border-radius: 6px;
                padding: 6px;
            }
            QMenu::item {
                padding: 8px 32px 8px 24px;
                border-radius: 4px;
                margin: 2px;
            }
            QMenu::item:selected {
                background-color: #4a9eff;
            }
            QMenu::separator {
                height: 1px;
                background: #3d3d3d;
                margin: 6px 12px;
            }
            
            /* === Status Bar === */
            QStatusBar {
                background-color: #1e1e1e;
                border-top: 1px solid #333;
                color: #888;
            }
            
            /* === Scrollbars - Minimal === */
            QScrollBar:vertical {
                background-color: transparent;
                width: 10px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background-color: #4a4a4a;
                min-height: 30px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #5a5a5a;
            }
            QScrollBar::handle:vertical:pressed {
                background-color: #4a9eff;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: transparent;
                height: 0;
            }
            QScrollBar:horizontal {
                background-color: transparent;
                height: 10px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background-color: #4a4a4a;
                min-width: 30px;
                border-radius: 5px;
                margin: 2px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #5a5a5a;
            }
            QScrollBar::handle:horizontal:pressed {
                background-color: #4a9eff;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: transparent;
                width: 0;
            }
            
            /* === Splitter Handles === */
            QSplitter::handle {
                background-color: #333;
            }
            QSplitter::handle:hover {
                background-color: #4a9eff;
            }
            QSplitter::handle:horizontal {
                width: 3px;
            }
            QSplitter::handle:vertical {
                height: 3px;
            }
            
            /* === Tool Tips === */
            QToolTip {
                background-color: #333;
                color: #e8e8e8;
                border: 1px solid #4a4a4a;
                border-radius: 4px;
                padding: 6px 10px;
            }
            
            /* === Text Browser (Theory Dialog) === */
            QTextBrowser {
                background-color: #1e1e1e;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
    
    def _set_app_icon(self):
        """Set application icon if available."""
        # Try multiple icon locations (includes .ico for Windows)
        icon_paths = [
            Path(__file__).parent / "icon.png",
            Path(__file__).parent.parent / "icon.png",
            Path(__file__).parent / "assets" / "icon.png",
            Path(__file__).parent / "assets" / "icon.ico",
        ]
        
        for icon_path in icon_paths:
            if icon_path.exists():
                self.setWindowIcon(QIcon(str(icon_path)))
                return
        
        # No icon found - that's OK, use system default


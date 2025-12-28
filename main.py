"""
JPEG-DSP Studio
JPEG-Inspired Image Compression + Optional RTX Enhancement
"""

import sys
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)


def run_gui():
    """Launch the GUI application."""
    from pathlib import Path
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QIcon
    from gui.main_window import MainWindow
    
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    
    app = QApplication(sys.argv)
    app.setApplicationName("JPEG-DSP Studio")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("DSP Course Project")
    app.setStyle("Fusion")
    
    # Set icon
    icon_paths = [
        Path(__file__).parent / "gui" / "icon.png",
        Path(__file__).parent / "icon.png",
        Path(__file__).parent / "gui" / "assets" / "icon.png",
    ]
    for icon_path in icon_paths:
        if icon_path.exists():
            app.setWindowIcon(QIcon(str(icon_path)))
            break
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


def run_cli():
    """Run CLI mode for testing."""
    import numpy as np
    from models.compression_params import CompressionParams
    from engines.pipeline import compress_reconstruct
    from utils.test_images import generate_colored_checkerboard
    from utils.image_io import load_image, save_image
    
    args = sys.argv[2:]
    
    if not args or args[0] == '--help':
        print("Usage: python main.py --cli <image_path> [quality]")
        print("       python main.py --cli --synthetic [quality]")
        sys.exit(0)
    
    if args[0] == '--synthetic':
        print("Generating test image...")
        image = generate_colored_checkerboard(256)
        quality = int(args[1]) if len(args) > 1 else 50
    else:
        image_path = args[0]
        print(f"Loading: {image_path}")
        image = load_image(image_path)
        quality = int(args[1]) if len(args) > 1 else 50
    
    print(f"Image: {image.shape[1]}x{image.shape[0]}")
    print(f"Quality: {quality}")
    
    params = CompressionParams(
        quality=quality,
        block_size=8,
        subsampling_mode='4:2:0',
        use_prefilter=False
    )
    
    result, _ = compress_reconstruct(image, params)
    
    print("\n=== Results ===")
    print(f"PSNR (Y):  {result.psnr_y:.2f} dB")
    print(f"SSIM (Y):  {result.ssim_y:.4f}")
    print(f"BPP:       {result.bpp:.3f}")
    print(f"Ratio:     {result.compression_ratio:.2f}:1")
    print(f"Time:      {result.encode_time_ms + result.decode_time_ms:.2f} ms")
    
    save_image(result.reconstructed_image, "reconstructed.png")
    print("\nSaved: reconstructed.png")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        run_cli()
    else:
        run_gui()


if __name__ == '__main__':
    main()

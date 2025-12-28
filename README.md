# JPEG-DSP Studio

A desktop application for exploring JPEG-style image compression. Built for a Digital Signal Processing course, demonstrating DCT transforms, quantization, chroma subsampling, and their visual effects on image quality. Includes optional AI-based upscaling for enhanced output.

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)
![PySide6](https://img.shields.io/badge/GUI-PySide6-green)
![Version](https://img.shields.io/badge/Version-1.0-orange)

---

## Features

### Compression Lab
- Block-based DCT with adjustable quality factor (1–100)
- Chroma subsampling: 4:4:4, 4:2:2, 4:2:0
- Anti-aliasing prefilter toggle
- Real-time metrics: PSNR, SSIM, bits-per-pixel
- Analysis plots: DCT heatmaps, coefficient histograms, error maps
- Batch quality sweep with rate–distortion curves
- PDF report export

### Enhancement (Optional)
- Target presets: 1080p, 1440p, 4K
- Fit/Fill scaling modes
- Real-ESRGAN x4 upscaling when GPU available
- CPU Lanczos fallback

### Interface
- Drag & drop image loading
- A/B compare toggle
- Demo menu with test patterns
- System info dashboard
- Clear/Reset session

---

## DSP Concepts Demonstrated

| Concept | Where to See It |
|---------|-----------------|
| Energy compaction | DCT heatmap — most energy in top-left |
| Quantization noise | Lower Q → more zeros → more artifacts |
| Aliasing from decimation | 4:2:0 + prefilter OFF on chroma stripes |
| Rate–distortion tradeoff | Batch sweep curves |
| Perceptual metrics | PSNR vs SSIM differences |

---

## Quick Start

**Requirements**: Python 3.10+

```bash
pip install -r requirements.txt
python main.py
```

The app runs on CPU with all core features.

---

## Optional: GPU Enhancement

To enable Real-ESRGAN acceleration on NVIDIA GPUs:

```bash
# Install PyTorch with CUDA (example for CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install Real-ESRGAN
pip install realesrgan basicsr
```

For live GPU stats in System Info:
```bash
pip install pynvml
```

GPU features are optional. The app auto-detects hardware and falls back gracefully.

---

## 60-Second Demo

1. **Demo → Checkerboard** — load a synthetic pattern
2. Set **Q=10**, **4:2:0**, **Prefilter OFF**
3. Click **Run Compression** — observe blocking and color fringing
4. Toggle **Prefilter ON** — aliasing disappears
5. Click **Batch Sweep** — see rate–distortion curves
6. **File → Export PDF Report** — generate analysis report
7. Switch to **Enhancement** tab → **1440p Fit** → **Enhance**

---

## Project Structure

```
JPEG-DSP-Studio/
├── main.py              # Entry point
├── engines/             # DSP core (DCT, quantization, pipeline)
│   ├── dct_engine.py    # DCT/IDCT implementation
│   ├── quantizer.py     # Quantization with JPEG matrix
│   ├── pipeline.py      # Main compression loop
│   ├── color_space.py   # RGB/YCbCr conversion
│   └── block_processor.py
├── gui/                 # PySide6 interface
│   ├── main_window.py
│   ├── compression_tab.py
│   ├── enhancement_tab.py
│   ├── dialogs/         # Theory, About, System Info
│   └── widgets/         # Custom UI components
├── enhancement/         # Real-ESRGAN wrapper
├── models/              # Data classes
├── utils/               # Metrics, I/O, test images
└── tests/               # Unit tests
```

---

## Known Limitations

- **Block size**: Only 8×8 blocks are supported. The block size selector includes 16×16 but it's not fully implemented yet.
- **Large images**: Processing 4K images computes full SSIM which can take a few seconds. No progress cancellation exists—just wait for it to finish.
- **Enhancement queue**: Starting a new enhancement while one is running queues the request. There's no cancel button yet.

These are tracked for a future update.

---

## Troubleshooting

**CUDA not detected?**
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
If `False`, reinstall PyTorch with CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).

**First GPU run slow?**  
First enhancement downloads model weights (~64MB) to cache. Subsequent runs are faster.

**basicsr import errors?**
```bash
pip install git+https://github.com/xinntao/BasicSR.git
```

---

## Credits

- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) by Xintao Wang et al.
- JPEG standard (ITU-T T.81)

---

## License

MIT License — use freely for learning and educational purposes.

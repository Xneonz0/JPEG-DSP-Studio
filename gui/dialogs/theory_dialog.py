"""Theory dialog with DSP explanations."""

import io
import base64
from PySide6.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QTextBrowser, QPushButton, QLabel
from PySide6.QtCore import Qt


def render_latex_to_base64(latex: str, fontsize: int = 14, dpi: int = 120) -> str:
    """Render LaTeX formula to base64 PNG for HTML embedding."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        ax.axis('off')
        ax.text(0, 0, f"${latex}$", fontsize=fontsize, color='#e0e0e0', ha='left', va='bottom')
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0.05, transparent=True)
        plt.close(fig)
        
        buf.seek(0)
        return f'data:image/png;base64,{base64.b64encode(buf.read()).decode("utf-8")}'
    except Exception:
        return ""


def formula_html(latex: str, fallback: str, fontsize: int = 14) -> str:
    """Generate HTML for a formula."""
    b64_img = render_latex_to_base64(latex, fontsize)
    if b64_img:
        return f'<img src="{b64_img}" style="vertical-align: middle; margin: 8px 0;" />'
    return f'<code style="background: #2a2a2a; padding: 8px 12px; display: block; font-family: monospace;">{fallback}</code>'


# Pre-rendered formulas
FORMULA_DCT = formula_html(
    r"F(u,v) = C_u C_v \sum_{x=0}^{N-1} \sum_{y=0}^{N-1} f(x,y) \cos\left[\frac{\pi(2x+1)u}{2N}\right] \cos\left[\frac{\pi(2y+1)v}{2N}\right]",
    "F(u,v) = Cu*Cv * sum f(x,y) * cos[pi(2x+1)u/2N] * cos[pi(2y+1)v/2N]",
    fontsize=12
)

FORMULA_C = formula_html(
    r"C_k = \begin{cases} \sqrt{1/N} & k=0 \\ \sqrt{2/N} & k>0 \end{cases}",
    "C(k) = sqrt(1/N) for k=0, sqrt(2/N) for k>0"
)

FORMULA_QUANT = formula_html(
    r"\text{Quantized}[u,v] = \text{round}\left(\frac{\text{DCT}[u,v]}{Q[u,v]}\right)",
    "Quantized[u,v] = round(DCT[u,v] / Q[u,v])"
)

FORMULA_DEQUANT = formula_html(
    r"\text{Dequantized}[u,v] = \text{Quantized}[u,v] \times Q[u,v]",
    "Dequantized[u,v] = Quantized[u,v] * Q[u,v]"
)

FORMULA_MSE = formula_html(
    r"\text{MSE} = \frac{1}{MN} \sum_{i,j} \left( I_{\text{orig}}[i,j] - I_{\text{recon}}[i,j] \right)^2",
    "MSE = (1/MN) * sum (original - reconstructed)^2"
)

FORMULA_PSNR = formula_html(
    r"\text{PSNR} = 10 \cdot \log_{10}\left(\frac{255^2}{\text{MSE}}\right) \text{ dB}",
    "PSNR = 10 * log10(255^2 / MSE) dB"
)

FORMULA_SSIM = formula_html(
    r"\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}",
    "SSIM(x,y) = [(2*mu_x*mu_y + C1)(2*sigma_xy + C2)] / [(mu_x^2 + mu_y^2 + C1)(sigma_x^2 + sigma_y^2 + C2)]",
    fontsize=11
)


class TheoryDialog(QDialog):
    """Dialog with DSP theory explanations."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DSP Theory Reference")
        self.setMinimumSize(750, 600)
        self._init_ui()
        self._apply_style()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        title = QLabel("<b>JPEG Compression: Theory Reference</b>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 14px; padding: 10px;")
        layout.addWidget(title)
        
        tabs = QTabWidget()
        tabs.addTab(self._create_dct_tab(), "DCT")
        tabs.addTab(self._create_quantization_tab(), "Quantization")
        tabs.addTab(self._create_subsampling_tab(), "Subsampling")
        tabs.addTab(self._create_metrics_tab(), "Metrics")
        tabs.addTab(self._create_artifacts_tab(), "Artifacts")
        tabs.addTab(self._create_performance_tab(), "Performance")
        tabs.addTab(self._create_about_tab(), "About")
        layout.addWidget(tabs)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
    
    def _create_text_browser(self, html: str) -> QTextBrowser:
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(html)
        browser.setStyleSheet("""
            QTextBrowser {
                background-color: #1e1e1e;
                color: #e0e0e0;
                border: none;
                padding: 10px;
                font-size: 12px;
            }
        """)
        return browser
    
    def _create_dct_tab(self) -> QTextBrowser:
        html = f"""
        <h2 style="color: #4a9eff;">Discrete Cosine Transform</h2>
        
        <h3>Why DCT?</h3>
        <p>Natural images have high spatial correlation. DCT transforms correlated pixels into 
        decorrelated frequency coefficients, concentrating most energy into a few low-frequency terms.</p>
        
        <p>For Markov-1 sources (typical images), DCT approaches the optimal Karhunen-Loeve Transform 
        but with O(N log N) complexity via FFT.</p>
        
        <h3>Level Shift</h3>
        <p>Before DCT, subtract 128 from each pixel:</p>
        <ul>
            <li>Centers 8-bit data [0,255] around zero [-128,127]</li>
            <li>Reduces DC coefficient magnitude</li>
            <li>Matches JPEG specification</li>
        </ul>
        
        <h3>Formula (2D, Orthonormal)</h3>
        {FORMULA_DCT}
        <p>where:</p>
        {FORMULA_C}
        
        <h3>Implementation</h3>
        <p>Using <code>scipy.fft.dctn</code> (modern API, replaces deprecated <code>scipy.fftpack</code>).</p>
        """
        return self._create_text_browser(html)
    
    def _create_quantization_tab(self) -> QTextBrowser:
        html = f"""
        <h2 style="color: #4a9eff;">Quantization</h2>
        
        <h3>The Lossy Step</h3>
        <p>Quantization is where information is permanently discarded. Each DCT coefficient 
        is divided by a step size and rounded to the nearest integer.</p>
        
        {FORMULA_QUANT}
        {FORMULA_DEQUANT}
        
        <h3>JPEG Quantization Matrix</h3>
        <p>The standard matrix uses:</p>
        <ul>
            <li>Small values (top-left) for low frequencies - fine quantization</li>
            <li>Large values (bottom-right) for high frequencies - coarse quantization</li>
        </ul>
        <p>This exploits reduced human sensitivity to high-frequency errors.</p>
        
        <h3>Quality Factor Scaling</h3>
        <pre style="background: #2a2a2a; padding: 10px; border-radius: 4px;">
if quality < 50:
    scale = 5000 / quality
else:
    scale = 200 - 2 * quality

Q = clip(floor((base * scale + 50) / 100), 1, 255)</pre>
        
        <ul>
            <li><b>Q=50:</b> Base matrix unchanged</li>
            <li><b>Q>50:</b> Smaller step sizes, better quality</li>
            <li><b>Q<50:</b> Larger step sizes, more compression</li>
        </ul>
        
        <h3>Compression</h3>
        <p>Coarse quantization produces zeros. With entropy coding, zeros compress 
        efficiently via run-length encoding.</p>
        """
        return self._create_text_browser(html)
    
    def _create_subsampling_tab(self) -> QTextBrowser:
        html = f"""
        <h2 style="color: #4a9eff;">Chroma Subsampling</h2>
        
        <h3>YCbCr Color Space</h3>
        <p>Separates luminance (Y) from chrominance (Cb, Cr). Human vision is more 
        sensitive to brightness than color detail, so chroma can be subsampled.</p>
        
        <h3>Subsampling Modes</h3>
        <table style="border-collapse: collapse; width: 100%;">
            <tr style="background: #3a3a3a;">
                <th style="padding: 8px; border: 1px solid #555;">Mode</th>
                <th style="padding: 8px; border: 1px solid #555;">Chroma Resolution</th>
                <th style="padding: 8px; border: 1px solid #555;">Savings</th>
            </tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">4:4:4</td>
                <td style="padding: 8px; border: 1px solid #555;">Full</td>
                <td style="padding: 8px; border: 1px solid #555;">0%</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">4:2:2</td>
                <td style="padding: 8px; border: 1px solid #555;">Half horizontal</td>
                <td style="padding: 8px; border: 1px solid #555;">33%</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">4:2:0</td>
                <td style="padding: 8px; border: 1px solid #555;">Quarter (2x2)</td>
                <td style="padding: 8px; border: 1px solid #555;">50%</td></tr>
        </table>
        
        <h3>Aliasing</h3>
        <p><b>Nyquist theorem:</b> Sampling rate must be at least 2x the highest frequency. 
        Without low-pass filtering before downsampling, high frequencies fold back as 
        spurious low frequencies (aliasing).</p>
        
        <h3>Prefilter</h3>
        <p>A Gaussian blur before subsampling removes high frequencies, preventing aliasing. 
        Trade-off: slight blur vs. moire patterns and color fringing.</p>
        """
        return self._create_text_browser(html)
    
    def _create_metrics_tab(self) -> QTextBrowser:
        html = f"""
        <h2 style="color: #4a9eff;">Quality Metrics</h2>
        
        <h3>PSNR (Peak Signal-to-Noise Ratio)</h3>
        {FORMULA_MSE}
        {FORMULA_PSNR}
        
        <p>Higher is better. Typical ranges:</p>
        <ul>
            <li>&lt;25 dB: Poor</li>
            <li>25-35 dB: Fair</li>
            <li>&gt;35 dB: Good</li>
            <li>&gt;45 dB: Near-perfect</li>
        </ul>
        <p>Limitation: PSNR measures pixel error but doesn't always match perception. 
        Blur can have high PSNR but look bad.</p>
        
        <h3>SSIM (Structural Similarity)</h3>
        {FORMULA_SSIM}
        
        <p>Compares luminance, contrast, and structure. Range [-1, 1], higher is better.</p>
        <ul>
            <li>&lt;0.8: Visible degradation</li>
            <li>&gt;0.95: Nearly imperceptible</li>
        </ul>
        <p>SSIM correlates better with human perception than PSNR.</p>
        
        <h3>BPP (Bits Per Pixel)</h3>
        <p>Total bits / number of pixels. Our estimate is conservative since we don't 
        implement entropy coding. Real JPEG achieves 2-3x better compression.</p>
        """
        return self._create_text_browser(html)
    
    def _create_artifacts_tab(self) -> QTextBrowser:
        html = f"""
        <h2 style="color: #4a9eff;">Compression Artifacts</h2>
        
        <h3>Block Artifacts</h3>
        <p><b>Cause:</b> Each 8x8 block processed independently. At low quality, 
        adjacent blocks have different DC values.</p>
        <p><b>Appearance:</b> Visible grid pattern, especially in smooth regions.</p>
        
        <h3>Ringing</h3>
        <p><b>Cause:</b> Coarse quantization of high-frequency coefficients near edges.</p>
        <p><b>Appearance:</b> Halos or ripples around sharp edges (Gibbs phenomenon).</p>
        
        <h3>Color Bleeding</h3>
        <p><b>Cause:</b> Chroma subsampling without proper low-pass filtering.</p>
        <p><b>Appearance:</b> Color bleeds across edges. Worst on high-frequency 
        color patterns (red/blue checkerboard).</p>
        
        <h3>Posterization</h3>
        <p><b>Cause:</b> Coarse quantization in gradient regions.</p>
        <p><b>Appearance:</b> Visible steps instead of smooth transitions.</p>
        
        <h3>Quality vs. Artifacts</h3>
        <p>Lower quality = larger quantization steps = more zeros = more information 
        lost = more visible distortion.</p>
        """
        return self._create_text_browser(html)
    
    def _create_performance_tab(self) -> QTextBrowser:
        html = f"""
        <h2 style="color: #4a9eff;">Performance Notes</h2>
        
        <h3>Processing Time</h3>
        <p>Block count scales with image size:</p>
        <ul>
            <li>720p: ~14,400 blocks/channel</li>
            <li>1080p: ~32,400 blocks/channel</li>
            <li>4K: ~129,600 blocks/channel</li>
        </ul>
        <p>Each block requires DCT + quantization + IDCT. With 3 channels, 
        4K processes ~390K blocks vs ~97K for 1080p.</p>
        
        <h3>Preview Mode</h3>
        <p>Downscales image before processing for faster parameter tuning. 
        Metrics are approximate but trends are preserved. Full resolution 
        is used for final export.</p>
        
        <h3>Chroma Prefilter</h3>
        <p>Gaussian blur before subsampling prevents aliasing per Nyquist theorem. 
        Trade-off: slight blur vs. moire patterns.</p>
        
        <h3>Real-ESRGAN GPU Settings</h3>
        <ul>
            <li><b>Tiling:</b> Process in 512x512 chunks to avoid running out of VRAM</li>
            <li><b>FP16:</b> Half precision saves ~50% memory, faster on RTX GPUs</li>
        </ul>
        
        <h3>BPP Estimate</h3>
        <p>Our estimate counts raw coefficient bits without entropy coding. 
        Actual JPEG files are 2-3x smaller due to Huffman coding. 
        Useful for relative comparison (Q=10 vs Q=50) but not file size prediction.</p>
        """
        return self._create_text_browser(html)
    
    def _create_about_tab(self) -> QTextBrowser:
        html = f"""
        <h2 style="color: #4a9eff;">About</h2>
        
        <h3>JPEG-DSP Studio</h3>
        <p>Desktop application demonstrating JPEG-style compression using DSP techniques.</p>
        
        <h3>Pipeline</h3>
        <p>RGB to YCbCr to DCT to Quantize to IDCT to YCbCr to RGB</p>
        
        <h3>Components</h3>
        <ul>
            <li><b>DCT/IDCT:</b> scipy.fft.dctn (Type-II, ortho norm)</li>
            <li><b>Level Shift:</b> -128 before DCT, +128 after IDCT</li>
            <li><b>Quantization:</b> JPEG luminance matrix with quality scaling</li>
            <li><b>Color Space:</b> ITU-R BT.601</li>
            <li><b>Subsampling:</b> 4:4:4, 4:2:2, 4:2:0</li>
        </ul>
        
        <h3>Project Structure</h3>
        <table style="border-collapse: collapse; width: 100%;">
            <tr style="background: #3a3a3a;">
                <th style="padding: 8px; border: 1px solid #555;">Directory</th>
                <th style="padding: 8px; border: 1px solid #555;">Purpose</th>
            </tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">engines/</td>
                <td style="padding: 8px; border: 1px solid #555;">DSP logic</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">models/</td>
                <td style="padding: 8px; border: 1px solid #555;">Data structures</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">utils/</td>
                <td style="padding: 8px; border: 1px solid #555;">Metrics, I/O</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">gui/</td>
                <td style="padding: 8px; border: 1px solid #555;">PySide6 interface</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">enhancement/</td>
                <td style="padding: 8px; border: 1px solid #555;">Real-ESRGAN module</td></tr>
        </table>
        
        <h3>Formulas</h3>
        <table style="border-collapse: collapse; width: 100%;">
            <tr style="background: #3a3a3a;">
                <th style="padding: 8px; border: 1px solid #555;">Operation</th>
                <th style="padding: 8px; border: 1px solid #555;">Formula</th>
            </tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">Level Shift</td>
                <td style="padding: 8px; border: 1px solid #555;">block - 128</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">Quantize</td>
                <td style="padding: 8px; border: 1px solid #555;">round(DCT / Q)</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">Dequantize</td>
                <td style="padding: 8px; border: 1px solid #555;">Quantized * Q</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #555;">PSNR</td>
                <td style="padding: 8px; border: 1px solid #555;">10*log10(255^2/MSE) dB</td></tr>
        </table>
        
        <p style="color: #888; margin-top: 20px; text-align: center;">
        Digital Signal Processing Course Project<br>
        Python, PySide6, NumPy, SciPy
        </p>
        """
        return self._create_text_browser(html)
    
    def _apply_style(self):
        self.setStyleSheet("""
            QDialog { background-color: #2b2b2b; color: #e0e0e0; }
            QTabWidget::pane { border: 1px solid #555; }
            QTabBar::tab { background-color: #3a3a3a; border: 1px solid #555; padding: 8px 16px; }
            QTabBar::tab:selected { background-color: #2b2b2b; }
            QPushButton { background-color: #3a3a3a; border: 1px solid #555; border-radius: 4px; padding: 8px 16px; }
            QPushButton:hover { background-color: #4a4a4a; }
        """)

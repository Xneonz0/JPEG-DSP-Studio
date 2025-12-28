"""PDF Report exporter for compression results."""

import io
from pathlib import Path
from datetime import datetime

import numpy as np
from PySide6.QtCore import QObject, Signal, QThread

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from models.compression_result import CompressionResult
from models.compression_params import CompressionParams
from models.intermediate_data import IntermediateData

# App metadata for footer
APP_VERSION = "1.0"


def _truncate_path(path: str, max_chars: int = 60) -> str:
    """Truncate long paths for PDF display, keeping last ~max_chars with leading '…'."""
    if not path or path == "N/A" or len(path) <= max_chars:
        return path
    return "…" + path[-(max_chars - 1):]


def get_system_snapshot_for_report() -> str:
    """
    Get system snapshot text for PDF report.
    
    Uses the same data as System Info dialog but formatted for PDF.
    Long paths are truncated for readability.
    """
    try:
        from gui.system_info import get_static_info, get_live_info, format_bytes
        static = get_static_info()
        live = get_live_info(static)
        
        # Build PDF-specific snapshot with truncated paths
        lines = []
        
        # Header
        lines.append("JPEG-DSP Studio v1.0")
        lines.append(f"OS: {static.os_display}")
        lines.append(f"Python: {static.python_version} (requires {static.python_requirement})")
        
        # CPU
        cpu_line = f"CPU: {static.cpu_model}"
        if static.cpu_cores_physical > 0:
            cpu_line += f" | Cores: {static.cpu_cores_physical}"
        if static.cpu_cores_logical > 0:
            cpu_line += f" | Threads: {static.cpu_cores_logical}"
        cpu_line += f" | Load: {live.cpu_percent:.0f}%"
        lines.append(cpu_line)
        
        # RAM
        if static.ram_total_bytes > 0:
            lines.append(f"RAM: {live.ram_used_display} / {static.ram_total_display} ({live.ram_percent:.0f}%)")
        else:
            lines.append("RAM: N/A")
        
        # Disk
        if static.disk_total_bytes > 0:
            lines.append(f"Disk ({static.disk_path}): Free {live.disk_free_display} / {static.disk_total_display}")
        else:
            lines.append("Disk: N/A")
        
        # GPU
        gpu_line = f"GPU: {static.gpu_name}"
        gpu_line += f" | CUDA: {'Yes' if static.cuda_available else 'No'}"
        lines.append(gpu_line)
        
        # GPU live stats
        if live.gpu_available:
            gpu_stats = []
            if live.gpu_util_percent >= 0:
                gpu_stats.append(f"Util: {live.gpu_util_percent:.0f}%")
            if live.gpu_memory_total_bytes > 0:
                gpu_stats.append(f"VRAM: {live.gpu_memory_display}")
            if live.gpu_temp_celsius >= 0:
                gpu_stats.append(f"Temp: {live.gpu_temp_celsius}°C")
            if gpu_stats:
                lines.append("GPU Stats: " + " | ".join(gpu_stats))
        else:
            lines.append("GPU Stats: N/A (pynvml not installed)")
        
        # Software
        torch_str = f"v{static.torch_version}" if static.torch_installed else "Not installed"
        esrgan_str = "Installed" if static.realesrgan_installed else "Not installed"
        basicsr_str = "Installed" if static.basicsr_installed else "Not installed"
        lines.append(f"PyTorch: {torch_str} | Real-ESRGAN: {esrgan_str} | basicsr: {basicsr_str}")
        
        # Weights - truncate path for PDF
        lines.append(f"Weights: {static.weights_status}")
        if static.weights_path != "N/A" and static.weights_status.startswith("Cached"):
            truncated_path = _truncate_path(static.weights_path, 60)
            lines.append(f"  Path: {truncated_path}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"System info unavailable: {e}"


class ReportExportWorker(QObject):
    """Worker to generate PDF report in background thread."""
    
    finished = Signal(str)  # Output path
    error = Signal(str)
    progress = Signal(str)
    
    def __init__(self, output_path: str, result: CompressionResult, 
                 params: CompressionParams, intermediate: IntermediateData,
                 batch_results: list = None, enhancement_result=None,
                 image_path: Path = None):
        super().__init__()
        self.output_path = output_path
        self.result = result
        self.params = params
        self.intermediate = intermediate
        self.batch_results = batch_results
        self.enhancement_result = enhancement_result  # Optional EnhancementResult
        self.image_path = image_path  # Original image path for filename
        
        # Page tracking for footer
        self._page_num = 0
        self._total_pages = self._count_pages()
        self._timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    def _count_pages(self) -> int:
        """Count total pages for footer."""
        count = 3  # Summary, Images, Analysis
        if self.batch_results and len(self.batch_results) >= 2:
            count += 1  # Rate-distortion (only if meaningful sweep data)
        if self.enhancement_result is not None:
            count += 1  # Enhancement
        count += 1  # System Snapshot (always included)
        return count
    
    def run(self):
        try:
            self.progress.emit("Generating PDF report...")
            self._generate_report()
            self.finished.emit(self.output_path)
        except Exception as e:
            self.error.emit(str(e))
    
    def _generate_report(self):
        """Generate multi-page PDF report."""
        with PdfPages(self.output_path) as pdf:
            # Page 1: Summary and metrics
            self._add_summary_page(pdf)
            
            # Page 2: Image comparison
            self._add_image_comparison_page(pdf)
            
            # Page 3: Analysis plots
            self._add_analysis_page(pdf)
            
            # Page 4: Rate-distortion (if available)
            if self.batch_results:
                self._add_rd_curve_page(pdf)
            
            # Page 5: Enhancement results (if available)
            if self.enhancement_result is not None:
                self._add_enhancement_page(pdf)
            
            # Final page: System Snapshot (appendix)
            self._add_system_snapshot_page(pdf)
    
    def _add_page_footer(self, fig: Figure):
        """Add consistent footer to page."""
        self._page_num += 1
        footer_text = f"JPEG-DSP Studio v{APP_VERSION} — Page {self._page_num}/{self._total_pages} — {self._timestamp}"
        fig.text(0.5, 0.02, footer_text, ha='center', fontsize=8, color='#888888')
    
    def _add_summary_page(self, pdf):
        """Add summary page with parameters and metrics."""
        fig = Figure(figsize=(8.5, 11))
        fig.set_facecolor('white')
        
        # Title
        fig.suptitle('JPEG Compression Report', fontsize=16, fontweight='bold', y=0.97)
        
        # Date
        fig.text(0.5, 0.94, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                 ha='center', fontsize=10, color='gray')
        
        # Table of Contents
        toc_lines = [
            "Contents:",
            f"  1. Summary & Metrics ............ {1}",
            f"  2. Image Comparison ............. {2}",
            f"  3. Analysis Plots ............... {3}",
        ]
        page_num = 4
        if self.batch_results and len(self.batch_results) >= 2:
            toc_lines.append(f"  4. Rate-Distortion .............. {page_num}")
            page_num += 1
        if self.enhancement_result is not None:
            toc_lines.append(f"  {page_num}. AI Enhancement .............. {page_num}")
            page_num += 1
        toc_lines.append(f"  {page_num}. System Snapshot ............. {page_num}")
        
        fig.text(0.1, 0.88, "\n".join(toc_lines), fontsize=9, family='monospace',
                 color='#555555', verticalalignment='top')
        
        # Input Image section
        ax0 = fig.add_axes([0.1, 0.68, 0.8, 0.08])
        ax0.axis('off')
        ax0.set_title('Input Image', loc='left', fontweight='bold', fontsize=12)
        
        # Get filename and dimensions
        if self.image_path:
            filename = self.image_path.name
        else:
            filename = "(loaded from memory)"
        orig_h, orig_w = self.result.original_image.shape[:2]
        
        input_text = f"Filename: {filename}\nResolution: {orig_w} × {orig_h} pixels"
        ax0.text(0, 0.7, input_text, fontsize=11, family='monospace',
                 verticalalignment='top', transform=ax0.transAxes)
        
        # Parameters section
        ax1 = fig.add_axes([0.1, 0.52, 0.8, 0.12])
        ax1.axis('off')
        ax1.set_title('Compression Parameters', loc='left', fontweight='bold', fontsize=12)
        
        params_text = (
            f"Block Size: {self.params.block_size}×{self.params.block_size}\n"
            f"Quality Factor: {self.params.quality}\n"
            f"Subsampling: {self.params.subsampling_mode}\n"
            f"Prefilter: {'Yes' if self.params.use_prefilter else 'No'}"
        )
        ax1.text(0, 0.9, params_text, fontsize=11, family='monospace',
                 verticalalignment='top', transform=ax1.transAxes)
        
        # Metrics table
        ax2 = fig.add_axes([0.1, 0.15, 0.8, 0.32])
        ax2.axis('off')
        ax2.set_title('Quality Metrics', loc='left', fontweight='bold', fontsize=12)
        
        # Build bitrate label with note
        bpp_label = f'{self.result.bpp:.3f} (estimated, no entropy coding)'
        
        table_data = [
            ['Metric', 'Value'],
            ['PSNR (Y)', f'{self.result.psnr_y:.2f} dB'],
            ['SSIM (Y)', f'{self.result.ssim_y:.4f}'],
            ['PSNR (RGB)', f'{self.result.psnr_rgb:.2f} dB'],
            ['SSIM (RGB)', f'{self.result.ssim_rgb:.4f}'],
            ['Rate Estimate (BPP)', bpp_label],
            ['Compression Ratio', f'{self.result.compression_ratio:.2f}:1'],
            ['Non-zero Coefficients', f'{self.result.nonzero_coeffs:,} / {self.result.total_coeffs:,}'],
            ['Encode Time', f'{self.result.encode_time_ms:.2f} ms'],
            ['Decode Time', f'{self.result.decode_time_ms:.2f} ms'],
        ]
        
        table = ax2.table(cellText=table_data[1:], colLabels=table_data[0],
                          loc='upper left', cellLoc='left',
                          colWidths=[0.4, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)
        
        # Style header
        for j in range(2):
            table[(0, j)].set_facecolor('#4a9eff')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        # Interpretation note
        interpretation = "Interpretation: Higher PSNR/SSIM = higher fidelity; higher BPP = lower compression."
        fig.text(0.1, 0.10, interpretation, fontsize=9, style='italic', color='#555555')
        
        # Method Notes line
        prefilter_str = "prefilter ON" if self.params.use_prefilter else "prefilter OFF"
        method_notes = (
            f"Method: DCT-II (ortho), level shift −128, {self.params.block_size}×{self.params.block_size} blocks, "
            f"Q={self.params.quality}, {self.params.subsampling_mode}, {prefilter_str}, estimated BPP (no entropy coding)"
        )
        fig.text(0.1, 0.07, method_notes, fontsize=8, color='#666666', family='monospace')
        
        self._add_page_footer(fig)
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_image_comparison_page(self, pdf):
        """Add page with original and reconstructed image thumbnails."""
        fig = Figure(figsize=(8.5, 11))
        fig.set_facecolor('white')
        fig.suptitle('Image Comparison', fontsize=14, fontweight='bold', y=0.96)
        
        orig_h, orig_w = self.result.original_image.shape[:2]
        recon_h, recon_w = self.result.reconstructed_image.shape[:2]
        
        # Larger images with less whitespace
        # Original image
        ax1 = fig.add_axes([0.05, 0.25, 0.42, 0.60])
        ax1.imshow(self.result.original_image)
        ax1.axis('off')
        # Caption below image
        ax1.set_title(f'Original ({orig_w}×{orig_h})', fontsize=11, pad=5)
        
        # Reconstructed image
        ax2 = fig.add_axes([0.53, 0.25, 0.42, 0.60])
        ax2.imshow(self.result.reconstructed_image)
        ax2.axis('off')
        ax2.set_title(f'Reconstructed ({recon_w}×{recon_h})', fontsize=11, pad=5)
        
        # Quality summary below images
        quality_text = f"Q={self.params.quality}  |  {self.params.subsampling_mode}  |  PSNR: {self.result.psnr_y:.2f} dB  |  SSIM: {self.result.ssim_y:.4f}"
        fig.text(0.5, 0.18, quality_text, ha='center', fontsize=10, color='#333333')
        
        # Image quality note
        fig.text(0.5, 0.12, "Note: Images are embedded as thumbnails for PDF size. Full resolution available in exported PNG.",
                 ha='center', fontsize=8, style='italic', color='#888888')
        
        self._add_page_footer(fig)
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_analysis_page(self, pdf):
        """Add page with DCT heatmap, histogram, and error map."""
        fig = Figure(figsize=(8.5, 11))
        fig.set_facecolor('white')
        fig.suptitle('Analysis Plots', fontsize=14, fontweight='bold', y=0.96)
        
        # Get selected block info
        block_idx = self.intermediate.selected_block_idx if self.intermediate else "(0, 0)"
        
        # DCT Heatmap
        ax1 = fig.add_subplot(2, 2, 1)
        if self.intermediate and self.intermediate.selected_block_dct is not None:
            dct_data = np.log10(np.abs(self.intermediate.selected_block_dct) + 1)
            im = ax1.imshow(dct_data, cmap='viridis', aspect='equal')
            fig.colorbar(im, ax=ax1, shrink=0.8)
        ax1.set_title(f'DCT Coefficients (log scale)\nSelected block: {block_idx}', fontsize=10)
        
        # Histogram with log scale for better visualization
        ax2 = fig.add_subplot(2, 2, 2)
        if self.intermediate and self.intermediate.all_quantized_coeffs is not None:
            coeffs = self.intermediate.all_quantized_coeffs.flatten()
            ax2.hist(coeffs, bins=50, color='#4a9eff', edgecolor='#2b6cb0', alpha=0.8)
            ax2.set_yscale('log')  # Log scale to show distribution better
            
            # Count zeros for annotation
            zero_count = np.sum(coeffs == 0)
            total_count = len(coeffs)
            zero_pct = 100 * zero_count / total_count if total_count > 0 else 0
            
            # Add annotation about zeros
            ax2.annotate(f'{zero_pct:.1f}% zeros', 
                        xy=(0.02, 0.95), xycoords='axes fraction',
                        fontsize=9, color='#d63031', fontweight='bold')
        ax2.set_title('Quantized Coefficient Distribution (log scale)', fontsize=10)
        ax2.set_xlabel('Coefficient Value', fontsize=9)
        ax2.set_ylabel('Count (log)', fontsize=9)
        ax2.tick_params(axis='both', labelsize=8)
        
        # Note under histogram
        fig.text(0.55, 0.47, 
                 "Note: Large spike at 0 = many coefficients quantized to zero → better compression.",
                 fontsize=8, color='#555555', ha='left', style='italic')
        
        # Error Map with block grid overlay
        ax3 = fig.add_subplot(2, 2, 3)
        if self.intermediate and self.intermediate.error_map_y is not None:
            error_display = np.clip(self.intermediate.error_map_y * 10, 0, 255)
            im = ax3.imshow(error_display, cmap='hot', aspect='equal')
            fig.colorbar(im, ax=ax3, shrink=0.8)
            
            # Add 8x8 block grid overlay
            h, w = error_display.shape[:2]
            block_size = self.params.block_size
            for x in range(0, w, block_size):
                ax3.axvline(x, color='white', linewidth=0.3, alpha=0.4)
            for y in range(0, h, block_size):
                ax3.axhline(y, color='white', linewidth=0.3, alpha=0.4)
        ax3.set_title(f'Error Map (Y, 10× amplified, {self.params.block_size}×{self.params.block_size} grid)', fontsize=9)
        ax3.axis('off')
        
        # Quantization matrix
        ax4 = fig.add_subplot(2, 2, 4)
        from engines.quantizer import scale_quant_matrix
        from utils.constants import JPEG_LUMA_Q50
        Q = scale_quant_matrix(JPEG_LUMA_Q50, self.params.quality)
        im = ax4.imshow(Q, cmap='YlOrRd', aspect='equal')
        fig.colorbar(im, ax=ax4, shrink=0.8)
        ax4.set_title(f'Quantization Matrix (Q={self.params.quality})', fontsize=10)
        
        fig.tight_layout(rect=[0, 0.05, 1, 0.94])
        self._add_page_footer(fig)
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_rd_curve_page(self, pdf):
        """Add rate-distortion curve page."""
        # Skip entirely if no batch data or empty list
        if not self.batch_results or len(self.batch_results) < 2:
            return
        
        try:
            # Validate batch results structure
            qualities = [r[0] for r in self.batch_results]
            if not qualities:
                return
            q_min, q_max = min(qualities), max(qualities)
        except (TypeError, IndexError, ValueError):
            # Malformed batch data - skip page
            return
        
        fig = Figure(figsize=(8.5, 11))
        fig.set_facecolor('white')
        
        fig.suptitle(f'Rate-Distortion Analysis (Q ∈ [{q_min}, {q_max}])', 
                     fontsize=14, fontweight='bold', y=0.96)
        
        bpp_values = [r[1].bpp for r in self.batch_results]
        psnr_y_values = [r[1].psnr_y for r in self.batch_results]
        ssim_y_values = [r[1].ssim_y for r in self.batch_results]
        
        # PSNR vs BPP
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(bpp_values, psnr_y_values, 'o-', color='#4a9eff', 
                 linewidth=2, markersize=6, label='PSNR (Y)')
        
        # Mark current quality with star
        if self.params.quality in qualities:
            idx = qualities.index(self.params.quality)
            ax1.scatter([bpp_values[idx]], [psnr_y_values[idx]], 
                        s=200, c='#e74c3c', marker='*', zorder=5, 
                        label=f'★ Current Q={self.params.quality}')
        
        ax1.set_xlabel('BPP (bits per pixel)', fontsize=10)
        ax1.set_ylabel('PSNR (dB)', fontsize=10)
        ax1.set_title('PSNR vs Bitrate', fontsize=11)
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='both', labelsize=9)
        
        # SSIM vs BPP
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(bpp_values, ssim_y_values, 'o-', color='#51cf66', 
                 linewidth=2, markersize=6, label='SSIM (Y)')
        
        if self.params.quality in qualities:
            idx = qualities.index(self.params.quality)
            ax2.scatter([bpp_values[idx]], [ssim_y_values[idx]], 
                        s=200, c='#e74c3c', marker='*', zorder=5, 
                        label=f'★ Current Q={self.params.quality}')
        
        ax2.set_xlabel('BPP (bits per pixel)', fontsize=10)
        ax2.set_ylabel('SSIM', fontsize=10)
        ax2.set_title('SSIM vs Bitrate', fontsize=11)
        ax2.legend(loc='lower right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='both', labelsize=9)
        
        # Legend explanation
        fig.text(0.5, 0.08, "★ Star marker = current Quality setting", 
                 ha='center', fontsize=9, color='#555555', style='italic')
        
        fig.tight_layout(rect=[0, 0.10, 1, 0.94])
        self._add_page_footer(fig)
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_enhancement_page(self, pdf):
        """Add enhancement results page (bonus RTX enhancement)."""
        if self.enhancement_result is None:
            return
        
        er = self.enhancement_result
        
        fig = Figure(figsize=(8.5, 11))
        fig.set_facecolor('white')
        fig.suptitle('Bonus: AI Enhancement (Real-ESRGAN)', fontsize=16, fontweight='bold', y=0.96)
        
        # Explanatory sentence
        fig.text(0.5, 0.92, 
                 'Real-ESRGAN runs ×4 super-resolution, then resizes/crops to the selected target preset.',
                 ha='center', fontsize=9, style='italic', color='#555555')
        
        # Image comparison - top section (3 images: input, output, difference)
        # Input thumbnail
        ax1 = fig.add_axes([0.03, 0.56, 0.30, 0.30])
        ax1.imshow(er.input_thumbnail)
        ax1.set_title(f'Input\n({er.input_width}×{er.input_height})', fontsize=10)
        ax1.axis('off')
        
        # Output thumbnail
        ax2 = fig.add_axes([0.35, 0.56, 0.30, 0.30])
        ax2.imshow(er.output_thumbnail)
        ax2.set_title(f'Enhanced\n({er.output_width}×{er.output_height})', fontsize=10)
        ax2.axis('off')
        
        # Difference heatmap
        ax3 = fig.add_axes([0.67, 0.56, 0.30, 0.30])
        diff_map = self._compute_difference_map(er.input_thumbnail, er.output_thumbnail)
        if diff_map is not None:
            im = ax3.imshow(diff_map, cmap='hot', vmin=0, vmax=255)
            ax3.set_title('Difference\n(|enhanced - input|)', fontsize=10)
            # Small colorbar
            cbar_ax = fig.add_axes([0.94, 0.58, 0.015, 0.26])
            fig.colorbar(im, cax=cbar_ax)
        else:
            ax3.text(0.5, 0.5, 'Size mismatch', ha='center', va='center', fontsize=9, color='gray')
            ax3.set_title('Difference', fontsize=10)
        ax3.axis('off')
        
        # Calculate scale factor
        scale_factor = max(er.output_width / er.input_width, er.output_height / er.input_height)
        
        # Settings and metrics table - bottom section
        ax4 = fig.add_axes([0.1, 0.12, 0.8, 0.38])
        ax4.axis('off')
        ax4.set_title('Enhancement Details', loc='left', fontweight='bold', fontsize=12)
        
        # Build device info string
        if er.gpu_name:
            device_str = f"{er.device} ({er.gpu_name})"
        else:
            device_str = er.device
        
        # Build method string
        if er.sr_skipped:
            method_str = f"{er.method} (SR skipped, scale < 1.1×)"
        elif er.used_gpu:
            method_str = f"{er.method} + resize"
        else:
            method_str = er.method
        
        # Tile size display
        tile_str = "Auto (512)" if er.tile_size == 512 else str(er.tile_size)
        
        table_data = [
            ['Setting', 'Value'],
            ['Target Preset', er.target_preset],
            ['Mode', er.mode.capitalize()],
            ['Scale Factor', f'≈ {scale_factor:.2f}×'],
            ['Device', device_str],
            ['Method', method_str],
            ['FP16 Autocast', 'Yes' if er.fp16 else 'No'],
            ['Tile Size', tile_str],
            ['Runtime', f'{er.runtime_seconds:.2f}s'],
            ['Input Size', f'{er.input_width}×{er.input_height}'],
            ['Target Size', f'{er.target_width}×{er.target_height}'],
            ['Output Size', f'{er.output_width}×{er.output_height}'],
        ]
        
        table = ax4.table(
            cellText=table_data[1:], 
            colLabels=table_data[0],
            loc='upper left', 
            cellLoc='left',
            colWidths=[0.35, 0.65]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)
        
        # Style header row
        for j in range(2):
            table[(0, j)].set_facecolor('#51cf66')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        # Highlight GPU usage row
        if er.used_gpu:
            table[(5, 1)].set_text_props(color='#2d8a2e', fontweight='bold')
        
        # Mode explanation
        if er.mode.lower() == 'fit':
            mode_note = "Fit mode: Preserves aspect ratio; output may be smaller than target (fits inside bounding box)."
        else:
            mode_note = "Fill mode: Fills target exactly; may crop edges to preserve aspect ratio."
        fig.text(0.1, 0.08, mode_note, fontsize=9, style='italic', color='#555555')
        
        self._add_page_footer(fig)
        pdf.savefig(fig)
        plt.close(fig)
    
    def _add_system_snapshot_page(self, pdf):
        """Add system snapshot appendix page."""
        fig = Figure(figsize=(8.5, 11))
        fig.set_facecolor('white')
        fig.suptitle('Appendix: System Snapshot', fontsize=14, fontweight='bold', y=0.96)
        
        # Get system snapshot text
        snapshot_text = get_system_snapshot_for_report()
        
        # Create text area
        ax = fig.add_axes([0.08, 0.10, 0.84, 0.80])
        ax.axis('off')
        
        # Title for the section
        ax.text(0, 1.0, "System Information at Report Generation",
                fontsize=12, fontweight='bold', color='#333333',
                verticalalignment='top', transform=ax.transAxes)
        
        # System info in monospace (compact format)
        ax.text(0, 0.93, snapshot_text,
                fontsize=9, family='monospace', color='#333333',
                verticalalignment='top', transform=ax.transAxes,
                linespacing=1.4)
        
        # Note at bottom
        note_text = (
            "Note: This snapshot captures the system state at report generation time. "
            "GPU stats may vary during actual processing."
        )
        fig.text(0.5, 0.05, note_text, ha='center', fontsize=8, 
                 style='italic', color='#888888')
        
        self._add_page_footer(fig)
        pdf.savefig(fig)
        plt.close(fig)
    
    def _compute_difference_map(self, input_thumb: np.ndarray, output_thumb: np.ndarray) -> np.ndarray | None:
        """
        Compute absolute difference between input and output thumbnails.
        
        Resizes to match if needed, computes grayscale difference.
        
        Returns:
            Grayscale difference map, or None if computation fails
        """
        import cv2
        
        try:
            # Resize output to match input size for fair comparison
            h1, w1 = input_thumb.shape[:2]
            h2, w2 = output_thumb.shape[:2]
            
            if (h1, w1) != (h2, w2):
                # Resize output to input size
                output_resized = cv2.resize(output_thumb, (w1, h1), interpolation=cv2.INTER_AREA)
            else:
                output_resized = output_thumb
            
            # Convert to float for difference
            input_f = input_thumb.astype(np.float32)
            output_f = output_resized.astype(np.float32)
            
            # Compute per-channel absolute difference, then average
            diff = np.abs(output_f - input_f)
            diff_gray = np.mean(diff, axis=2)  # Average across RGB channels
            
            # Scale to 0-255 for visualization (amplify small differences)
            diff_amplified = np.clip(diff_gray * 2, 0, 255).astype(np.uint8)
            
            return diff_amplified
            
        except Exception:
            return None

"""Matplotlib canvas widget for embedding plots in Qt."""

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy


class MplCanvas(FigureCanvas):
    """
    Matplotlib canvas for embedding in PySide6.
    
    Supports: heatmaps, histograms, and line curves.
    """
    
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.set_facecolor('#2b2b2b')
        self.axes = self.fig.add_subplot(111)
        self._style_axes(self.axes)
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        # Size policy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
    
    def _style_axes(self, ax):
        """Apply dark theme to axes."""
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white', which='both')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#555555')
    
    def clear(self):
        """Clear the plot."""
        self.axes.clear()
        self._style_axes(self.axes)
        self.draw()
    
    def plot_heatmap(self, data: np.ndarray, title: str = "", colorbar: bool = True, log_scale: bool = False):
        """
        Plot a 2D heatmap (e.g., DCT coefficients).
        
        Args:
            data: 2D array to display
            title: Plot title
            colorbar: Whether to show colorbar
            log_scale: Use log scale for magnitude
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._style_axes(ax)
        
        if log_scale and data is not None:
            # Log scale for better visualization of DCT magnitudes
            display_data = np.log10(np.abs(data) + 1)
        else:
            display_data = data
        
        if data is not None:
            im = ax.imshow(display_data, cmap='viridis', aspect='equal')
            if colorbar:
                cbar = self.fig.colorbar(im, ax=ax)
                cbar.ax.yaxis.set_tick_params(color='white')
                cbar.outline.set_edgecolor('#555555')
                for label in cbar.ax.yaxis.get_ticklabels():
                    label.set_color('white')
        
        ax.set_title(title, color='white', fontsize=10)
        ax.set_xticks(range(data.shape[1]) if data is not None else [])
        ax.set_yticks(range(data.shape[0]) if data is not None else [])
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_histogram(self, data: np.ndarray, title: str = "", bins: int = 50):
        """
        Plot histogram of values.
        
        Args:
            data: 1D array of values
            title: Plot title
            bins: Number of histogram bins
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._style_axes(ax)
        
        if data is not None and len(data) > 0:
            ax.hist(data.flatten(), bins=bins, color='#4a9eff', edgecolor='#2b6cb0', alpha=0.8)
        
        ax.set_title(title, color='white', fontsize=10)
        ax.set_xlabel('Value', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_error_map(self, data: np.ndarray, title: str = ""):
        """
        Plot error heatmap with enhanced visibility.
        
        Args:
            data: 2D error array
            title: Plot title
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._style_axes(ax)
        
        if data is not None:
            # Amplify for visibility
            display_data = np.clip(data * 10, 0, 255)
            im = ax.imshow(display_data, cmap='hot', aspect='equal')
            cbar = self.fig.colorbar(im, ax=ax)
            cbar.ax.yaxis.set_tick_params(color='white')
            cbar.outline.set_edgecolor('#555555')
            for label in cbar.ax.yaxis.get_ticklabels():
                label.set_color('white')
        
        ax.set_title(title, color='white', fontsize=10)
        ax.axis('off')
        
        self.fig.tight_layout()
        self.draw()
    
    def plot_curves(self, x_data: list, y_data_list: list, labels: list,
                    title: str = "", xlabel: str = "", ylabel: str = ""):
        """
        Plot multiple curves (e.g., PSNR vs BPP).
        
        Args:
            x_data: X-axis values (shared)
            y_data_list: List of Y-axis value arrays
            labels: List of labels for each curve
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        self._style_axes(ax)
        
        colors = ['#4a9eff', '#ff6b6b', '#51cf66', '#ffd43b', '#be4bdb']
        
        for i, (y_data, label) in enumerate(zip(y_data_list, labels)):
            color = colors[i % len(colors)]
            ax.plot(x_data, y_data, 'o-', color=color, label=label, linewidth=2, markersize=6)
        
        ax.set_title(title, color='white', fontsize=10)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(facecolor='#2b2b2b', edgecolor='#555555', labelcolor='white')
        ax.grid(True, alpha=0.3, color='#555555')
        
        self.fig.tight_layout()
        self.draw()


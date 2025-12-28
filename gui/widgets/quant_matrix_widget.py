"""Quantization matrix preview widget."""

import numpy as np
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from engines.quantizer import scale_quant_matrix
from utils.constants import JPEG_LUMA_Q50


class QuantMatrixWidget(QWidget):
    """
    Widget displaying the current 8x8 quantization matrix.
    Updates instantly when quality changes.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._quality = 50
        self._init_ui()
        self._update_display()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(2)
        
        # Title
        self._title_label = QLabel("Quantization Matrix (Q=50)")
        self._title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._title_label.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(self._title_label)
        
        # Matrix grid
        self._grid_widget = QWidget()
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setSpacing(1)
        self._grid_layout.setContentsMargins(2, 2, 2, 2)
        
        # Create 8x8 grid of labels
        self._cells = []
        cell_font = QFont("Consolas", 7)
        for i in range(8):
            row = []
            for j in range(8):
                cell = QLabel("0")
                cell.setAlignment(Qt.AlignmentFlag.AlignCenter)
                cell.setFont(cell_font)
                cell.setMinimumSize(28, 20)
                cell.setMaximumSize(40, 25)
                self._grid_layout.addWidget(cell, i, j)
                row.append(cell)
            self._cells.append(row)
        
        layout.addWidget(self._grid_widget)
        
        # Info label
        self._info_label = QLabel("Higher values = more quantization")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("color: #888; font-size: 9px;")
        layout.addWidget(self._info_label)
    
    def set_quality(self, quality: int):
        """Update the displayed matrix for new quality value."""
        self._quality = quality
        self._update_display()
    
    def _update_display(self):
        """Recalculate and display the quantization matrix."""
        # Compute scaled matrix using engine logic
        Q = scale_quant_matrix(JPEG_LUMA_Q50, self._quality)
        
        # Update title
        self._title_label.setText(f"Quantization Matrix (Q={self._quality})")
        
        # Find min/max for color scaling
        q_min, q_max = Q.min(), Q.max()
        
        # Update cells
        for i in range(8):
            for j in range(8):
                val = int(Q[i, j])
                self._cells[i][j].setText(str(val))
                
                # Color code: low values (blue) = fine, high values (red) = coarse
                if q_max > q_min:
                    ratio = (Q[i, j] - q_min) / (q_max - q_min)
                else:
                    ratio = 0.5
                
                # Gradient from blue (#4a9eff) to red (#ff6b6b)
                r = int(74 + ratio * (255 - 74))
                g = int(158 - ratio * (158 - 107))
                b = int(255 - ratio * (255 - 107))
                
                self._cells[i][j].setStyleSheet(
                    f"background-color: rgb({r},{g},{b}); "
                    f"color: {'white' if ratio > 0.5 else 'black'}; "
                    f"border: 1px solid #444; border-radius: 2px;"
                )


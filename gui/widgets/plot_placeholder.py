"""Placeholder widget for plot areas when no data is available."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt


class PlotPlaceholder(QWidget):
    """
    A placeholder widget shown when no plot data is available.
    
    Displays a centered message with icon.
    """
    
    def __init__(self, message: str = "Run compression to generate analysis plots", 
                 icon: str = "📊", parent=None):
        super().__init__(parent)
        
        self._message = message
        self._icon = icon
        
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 40, 20, 40)
        
        # Container for centering
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Icon
        icon_label = QLabel(self._icon)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet("font-size: 48px; color: #555;")
        container_layout.addWidget(icon_label)
        
        # Message
        msg_label = QLabel(self._message)
        msg_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg_label.setWordWrap(True)
        msg_label.setStyleSheet("color: #888; font-size: 12px; margin-top: 10px;")
        container_layout.addWidget(msg_label)
        
        layout.addStretch()
        layout.addWidget(container)
        layout.addStretch()
        
        self.setStyleSheet("""
            PlotPlaceholder {
                background-color: #1e1e1e;
                border: 1px dashed #444;
                border-radius: 4px;
            }
        """)
    
    def set_message(self, message: str, icon: str = None):
        """Update the placeholder message."""
        self._message = message
        if icon:
            self._icon = icon
        
        # Rebuild UI
        for child in self.children():
            if isinstance(child, QWidget):
                child.deleteLater()
        self._init_ui()


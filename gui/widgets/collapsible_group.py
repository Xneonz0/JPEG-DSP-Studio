"""Collapsible group box with animation."""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFrame, QSizePolicy
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, Signal, QSettings


class CollapsibleGroupBox(QWidget):
    """Collapsible group box with smooth expand/collapse animation."""
    
    toggled = Signal(bool)
    
    def __init__(self, title: str, parent=None, settings_key: str = None, initially_expanded: bool = True):
        super().__init__(parent)
        
        self._title = title
        self._settings_key = settings_key
        self._expanded = initially_expanded
        self._animation_duration = 200
        
        if settings_key:
            settings = QSettings("DSPLab", "DSPLab")
            self._expanded = settings.value(f"collapse/{settings_key}", initially_expanded, type=bool)
        
        self._init_ui()
    
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        self._header = QPushButton()
        self._header.setCheckable(True)
        self._header.setChecked(self._expanded)
        self._header.clicked.connect(self._on_toggle)
        self._header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._header.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #404040, stop:1 #363636);
                border: 1px solid #4a4a4a;
                border-radius: 5px;
                padding: 6px 12px;
                text-align: left;
                font-weight: 600;
                font-size: 11px;
                color: #e0e0e0;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #404040); }
            QPushButton:pressed { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #353535, stop:1 #3a3a3a); }
            QPushButton:checked { border-bottom-left-radius: 0; border-bottom-right-radius: 0; }
        """)
        self._update_header_text()
        main_layout.addWidget(self._header)
        
        self._content_container = QFrame()
        self._content_container.setStyleSheet("""
            QFrame {
                background-color: #2a2a2a;
                border: 1px solid #3d3d3d;
                border-top: none;
                border-bottom-left-radius: 6px;
                border-bottom-right-radius: 6px;
            }
        """)
        self._content_layout = QVBoxLayout(self._content_container)
        self._content_layout.setContentsMargins(6, 6, 6, 6)
        self._content_layout.setSpacing(4)
        
        main_layout.addWidget(self._content_container)
        
        if not self._expanded:
            self._content_container.setMaximumHeight(0)
            self._content_container.setVisible(False)
    
    def _update_header_text(self):
        arrow = "▼" if self._expanded else "▶"
        self._header.setText(f"{arrow}  {self._title}")
    
    def _on_toggle(self):
        self._expanded = self._header.isChecked()
        self._update_header_text()
        
        if self._expanded:
            self._expand()
        else:
            self._collapse()
        
        if self._settings_key:
            settings = QSettings("DSPLab", "DSPLab")
            settings.setValue(f"collapse/{self._settings_key}", self._expanded)
        
        self.toggled.emit(self._expanded)
    
    def _expand(self):
        self._content_container.setVisible(True)
        content_height = self._content_container.sizeHint().height()
        
        anim = QPropertyAnimation(self._content_container, b"maximumHeight", self)
        anim.setDuration(self._animation_duration)
        anim.setStartValue(0)
        anim.setEndValue(content_height)
        anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        anim.finished.connect(lambda: self._content_container.setMaximumHeight(16777215))
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    
    def _collapse(self):
        content_height = self._content_container.height()
        
        anim = QPropertyAnimation(self._content_container, b"maximumHeight", self)
        anim.setDuration(self._animation_duration)
        anim.setStartValue(content_height)
        anim.setEndValue(0)
        anim.setEasingCurve(QEasingCurve.Type.InQuad)
        anim.finished.connect(lambda: self._content_container.setVisible(False))
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    
    def add_widget(self, widget: QWidget):
        self._content_layout.addWidget(widget)
    
    def add_layout(self, layout):
        self._content_layout.addLayout(layout)
    
    def content_layout(self):
        return self._content_layout
    
    def set_expanded(self, expanded: bool):
        if expanded != self._expanded:
            self._header.setChecked(expanded)
            self._on_toggle()
    
    def is_expanded(self) -> bool:
        return self._expanded

"""Image viewer with zoom, pan, and block selection."""

import numpy as np
from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGraphicsOpacityEffect, QFrame
)
from PySide6.QtGui import (
    QPixmap, QImage, QPen, QColor, QWheelEvent, QMouseEvent,
    QDragEnterEvent, QDropEvent, QDragMoveEvent, QPainter, QBrush, QFont
)
from PySide6.QtCore import (
    Qt, Signal, QRectF, QPropertyAnimation, QEasingCurve, Property, QMimeData
)


class ImageViewer(QGraphicsView):
    """QGraphicsView with zoom/pan and 8x8 block selection for DCT analysis."""
    
    blockClicked = Signal(int, int)
    viewChanged = Signal()
    imageDropped = Signal(str)
    
    def __init__(self, parent=None, block_size: int = 8, label: str = ""):
        super().__init__(parent)
        
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        
        self._pixmap_item = None
        self._block_overlay = None
        self._image_array = None
        self._block_size = block_size
        self._selected_block = None
        self._label = label
        
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self.setMinimumSize(200, 200)
        
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        
        self._zoom_factor = 1.0
        self._min_zoom = 0.1
        self._max_zoom = 20.0
        
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)
        
        self._placeholder_item = None
        self._show_placeholder()
    
    def set_label(self, label: str):
        self._label = label
    
    def set_block_size(self, block_size: int):
        self._block_size = block_size
        self._update_overlay()
    
    def set_image(self, image: np.ndarray, animate: bool = True):
        """Display RGB numpy array."""
        self._image_array = image
        h, w, c = image.shape
        bytes_per_line = 3 * w
        qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        
        self._scene.clear()
        self._placeholder_item = False
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._block_overlay = None
        self._selected_block = None
        
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom_factor = self.transform().m11()
        
        if animate:
            self._fade_in()
    
    def _fade_in(self):
        self._opacity_effect.setOpacity(0.0)
        anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        anim.setDuration(200)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    
    def clear_image(self):
        self._scene.clear()
        self._pixmap_item = None
        self._block_overlay = None
        self._image_array = None
        self._selected_block = None
        self._placeholder_item = False
        self._show_placeholder()
    
    def _show_placeholder(self):
        self._scene.clear()
        self._pixmap_item = None
        self._placeholder_item = True
        self.setSceneRect(QRectF(-100, -100, 200, 200))
        self.resetTransform()
        self.viewport().update()
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        if self._placeholder_item and self._pixmap_item is None:
            painter = QPainter(self.viewport())
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            rect = self.viewport().rect()
            cx, cy = rect.width() // 2, rect.height() // 2
            
            painter.setFont(QFont("Segoe UI", 11))
            painter.setPen(QColor(100, 100, 100))
            text = "Drop an image here"
            tw = painter.fontMetrics().horizontalAdvance(text)
            painter.drawText(cx - tw // 2, cy - 5, text)
            
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor(80, 80, 80))
            text2 = "Ctrl+O • Demo menu"
            tw2 = painter.fontMetrics().horizontalAdvance(text2)
            painter.drawText(cx - tw2 // 2, cy + 18, text2)
            painter.end()
    
    def swap_pixmap(self, image: np.ndarray):
        """Swap image without resetting zoom (for A/B toggle)."""
        if self._pixmap_item is None:
            return
        h, w, c = image.shape
        bytes_per_line = 3 * w
        qimage = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimage))
    
    def reset_view(self):
        if self._pixmap_item:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom_factor = self.transform().m11()
            self.viewChanged.emit()
    
    def get_image_dimensions(self) -> tuple[int, int] | None:
        if self._image_array is not None:
            h, w = self._image_array.shape[:2]
            return (w, h)
        return None
    
    def has_image(self) -> bool:
        return self._image_array is not None
    
    def set_selected_block(self, block_row: int, block_col: int):
        self._selected_block = (block_row, block_col)
        self._update_overlay()
    
    def _update_overlay(self):
        if self._block_overlay:
            self._scene.removeItem(self._block_overlay)
            self._block_overlay = None
        
        if self._selected_block is None or self._image_array is None:
            return
        
        row, col = self._selected_block
        x = col * self._block_size
        y = row * self._block_size
        
        pen = QPen(QColor(80, 200, 220))
        pen.setWidth(2)
        pen.setCosmetic(True)
        pen.setStyle(Qt.PenStyle.DashLine)
        
        self._block_overlay = self._scene.addRect(
            x, y, self._block_size, self._block_size, pen
        )
        self._block_overlay.setBrush(QBrush(QColor(80, 200, 220, 30)))
    
    def wheelEvent(self, event: QWheelEvent):
        if self._image_array is None:
            event.ignore()
            return
        
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = self._zoom_factor * factor
        
        if self._min_zoom <= new_zoom <= self._max_zoom:
            self._zoom_factor = new_zoom
            self.scale(factor, factor)
        event.accept()
    
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._image_array is not None:
            pos = self.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())
            h, w = self._image_array.shape[:2]
            
            if 0 <= x < w and 0 <= y < h:
                col = x // self._block_size
                row = y // self._block_size
                self.set_selected_block(row, col)
                self.blockClicked.emit(row, col)
        
        super().mousePressEvent(event)
    
    def sync_view(self, other: 'ImageViewer'):
        self.setTransform(other.transform())
        self._zoom_factor = other._zoom_factor
        self.horizontalScrollBar().setValue(other.horizontalScrollBar().value())
        self.verticalScrollBar().setValue(other.verticalScrollBar().value())
    
    def get_zoom_factor(self) -> float:
        return self._zoom_factor
    
    def _is_valid_image_drop(self, mime_data) -> str | None:
        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                path = urls[0].toLocalFile()
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    return path
        return None
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._is_valid_image_drop(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event: QDragMoveEvent):
        if self._is_valid_image_drop(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        path = self._is_valid_image_drop(event.mimeData())
        if path:
            self.imageDropped.emit(path)
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()


class ImageViewerWithControls(QWidget):
    """ImageViewer with header bar (label, resolution badge, buttons)."""
    
    blockClicked = Signal(int, int)
    viewChanged = Signal()
    imageDropped = Signal(str)
    clearClicked = Signal()
    
    def __init__(self, label: str = "", parent=None, block_size: int = 8):
        super().__init__(parent)
        self._label = label
        self.setAcceptDrops(True)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        
        # Header
        header = QFrame()
        header.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        header.setAcceptDrops(False)
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #3a3a3a, stop:1 #303030);
                border: 1px solid #3d3d3d;
                border-radius: 5px;
            }
        """)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)
        header_layout.setSpacing(8)
        
        self._label_widget = QLabel(label)
        self._label_widget.setStyleSheet("color: #e8e8e8; font-weight: 600; font-size: 11px; background: transparent;")
        header_layout.addWidget(self._label_widget)
        
        self._res_badge = QLabel("")
        self._res_badge.setStyleSheet(
            "color: #bbb; font-size: 10px; background: #4a4a4a; "
            "padding: 3px 8px; border-radius: 4px; font-weight: 500;"
        )
        self._res_badge.setVisible(False)
        header_layout.addWidget(self._res_badge)
        
        self._context_badge = QLabel("")
        self._context_badge.setStyleSheet(
            "color: #8cf; font-size: 9px; background: #2a3a4a; "
            "padding: 3px 6px; border-radius: 4px; font-weight: 500;"
        )
        self._context_badge.setVisible(False)
        header_layout.addWidget(self._context_badge)
        
        header_layout.addStretch()
        
        self._reset_btn = QPushButton("Fit")
        self._reset_btn.setToolTip("Fit to Window")
        self._reset_btn.setFixedHeight(24)
        self._reset_btn.setStyleSheet("""
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4a4a, stop:1 #404040);
                border: 1px solid #555; border-radius: 4px; color: #ccc; font-size: 10px; padding: 2px 8px;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #555, stop:1 #4a4a4a); color: #fff; }
            QPushButton:pressed { background: #383838; }
        """)
        header_layout.addWidget(self._reset_btn)
        
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setToolTip("Clear Image")
        self._clear_btn.setFixedHeight(24)
        self._clear_btn.setStyleSheet("""
            QPushButton { 
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a4040, stop:1 #403535);
                border: 1px solid #5a4545; border-radius: 4px; color: #daa; font-size: 10px; padding: 2px 8px;
            }
            QPushButton:hover { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #5a4a4a, stop:1 #4a4040); color: #fcc; }
            QPushButton:pressed { background: #3a3030; }
        """)
        self._clear_btn.setVisible(False)
        self._clear_btn.clicked.connect(self.clearClicked.emit)
        self._reset_btn.clicked.connect(self._on_reset_view)
        header_layout.addWidget(self._clear_btn)
        
        layout.addWidget(header)
        
        self._viewer = ImageViewer(block_size=block_size, label=label)
        self._viewer.blockClicked.connect(self.blockClicked.emit)
        self._viewer.viewChanged.connect(self.viewChanged.emit)
        self._viewer.imageDropped.connect(self.imageDropped.emit)
        layout.addWidget(self._viewer)
    
    def viewer(self) -> ImageViewer:
        return self._viewer
    
    def set_image(self, image: np.ndarray, animate: bool = True):
        self._viewer.set_image(image, animate)
        h, w = image.shape[:2]
        self._res_badge.setText(f"{w}×{h}")
        self._res_badge.setVisible(True)
        self._clear_btn.setVisible(True)
    
    def clear_image(self):
        self._viewer.clear_image()
        self._res_badge.setVisible(False)
        self._context_badge.setVisible(False)
        self._clear_btn.setVisible(False)
    
    def swap_pixmap(self, image: np.ndarray):
        self._viewer.swap_pixmap(image)
    
    def set_block_size(self, block_size: int):
        self._viewer.set_block_size(block_size)
    
    def set_selected_block(self, row: int, col: int):
        self._viewer.set_selected_block(row, col)
    
    def sync_view(self, other: 'ImageViewerWithControls'):
        self._viewer.sync_view(other._viewer)
    
    def has_image(self) -> bool:
        return self._viewer.has_image()
    
    def get_image_dimensions(self) -> tuple[int, int] | None:
        return self._viewer.get_image_dimensions()
    
    def set_label(self, label: str):
        self._label = label
        self._label_widget.setText(label)
    
    def update_resolution_badge(self, width: int, height: int):
        self._res_badge.setText(f"{width}×{height}")
        self._res_badge.setVisible(True)
    
    def set_context_info(self, info: str):
        if info:
            self._context_badge.setText(info)
            self._context_badge.setVisible(True)
        else:
            self._context_badge.setVisible(False)
    
    def _on_reset_view(self):
        self._viewer.reset_view()
    
    def _is_valid_image_drop(self, mime_data) -> str | None:
        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                path = urls[0].toLocalFile()
                if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    return path
        return None
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if self._is_valid_image_drop(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event: QDragMoveEvent):
        if self._is_valid_image_drop(event.mimeData()):
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        path = self._is_valid_image_drop(event.mimeData())
        if path:
            self.imageDropped.emit(path)
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

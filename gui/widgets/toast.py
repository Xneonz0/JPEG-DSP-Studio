"""Toast notification widget."""

from PySide6.QtWidgets import QWidget, QLabel, QHBoxLayout, QGraphicsOpacityEffect
from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint, QParallelAnimationGroup


class ToastNotification(QWidget):
    """Toast that appears briefly with slide + fade animation."""
    
    _active_toasts = []
    
    def __init__(self, parent: QWidget, message: str, toast_type: str = "info", duration: int = 3000):
        super().__init__(parent)
        
        self._duration = duration
        self._toast_type = toast_type
        
        styles = {
            "success": {"bg": "#2d5a3d", "border": "#4ade80", "icon": "✓"},
            "error": {"bg": "#5a2d2d", "border": "#f87171", "icon": "✗"},
            "warning": {"bg": "#5a4a2d", "border": "#fbbf24", "icon": "⚠"},
            "info": {"bg": "#2d3d5a", "border": "#60a5fa", "icon": "ℹ"},
        }
        style = styles.get(toast_type, styles["info"])
        
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool | 
                           Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)
        
        self.setStyleSheet(f"""
            ToastNotification {{
                background-color: {style['bg']};
                border: 1px solid {style['border']};
                border-radius: 6px;
            }}
        """)
        
        icon_label = QLabel(style["icon"])
        icon_label.setStyleSheet(f"color: {style['border']}; font-size: 14px; font-weight: bold;")
        layout.addWidget(icon_label)
        
        msg_label = QLabel(message)
        msg_label.setStyleSheet("color: #e0e0e0; font-size: 12px;")
        msg_label.setWordWrap(True)
        msg_label.setMaximumWidth(300)
        layout.addWidget(msg_label)
        
        self._opacity = QGraphicsOpacityEffect(self)
        self._opacity.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity)
        
        self.adjustSize()
    
    def show_animated(self):
        ToastNotification._active_toasts.append(self)
        self._update_position()
        final_x = self.x()
        self.move(final_x + 100, self.y())
        self.show()
        
        anim_group = QParallelAnimationGroup(self)
        
        fade_in = QPropertyAnimation(self._opacity, b"opacity", self)
        fade_in.setDuration(250)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(0.95)
        fade_in.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim_group.addAnimation(fade_in)
        
        slide_in = QPropertyAnimation(self, b"pos", self)
        slide_in.setDuration(300)
        slide_in.setStartValue(self.pos())
        slide_in.setEndValue(QPoint(final_x, self.y()))
        slide_in.setEasingCurve(QEasingCurve.Type.OutBack)
        anim_group.addAnimation(slide_in)
        
        anim_group.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
        
        if self._duration > 0:
            QTimer.singleShot(self._duration, self._fade_out)
    
    def _fade_out(self):
        anim_group = QParallelAnimationGroup(self)
        
        fade_out = QPropertyAnimation(self._opacity, b"opacity", self)
        fade_out.setDuration(250)
        fade_out.setStartValue(0.95)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InCubic)
        anim_group.addAnimation(fade_out)
        
        slide_out = QPropertyAnimation(self, b"pos", self)
        slide_out.setDuration(250)
        slide_out.setStartValue(self.pos())
        slide_out.setEndValue(QPoint(self.x() + 50, self.y()))
        slide_out.setEasingCurve(QEasingCurve.Type.InBack)
        anim_group.addAnimation(slide_out)
        
        anim_group.finished.connect(self._cleanup)
        anim_group.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)
    
    def _cleanup(self):
        if self in ToastNotification._active_toasts:
            ToastNotification._active_toasts.remove(self)
        self.close()
        self.deleteLater()
        
        for toast in ToastNotification._active_toasts:
            toast._update_position()
    
    def _update_position(self):
        if not self.parent():
            return
        
        parent = self.parent()
        parent_rect = parent.rect()
        
        my_index = ToastNotification._active_toasts.index(self) if self in ToastNotification._active_toasts else 0
        y_offset = 20
        for i in range(my_index):
            y_offset += ToastNotification._active_toasts[i].height() + 8
        
        x = parent_rect.width() - self.width() - 20
        y = parent_rect.height() - self.height() - y_offset
        self.move(x, y)
    
    @classmethod
    def show_toast(cls, parent: QWidget, message: str, toast_type: str = "info", duration: int = 3000):
        toast = cls(parent, message, toast_type, duration)
        toast.show_animated()
        return toast

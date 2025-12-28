"""Styled ComboBox with proper dropdown rendering."""

from PySide6.QtWidgets import QComboBox, QStyledItemDelegate
from PySide6.QtCore import QSize


class ComboBoxItemDelegate(QStyledItemDelegate):
    """Delegate ensuring consistent item height in dropdown."""
    
    def __init__(self, item_height=26, parent=None):
        super().__init__(parent)
        self._item_height = item_height
    
    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        size.setHeight(self._item_height)
        return size


def style_combobox(combo: QComboBox):
    """Apply delegate for consistent item sizing without replacing view."""
    delegate = ComboBoxItemDelegate(item_height=26, parent=combo)
    combo.setItemDelegate(delegate)
    
    # Ensure minimum height for better click target
    combo.setMinimumHeight(28)


class StyledComboBox(QComboBox):
    """ComboBox with proper dropdown behavior."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        style_combobox(self)

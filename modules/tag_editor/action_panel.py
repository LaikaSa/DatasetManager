from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal

class ActionPanel(QWidget):
    remove_tags_requested = Signal(set)  # Signal to emit when remove tags is requested

    def __init__(self):
        super().__init__()
        self.tags_to_remove = set()  # Store tags marked for removal
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Remove selected tags button
        self.remove_tags_btn = QPushButton("Remove Selected Tags")
        self.remove_tags_btn.clicked.connect(self.on_remove_tags_clicked)
        layout.addWidget(self.remove_tags_btn)
        
        # Add stretch to push buttons to top
        layout.addStretch()

    def on_remove_tags_clicked(self):
        """Queue tags for removal"""
        # This will be connected to get currently selected tags
        pass

    def clear_pending_changes(self):
        """Clear any pending changes"""
        self.tags_to_remove.clear()
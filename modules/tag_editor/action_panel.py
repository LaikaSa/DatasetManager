from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal

class ActionTab(QWidget):
    remove_tags_requested = Signal(set)  # Signal to emit selected tags for removal

    def __init__(self, tag_list_ref):
        super().__init__()
        self.tag_list_ref = tag_list_ref  # Reference to TagList widget
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # Remove tags button
        self.remove_tags_btn = QPushButton("Remove Selected Tags")
        self.remove_tags_btn.clicked.connect(self.queue_tag_removal)
        layout.addWidget(self.remove_tags_btn)
        
        # Status label
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        
        layout.addStretch()

    def queue_tag_removal(self):
        """Queue selected tags for removal"""
        selected_tags = {tag for tag, cb in self.tag_list_ref.tag_checkboxes.items() 
                        if cb.isChecked()}
        
        if selected_tags:
            self.remove_tags_requested.emit(selected_tags)
            self.status_label.setText(f"Queued {len(selected_tags)} tags for removal")
            self.status_label.setStyleSheet("color: orange")
        else:
            self.status_label.setText("No tags selected")
            self.status_label.setStyleSheet("color: red")

    def clear_status(self):
        """Clear the status message"""
        self.status_label.clear()

    def on_remove_tags_clicked(self):
        """Queue tags for removal"""
        # This will be connected to get currently selected tags
        pass

    def clear_pending_changes(self):
        """Clear any pending changes"""
        self.tags_to_remove.clear()
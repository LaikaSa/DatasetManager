from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, QScrollArea, 
                              QCheckBox, QGridLayout, QLabel)
from PySide6.QtCore import Signal, Qt
from collections import Counter
import os

class FilterByTagsPanel(QWidget):
    filter_changed = Signal(set)

    def __init__(self, tag_manager):
        super().__init__()
        self.tag_manager = tag_manager
        self.tag_counters = Counter()
        self.tag_checkboxes = {}
        self.selected_tags = set()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Search input
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search tags (comma-separated)")
        self.search_input.textChanged.connect(self.on_search_changed)
        layout.addWidget(self.search_input)

        # Tags scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Container for checkboxes
        self.tags_container = QWidget()
        self.tags_layout = QGridLayout(self.tags_container)
        self.scroll_area.setWidget(self.tags_container)
        layout.addWidget(self.scroll_area)

    def refresh_tag_display(self, filter_text=""):
        """Refresh the tag checkboxes display"""
        # Clear existing checkboxes
        while self.tags_layout.count():
            item = self.tags_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.tag_checkboxes.clear()

        # Filter and sort tags by frequency
        filter_terms = [t.strip().lower() for t in filter_text.split(',') if t.strip()]
        filtered_tags = []
        
        for tag, count in self.tag_counters.most_common():
            if not filter_terms or any(term in tag.lower() for term in filter_terms):
                filtered_tags.append((tag, count))

        # Create checkboxes for filtered tags
        for idx, (tag, count) in enumerate(filtered_tags):
            row = idx // 2
            col = idx % 2

            checkbox = QCheckBox(f"{tag} ({count})")
            checkbox.setChecked(tag in self.selected_tags)
            checkbox.stateChanged.connect(lambda state, t=tag: self.on_tag_toggled(t, state))
            
            self.tag_checkboxes[tag] = checkbox
            self.tags_layout.addWidget(checkbox, row, col)

    def on_search_changed(self, text):
        """Handle search input changes"""
        self.refresh_tag_display(text)

    def on_tag_toggled(self, tag, state):
        if state == Qt.CheckState.Checked.value:
            self.selected_tags.add(tag)
        else:
            self.selected_tags.discard(tag)
        
        # Emit selected tags for filtering
        self.filter_changed.emit(self.selected_tags)

    def update_tags(self, image_paths):
        self.tag_counters.clear()
        self.selected_tags.clear()
        
        # Count tag frequencies
        for path in image_paths:
            tags = self.tag_manager.get_tags(path).split(',')
            tags = [tag.strip() for tag in tags if tag.strip() and tag.strip() != '1234']
            self.tag_counters.update(tags)

        self.refresh_tag_display()
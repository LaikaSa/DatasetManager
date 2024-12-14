from PySide6.QtWidgets import (QScrollArea, QWidget, QVBoxLayout, 
                              QLineEdit, QCheckBox, QGridLayout,
                              QPushButton, QHBoxLayout, QButtonGroup,
                              QLabel)
from PySide6.QtCore import Signal, Qt
from collections import Counter

class TagPanel(QScrollArea):
    tag_toggled = Signal(str, bool)
    logic_changed = Signal(str, str)  # (AND/OR, POSITIVE/NEGATIVE)

    def __init__(self):
        super().__init__()
        self.tag_checkboxes = {}
        self.COLUMNS = 3
        self.init_ui()

    def init_ui(self):
        self.setWidgetResizable(True)
        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        
        # Search box
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Search tags...")
        self.search_box.textChanged.connect(self.filter_tags)
        self.layout.addWidget(self.search_box)
        
        # Clear filter button
        self.clear_button = QPushButton("Clear Filters")
        self.clear_button.clicked.connect(self.clear_filters)
        self.layout.addWidget(self.clear_button)
        
        # Logic selection groups
        # AND/OR logic
        combine_logic_layout = QHBoxLayout()
        self.combine_logic_group = QButtonGroup()
        self.combine_logic_group.setExclusive(True)
        
        self.and_logic = QCheckBox("AND")
        self.and_logic.setChecked(True)
        self.combine_logic_group.addButton(self.and_logic)
        combine_logic_layout.addWidget(self.and_logic)
        
        self.or_logic = QCheckBox("OR")
        self.combine_logic_group.addButton(self.or_logic)
        combine_logic_layout.addWidget(self.or_logic)
        
        self.layout.addLayout(combine_logic_layout)

        # POSITIVE/NEGATIVE logic
        filter_logic_layout = QHBoxLayout()
        self.filter_logic_group = QButtonGroup()
        self.filter_logic_group.setExclusive(True)
        
        self.positive_logic = QCheckBox("POSITIVE")
        self.positive_logic.setChecked(True)
        self.filter_logic_group.addButton(self.positive_logic)
        filter_logic_layout.addWidget(self.positive_logic)
        
        self.negative_logic = QCheckBox("NEGATIVE")
        self.filter_logic_group.addButton(self.negative_logic)
        filter_logic_layout.addWidget(self.negative_logic)
        
        self.layout.addLayout(filter_logic_layout)

        # Image counter
        self.image_counter = QLabel("Showing: 0 images")
        self.layout.addWidget(self.image_counter)
        
        # Connect logic changes
        self.and_logic.toggled.connect(self.on_logic_changed)
        self.or_logic.toggled.connect(self.on_logic_changed)
        self.positive_logic.toggled.connect(self.on_logic_changed)
        self.negative_logic.toggled.connect(self.on_logic_changed)
        
        # Tags container
        self.tags_widget = QWidget()
        self.tags_layout = QGridLayout(self.tags_widget)
        self.tags_layout.setSpacing(5)
        self.layout.addWidget(self.tags_widget)
        self.setWidget(self.container)

    def update_image_counter(self, count, total):
        """Update the image counter display"""
        self.image_counter.setText(f"Showing: {count} / {total} images")

    def clear_filters(self):
        """Uncheck all tag checkboxes"""
        for checkbox in self.tag_checkboxes.values():
            checkbox.setChecked(False)

    def on_logic_changed(self, checked):
        """Emit current logic combination when changed"""
        if checked:
            combine_logic = "AND" if self.and_logic.isChecked() else "OR"
            filter_logic = "POSITIVE" if self.positive_logic.isChecked() else "NEGATIVE"
            self.logic_changed.emit(combine_logic, filter_logic)

    def update_tags(self, tag_counts):
        # Clear existing tags
        while self.tags_layout.count():
            self.tags_layout.takeAt(0).widget().deleteLater()
        self.tag_checkboxes.clear()

        # Add new tags in frequency order
        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
        for idx, (tag, count) in enumerate(sorted_tags):
            tag_text = tag.strip()  # Clean the tag
            checkbox = QCheckBox(f"{tag_text} ({count})")
            
            # Use a lambda with default argument to capture the correct tag
            checkbox.stateChanged.connect(
                lambda state, t=tag_text: 
                self.tag_toggled.emit(t, state == Qt.CheckState.Checked.value)
            )
            
            # Calculate row and column positions with fixed 3 columns
            row = idx // self.COLUMNS
            col = idx % self.COLUMNS
            
            self.tags_layout.addWidget(checkbox, row, col)
            self.tag_checkboxes[tag_text] = checkbox

    def filter_tags(self, text):
        visible_count = 0
        search_terms = text.lower().split(',')
        
        # First, determine visibility
        visible_checkboxes = []
        for tag, checkbox in self.tag_checkboxes.items():
            visible = not text or any(term.strip() in tag.lower() for term in search_terms)
            if visible:
                visible_checkboxes.append(checkbox)
                checkbox.setVisible(True)
            else:
                checkbox.setVisible(False)

        # Then, reposition visible checkboxes
        for idx, checkbox in enumerate(visible_checkboxes):
            row = idx // self.COLUMNS
            col = idx % self.COLUMNS
            # Need to remove and re-add to reposition
            self.tags_layout.removeWidget(checkbox)
            self.tags_layout.addWidget(checkbox, row, col)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Reflow tags when panel is resized
        visible_checkboxes = [cb for cb in self.tag_checkboxes.values() if cb.isVisible()]
        
        for idx, checkbox in enumerate(visible_checkboxes):
            row = idx // self.COLUMNS
            col = idx % self.COLUMNS
            self.tags_layout.removeWidget(checkbox)
            self.tags_layout.addWidget(checkbox, row, col)
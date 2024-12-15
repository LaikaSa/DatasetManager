from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, QCheckBox, 
                              QGridLayout, QPushButton, QHBoxLayout, 
                              QTabWidget, QScrollArea, QLabel, QGroupBox)
from PySide6.QtCore import Signal, Qt
from collections import Counter

class TagList(QScrollArea):
    tag_toggled = Signal(str, bool)

    def __init__(self):
        super().__init__()
        self.tag_checkboxes = {}
        self.init_ui()

    def init_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        self.grid = QGridLayout(container)
        self.grid.setSpacing(5)
        self.setWidget(container)

    def update_tags(self, tag_counts: Counter):
        # Clear existing tags
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.tag_checkboxes.clear()

        # Add new tags in frequency order
        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
        columns = 3
        
        for idx, (tag, count) in enumerate(sorted_tags):
            checkbox = QCheckBox(f"{tag} ({count})")
            checkbox.stateChanged.connect(lambda state, t=tag: 
                self.tag_toggled.emit(t, state == Qt.CheckState.Checked.value))
            
            row = idx // columns
            col = idx % columns
            self.grid.addWidget(checkbox, row, col)
            self.tag_checkboxes[tag] = checkbox

    def clear_all_checks(self):
        for checkbox in self.tag_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

    def filter_visible_tags(self, search_text):
        search_terms = search_text.lower().split(',')
        visible_boxes = []
        
        for tag, checkbox in self.tag_checkboxes.items():
            visible = not search_text or any(term.strip() in tag.lower() 
                                           for term in search_terms)
            checkbox.setVisible(visible)
            if visible:
                visible_boxes.append(checkbox)

        self.reflow_checkboxes(visible_boxes)

    def reflow_checkboxes(self, visible_boxes):
        columns = 3
        for idx, checkbox in enumerate(visible_boxes):
            row = idx // columns
            col = idx % columns
            self.grid.removeWidget(checkbox)
            self.grid.addWidget(checkbox, row, col)

class FilterTab(QWidget):
    filter_changed = Signal(set, str, str)

    def __init__(self, tag_list):
        super().__init__()
        self.selected_tags = set()
        self.tag_list = tag_list
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Search box
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search tags...")
        self.search_input.textChanged.connect(self.filter_tag_list)
        layout.addWidget(self.search_input)

        # Clear button
        self.clear_btn = QPushButton("Clear Filters")
        self.clear_btn.clicked.connect(self.clear_filters)
        layout.addWidget(self.clear_btn)

        # Combine Logic (AND/OR)
        combine_group = QGroupBox("Combine Logic")
        combine_layout = QHBoxLayout()
        
        self.and_logic = QCheckBox("AND")
        self.or_logic = QCheckBox("OR")
        self.and_logic.setChecked(True)
        
        self.and_logic.toggled.connect(lambda checked: checked and self.or_logic.setChecked(False))
        self.or_logic.toggled.connect(lambda checked: checked and self.and_logic.setChecked(False))
        
        combine_layout.addWidget(self.and_logic)
        combine_layout.addWidget(self.or_logic)
        combine_group.setLayout(combine_layout)
        layout.addWidget(combine_group)

        # Filter Logic (POSITIVE/NEGATIVE)
        filter_group = QGroupBox("Filter Logic")
        filter_layout = QHBoxLayout()
        
        self.positive_logic = QCheckBox("POSITIVE")
        self.negative_logic = QCheckBox("NEGATIVE")
        self.positive_logic.setChecked(True)
        
        self.positive_logic.toggled.connect(lambda checked: checked and self.negative_logic.setChecked(False))
        self.negative_logic.toggled.connect(lambda checked: checked and self.positive_logic.setChecked(False))
        
        filter_layout.addWidget(self.positive_logic)
        filter_layout.addWidget(self.negative_logic)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Image counter
        self.counter_label = QLabel("Showing: 0/0 images")
        layout.addWidget(self.counter_label)

        layout.addStretch()

        # Connect logic changes
        for checkbox in [self.and_logic, self.or_logic, 
                        self.positive_logic, self.negative_logic]:
            checkbox.toggled.connect(self.emit_filter_change)

    def clear_filters(self):
        self.selected_tags.clear()
        self.tag_list.clear_all_checks()
        self.emit_filter_change()

    def emit_filter_change(self):
        combine_logic = "AND" if self.and_logic.isChecked() else "OR"
        filter_logic = "POSITIVE" if self.positive_logic.isChecked() else "NEGATIVE"
        self.filter_changed.emit(self.selected_tags, combine_logic, filter_logic)

    def filter_tag_list(self, text):
        self.tag_list.filter_visible_tags(text)

    def update_counter(self, visible, total):
        self.counter_label.setText(f"Showing: {visible}/{total} images")

class ActionTab(QWidget):
    remove_tags_requested = Signal(set)

    def __init__(self, tag_list):
        super().__init__()
        self.tag_list = tag_list
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
        selected_tags = {tag for tag, cb in self.tag_list.tag_checkboxes.items() 
                        if cb.isChecked()}
        
        if selected_tags:
            self.remove_tags_requested.emit(selected_tags)
            self.status_label.setText(f"Queued {len(selected_tags)} tags for removal")
            self.status_label.setStyleSheet("color: orange")
        else:
            self.status_label.setText("No tags selected")
            self.status_label.setStyleSheet("color: red")

    def clear_status(self):
        self.status_label.clear()

class TagPanel(QWidget):
    filter_changed = Signal(set, str, str)
    tags_removal_queued = Signal(set)
    
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Create tag list first
        self.tag_list = TagList()
        
        # Create tabs
        self.tab_widget = QTabWidget()
        
        # Filter tab
        self.filter_tab = FilterTab(self.tag_list)
        self.filter_tab.filter_changed.connect(self.filter_changed.emit)
        
        # Action tab
        self.action_tab = ActionTab(self.tag_list)
        self.action_tab.remove_tags_requested.connect(self.on_remove_tags_requested)
        
        self.tab_widget.addTab(self.filter_tab, "FILTER")
        self.tab_widget.addTab(self.action_tab, "ACTIONS")
        
        # Add tabs and tag list to layout
        layout.addWidget(self.tab_widget)
        layout.addWidget(self.tag_list)

        # Connect search box to tag list
        self.filter_tab.search_input.textChanged.connect(self.tag_list.filter_visible_tags)

    def update_tags(self, tag_counts):
        self.tag_list.update_tags(tag_counts)

    def update_counter(self, visible, total):
        self.filter_tab.update_counter(visible, total)

    def on_remove_tags_requested(self, tags):
        self.tags_removal_queued.emit(tags)

    def clear_pending_status(self):
        self.action_tab.clear_status()
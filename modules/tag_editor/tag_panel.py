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
        self.selected_tags = set()  # Keep track of selected tags
        self.init_ui()

    def init_ui(self):
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        container = QWidget()
        self.grid = QGridLayout(container)
        self.grid.setSpacing(5)
        self.setWidget(container)

    def on_tag_toggled(self, tag: str, checked: bool):
        if checked:
            self.selected_tags.add(tag)
        else:
            self.selected_tags.discard(tag)
        self.tag_toggled.emit(tag, checked)

    def update_tags(self, tag_counts: Counter):
        # Remember which tags were checked
        checked_tags = {tag for tag, cb in self.tag_checkboxes.items() 
                       if cb.isChecked()}
        
        # Clear existing tags
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.tag_checkboxes.clear()
        self.selected_tags.clear()  # Clear selected tags

        # Add new tags in frequency order
        sorted_tags = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))
        columns = 3
        
        for idx, (tag, count) in enumerate(sorted_tags):
            checkbox = QCheckBox(f"{tag} ({count})")
            checkbox.setChecked(tag in checked_tags)
            checkbox.stateChanged.connect(
                lambda state, t=tag: self.on_tag_toggled(t, bool(state))
            )
            
            row = idx // columns
            col = idx % columns
            self.grid.addWidget(checkbox, row, col)
            self.tag_checkboxes[tag] = checkbox

    def clear_all_checks(self):
        self.selected_tags.clear()
        for checkbox in self.tag_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

    def get_checked_tags(self) -> set[str]:
        return {tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()}

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

    def clear_all_checks(self):
        self.selected_tags.clear()
        for checkbox in self.tag_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

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
        self.tag_list = tag_list
        self.selected_tags = set()
        self.init_ui()
        
        # Connect tag list signals
        self.tag_list.tag_toggled.connect(self.on_tag_toggled)

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

        # Logic groups
        combine_group = QGroupBox("Combine Logic")
        combine_layout = QHBoxLayout()
        self.and_logic = QCheckBox("AND")
        self.or_logic = QCheckBox("OR")
        self.and_logic.setChecked(True)
        combine_layout.addWidget(self.and_logic)
        combine_layout.addWidget(self.or_logic)
        combine_group.setLayout(combine_layout)
        layout.addWidget(combine_group)

        filter_group = QGroupBox("Filter Logic")
        filter_layout = QHBoxLayout()
        self.positive_logic = QCheckBox("POSITIVE")
        self.negative_logic = QCheckBox("NEGATIVE")
        self.positive_logic.setChecked(True)
        filter_layout.addWidget(self.positive_logic)
        filter_layout.addWidget(self.negative_logic)
        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        # Counter
        self.counter_label = QLabel("Showing: 0/0 images")
        layout.addWidget(self.counter_label)

        layout.addStretch()

        # Connect logic changes
        self.and_logic.toggled.connect(self.on_logic_changed)
        self.or_logic.toggled.connect(self.on_logic_changed)
        self.positive_logic.toggled.connect(self.on_logic_changed)
        self.negative_logic.toggled.connect(self.on_logic_changed)

        # Make checkboxes mutually exclusive within groups
        self.and_logic.toggled.connect(lambda checked: checked and self.or_logic.setChecked(False))
        self.or_logic.toggled.connect(lambda checked: checked and self.and_logic.setChecked(False))
        self.positive_logic.toggled.connect(lambda checked: checked and self.negative_logic.setChecked(False))
        self.negative_logic.toggled.connect(lambda checked: checked and self.positive_logic.setChecked(False))

    def clear_filters(self):
        self.selected_tags.clear()
        self.tag_list.clear_all_checks()
        self.emit_filter_change()

    def on_logic_changed(self, checked):
        """Handle logic checkbox changes"""
        if checked:
            self.emit_filter_change()

    def on_tag_toggled(self, tag: str, checked: bool):
        print(f"FilterTab received tag toggle: {tag}, {checked}")
        if checked:
            self.selected_tags.add(tag)
        else:
            self.selected_tags.discard(tag)
        self.emit_filter_change()

    def emit_filter_change(self):
        combine_logic = "AND" if self.and_logic.isChecked() else "OR"
        filter_logic = "POSITIVE" if self.positive_logic.isChecked() else "NEGATIVE"
        print(f"Emitting filter: {len(self.selected_tags)} tags, {combine_logic}, {filter_logic}")
        print(f"Selected tags: {self.selected_tags}")  # Debug print
        self.filter_changed.emit(self.selected_tags, combine_logic, filter_logic)

    def filter_tag_list(self, text):
        search_terms = text.lower().split(',')
        visible_boxes = []
        
        for tag, checkbox in self.tag_list.tag_checkboxes.items():
            visible = not text or any(term.strip() in tag.lower() 
                                    for term in search_terms)
            checkbox.setVisible(visible)
            if visible:
                visible_boxes.append(checkbox)

        # Reflow visible checkboxes
        columns = 3
        for idx, checkbox in enumerate(visible_boxes):
            row = idx // columns
            col = idx % columns
            self.grid.removeWidget(checkbox)
            self.grid.addWidget(checkbox, row, col)

    def update_counter(self, visible: int, total: int):
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
        self.remove_btn = QPushButton("Remove Selected Tags")
        self.remove_btn.clicked.connect(self.on_remove_clicked)
        layout.addWidget(self.remove_btn)
        
        # Status label
        self.status_label = QLabel()
        layout.addWidget(self.status_label)
        
        layout.addStretch()

    def on_remove_clicked(self):
        tags = self.tag_list.get_checked_tags()
        if tags:
            self.remove_tags_requested.emit(tags)
            self.status_label.setText(f"Queued {len(tags)} tags for removal")
            self.status_label.setStyleSheet("color: orange")
        else:
            self.status_label.setText("No tags selected")
            self.status_label.setStyleSheet("color: red")

    def clear_status(self):
        self.status_label.clear()

class TagPanel(QWidget):
    filter_changed = Signal(set, str, str)
    tags_removal_requested = Signal(set)

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
        # Connect filter tab signals to our own signals
        self.filter_tab.filter_changed.connect(self.on_filter_changed)
        
        # Action tab
        self.action_tab = ActionTab(self.tag_list)
        self.action_tab.remove_tags_requested.connect(self.tags_removal_requested.emit)
        
        self.tab_widget.addTab(self.filter_tab, "FILTER")
        self.tab_widget.addTab(self.action_tab, "ACTIONS")
        
        layout.addWidget(self.tab_widget)
        layout.addWidget(self.tag_list)

    def on_filter_changed(self, tags: set, combine_logic: str, filter_logic: str):
        print(f"TagPanel forwarding filter: {len(tags)} tags")  # Debug print
        self.filter_changed.emit(tags, combine_logic, filter_logic)

    def update_tags(self, tag_counts: Counter):
        self.tag_list.update_tags(tag_counts)

    def update_counter(self, visible: int, total: int):
        self.filter_tab.update_counter(visible, total)

    def get_current_filter(self):
        """Get current filter settings"""
        tags = self.tag_list.get_checked_tags()
        combine_logic = "AND" if self.filter_tab.and_logic.isChecked() else "OR"
        filter_logic = "POSITIVE" if self.filter_tab.positive_logic.isChecked() else "NEGATIVE"
        return tags, combine_logic, filter_logic
    
    def clear_all_selections(self):
        """Clear all tag selections and reset filter state"""
        self.tag_list.clear_all_checks()
        self.filter_tab.selected_tags.clear()
        self.filter_tab.emit_filter_change()

    def clear_status(self):
        self.action_tab.clear_status()

    def clear(self):
        self.tag_list.clear_all_checks()
        self.action_tab.clear_status()
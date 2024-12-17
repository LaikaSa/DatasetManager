from PySide6.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, QCheckBox, 
                              QGridLayout, QPushButton, QHBoxLayout, QFileDialog, 
                              QTabWidget, QScrollArea, QLabel, QGroupBox,
                              QTextEdit, QMessageBox)
from PySide6.QtCore import Signal, Qt
from collections import Counter
import os

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
        """Clear all checkboxes and selected tags"""
        self.selected_tags.clear()
        for checkbox in self.tag_checkboxes.values():
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)

    def clear(self):
        """Clear all tags from the list"""
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.tag_checkboxes.clear()
        self.selected_tags.clear()

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

        # Search box and Clear button remain the same
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search tags...")
        self.search_input.textChanged.connect(self.filter_tag_list)
        layout.addWidget(self.search_input)

        self.clear_btn = QPushButton("Clear Filters")
        self.clear_btn.clicked.connect(self.clear_filters)
        layout.addWidget(self.clear_btn)

        # Logic groups - Modified implementation
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

        # Connect logic changes - Modified connections
        self.and_logic.toggled.connect(self.on_combine_logic_changed)
        self.or_logic.toggled.connect(self.on_combine_logic_changed)
        self.positive_logic.toggled.connect(self.on_filter_logic_changed)
        self.negative_logic.toggled.connect(self.on_filter_logic_changed)

    def on_combine_logic_changed(self, checked):
        """Handle changes in AND/OR logic"""
        if self.sender() == self.and_logic and checked:
            self.or_logic.setChecked(False)
        elif self.sender() == self.or_logic and checked:
            self.and_logic.setChecked(False)
        self.emit_filter_change()

    def on_filter_logic_changed(self, checked):
        """Handle changes in POSITIVE/NEGATIVE logic"""
        if self.sender() == self.positive_logic and checked:
            self.negative_logic.setChecked(False)
        elif self.sender() == self.negative_logic and checked:
            self.positive_logic.setChecked(False)
        self.emit_filter_change()

    def clear_filters(self):
        self.selected_tags.clear()
        self.tag_list.clear_all_checks()
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
        print(f"Emitting filter: {len(self.selected_tags)} tags")
        print(f"Selected tags: {self.selected_tags}")
        print(f"Combine logic: {combine_logic}")
        print(f"Filter logic: {filter_logic}")
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

    def clear_counter(self):
        """Clear the image counter"""
        self.counter_label.setText("Showing: 0/0 images")

    def clear(self):
        """Clear all filters and counters"""
        self.selected_tags.clear()
        self.counter_label.setText("Showing: 0/0 images")
        self.search_input.clear()

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

class EditCaptionTab(QWidget):
    caption_changed = Signal(str, str)  # (image_path, new_caption)
    
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Caption edit box
        self.caption_label = QLabel("No image selected")
        layout.addWidget(self.caption_label)
        
        self.caption_edit = QTextEdit()
        self.caption_edit.setPlaceholderText("Image tags will appear here when an image is selected...")
        self.caption_edit.textChanged.connect(self.on_caption_changed)
        layout.addWidget(self.caption_edit)

        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def set_caption(self, image_path: str, caption: str):
        """Load caption for selected image"""
        self.current_image = image_path
        self.caption_edit.blockSignals(True)
        self.caption_edit.setText(caption)
        self.caption_edit.blockSignals(False)
        self.caption_label.setText(f"Editing: {os.path.basename(image_path)}")
        self.status_label.clear()

    def on_caption_changed(self):
        """Handle caption changes"""
        if self.current_image:
            self.status_label.setText("Changes pending...")
            self.status_label.setStyleSheet("color: orange;")
            self.caption_changed.emit(self.current_image, self.caption_edit.toPlainText())

    def clear(self):
        """Clear current caption"""
        self.current_image = None
        self.caption_edit.clear()
        self.caption_label.setText("No image selected")
        self.status_label.clear()

class TagPanel(QWidget):
    filter_changed = Signal(set, str, str)
    tags_removal_requested = Signal(set)
    caption_changed = Signal(str, str)
    delete_requested = Signal(bool, bool)
    move_requested = Signal(str, bool, bool)

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
        self.filter_tab.filter_changed.connect(self.on_filter_changed)
        
        # Action tab
        self.action_tab = ActionTab(self.tag_list)
        self.action_tab.remove_tags_requested.connect(self.tags_removal_requested.emit)
        
        # Edit Caption tab
        self.caption_tab = EditCaptionTab()
        self.caption_tab.caption_changed.connect(self.caption_changed.emit)
        
        # Delete/Move tab
        self.delete_move_tab = DeleteMoveTab()
        self.delete_move_tab.delete_requested.connect(self.delete_requested.emit)
        self.delete_move_tab.move_requested.connect(self.move_requested.emit)
        
        # Add all tabs
        self.tab_widget.addTab(self.filter_tab, "FILTER")
        self.tab_widget.addTab(self.action_tab, "ACTIONS")
        self.tab_widget.addTab(self.caption_tab, "EDIT CAPTION")
        self.tab_widget.addTab(self.delete_move_tab, "DELETE/MOVE")
        
        # Add components to layout
        layout.addWidget(self.tab_widget)
        
        # Add tag list (only visible for Filter and Actions tabs)
        self.tag_list_container = QWidget()
        tag_list_layout = QVBoxLayout(self.tag_list_container)
        tag_list_layout.addWidget(self.tag_list)
        layout.addWidget(self.tag_list_container)
        
        # Connect tab change signal
        self.tab_widget.currentChanged.connect(self.on_tab_changed)

    def on_tab_changed(self, index):
        """Show/hide tag list based on selected tab"""
        self.tag_list_container.setVisible(
            self.tab_widget.tabText(index) != "EDIT CAPTION"
        )

    def on_filter_changed(self, tags: set, combine_logic: str, filter_logic: str):
        print(f"TagPanel forwarding filter: {len(tags)} tags")
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

    def clear_status(self):
        self.action_tab.clear_status()

    def clear(self):
        """Clear all data in the panel"""
        self.tag_list.clear_all_checks()
        self.action_tab.clear_status()
        self.caption_tab.clear()
        self.update_tags(Counter())  # Clear tags
        self.filter_tab.clear_counter()  # Clear the counter

    def set_caption(self, image_path: str, caption: str):
        """Set caption for editing"""
        self.caption_tab.set_caption(image_path, caption)
        # Switch to caption tab
        self.tab_widget.setCurrentWidget(self.caption_tab)

class DeleteMoveTab(QWidget):
    delete_requested = Signal(bool, bool)  # (move_images, move_captions)
    move_requested = Signal(str, bool, bool)  # (destination, move_images, move_captions)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Delete section
        delete_group = QGroupBox("Delete")
        delete_layout = QVBoxLayout()
        
        self.delete_btn = QPushButton("Delete Displayed")
        self.delete_btn.clicked.connect(self.on_delete_clicked)
        delete_layout.addWidget(self.delete_btn)
        
        delete_group.setLayout(delete_layout)
        layout.addWidget(delete_group)

        # Move section
        move_group = QGroupBox("Move")
        move_layout = QVBoxLayout()
        
        # Destination input
        dest_layout = QHBoxLayout()
        self.dest_input = QLineEdit()
        self.dest_input.setPlaceholderText("Destination folder path...")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_destination)
        dest_layout.addWidget(self.dest_input)
        dest_layout.addWidget(self.browse_btn)
        move_layout.addLayout(dest_layout)
        
        # Move button
        self.move_btn = QPushButton("Move Displayed")
        self.move_btn.clicked.connect(self.on_move_clicked)
        move_layout.addWidget(self.move_btn)
        
        move_group.setLayout(move_layout)
        layout.addWidget(move_group)

        # File type checkboxes
        self.file_group = QGroupBox("Files to Process")
        file_layout = QHBoxLayout()
        
        self.image_cb = QCheckBox("Images")
        self.caption_cb = QCheckBox("Caption Files")
        self.image_cb.setChecked(True)  # Default checked
        self.caption_cb.setChecked(True)  # Default checked
        
        file_layout.addWidget(self.image_cb)
        file_layout.addWidget(self.caption_cb)
        
        self.file_group.setLayout(file_layout)
        layout.addWidget(self.file_group)

        # Status label
        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        layout.addStretch()

    def browse_destination(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Destination Folder")
        if folder:
            self.dest_input.setText(folder)

    def on_delete_clicked(self):
        if not (self.image_cb.isChecked() or self.caption_cb.isChecked()):
            self.status_label.setText("Please select at least one file type")
            self.status_label.setStyleSheet("color: red;")
            return

        reply = QMessageBox.question(
            self, 
            'Confirm Delete',
            'Are you sure you want to move the displayed files to recycle bin?',  # Updated message
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.delete_requested.emit(
                self.image_cb.isChecked(),
                self.caption_cb.isChecked()
            )

    def on_move_clicked(self):
        destination = self.dest_input.text()
        if not destination:
            self.status_label.setText("Please specify destination folder")
            self.status_label.setStyleSheet("color: red;")
            return

        if not (self.image_cb.isChecked() or self.caption_cb.isChecked()):
            self.status_label.setText("Please select at least one file type")
            self.status_label.setStyleSheet("color: red;")
            return

        if not os.path.exists(destination):
            try:
                os.makedirs(destination)
            except Exception as e:
                self.status_label.setText(f"Error creating destination folder: {e}")
                self.status_label.setStyleSheet("color: red;")
                return

        self.move_requested.emit(
            destination,
            self.image_cb.isChecked(),
            self.caption_cb.isChecked()
        )

    def clear_status(self):
        self.status_label.clear()
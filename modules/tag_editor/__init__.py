from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                              QPushButton, QFileDialog)
from PySide6.QtCore import Qt
from .gallery import GalleryGrid
from .tag_panel import TagPanel
from . import image_handler
from . import tag_handler
import os

class TagEditorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.images = []
        self.selected_tags = set()
        self.combine_logic = "AND"  # AND/OR
        self.filter_logic = "POSITIVE"  # POSITIVE/NEGATIVE
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Top controls
        top_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter folder path...")
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_folder)
        top_layout.addWidget(self.path_input)
        top_layout.addWidget(self.browse_btn)
        layout.addLayout(top_layout)

        # Load/Unload buttons
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load")
        self.unload_btn = QPushButton("Unload")
        self.load_btn.clicked.connect(self.load_folder)
        self.unload_btn.clicked.connect(self.unload_folder)
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.unload_btn)
        layout.addLayout(btn_layout)

        # Split view
        split_layout = QHBoxLayout()
        
        # Gallery
        self.gallery = GalleryGrid()
        split_layout.addWidget(self.gallery, 1)
        
        # Tag panel
        self.tag_panel = TagPanel()
        self.tag_panel.tag_toggled.connect(self.on_tag_toggled)
        split_layout.addWidget(self.tag_panel, 1)
        
        layout.addLayout(split_layout)

        # Connect the new logic signal
        self.tag_panel.logic_changed.connect(self.on_logic_changed)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.path_input.setText(folder)

    def load_folder(self):
        folder = self.path_input.text()
        if not folder or not os.path.exists(folder):
            return

        # Load images and tags
        self.images, tags_list = image_handler.load_from_directory(folder)
        
        # Update tag panel
        tag_counts = tag_handler.count_tags(tags_list)
        self.tag_panel.update_tags(tag_counts)
        
        # Display images and update counter
        self.update_gallery()
        self.tag_panel.update_image_counter(len(self.images), len(self.images))

    def unload_folder(self):
        self.images = []
        self.selected_tags.clear()
        self.update_gallery()
        self.tag_panel.update_tags({})
        self.tag_panel.update_image_counter(0, 0)

    def on_tag_toggled(self, tag, checked):
        if checked:
            self.selected_tags.add(tag.lower())
        else:
            self.selected_tags.discard(tag.lower())
        self.update_gallery()

    def on_logic_changed(self, combine_logic, filter_logic):
        """Handle change in tag logic"""
        self.combine_logic = combine_logic
        self.filter_logic = filter_logic
        self.update_gallery()

    def update_gallery(self):
        """Update gallery with filtered images"""
        if not self.selected_tags:
            visible_thumbnails = self.images
            print(f"No tags selected, showing all {len(self.images)} images")
        else:
            required_tags = {tag.lower() for tag in self.selected_tags}
            print(f"Filtering with tags: {required_tags}")
            print(f"Using logic: {self.combine_logic} and {self.filter_logic}")
            
            # First apply AND/OR logic
            if self.combine_logic == "AND":
                matching_thumbnails = [
                    thumb for thumb in self.images
                    if thumb.has_all_tags(required_tags)
                ]
                print(f"AND logic found {len(matching_thumbnails)} matches")
            else:  # OR logic
                matching_thumbnails = [
                    thumb for thumb in self.images
                    if thumb.has_any_tags(required_tags)
                ]
                print(f"OR logic found {len(matching_thumbnails)} matches")
            
            # Then apply POSITIVE/NEGATIVE logic
            if self.filter_logic == "POSITIVE":
                visible_thumbnails = matching_thumbnails
            else:  # NEGATIVE logic
                visible_thumbnails = [
                    thumb for thumb in self.images
                    if thumb not in matching_thumbnails
                ]
            
            print(f"Final visible count: {len(visible_thumbnails)}")
        
        # Update gallery and counter
        self.gallery.display_images(visible_thumbnails)
        self.tag_panel.update_image_counter(len(visible_thumbnails), len(self.images))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_gallery()  # Reflow gallery when window is resized
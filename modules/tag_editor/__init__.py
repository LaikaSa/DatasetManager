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
        
        # Display images
        self.update_gallery()

    def unload_folder(self):
        self.images = []
        self.selected_tags.clear()
        self.update_gallery()
        self.tag_panel.update_tags({})

    def on_tag_toggled(self, tag, checked):
        if checked:
            self.selected_tags.add(tag.lower())
        else:
            self.selected_tags.discard(tag.lower())
        self.update_gallery()

    def update_gallery(self):
        """Update gallery with filtered images"""
        if not self.selected_tags:
            visible_thumbnails = self.images
        else:
            # Convert selected tags to set once for comparison
            required_tags = {tag.lower() for tag in self.selected_tags}
            # Filter images
            visible_thumbnails = [
                thumb for thumb in self.images
                if thumb.has_all_tags(required_tags)
            ]
        
        # Update gallery with filtered thumbnails
        self.gallery.display_images(visible_thumbnails)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_gallery()  # Reflow gallery when window is resized
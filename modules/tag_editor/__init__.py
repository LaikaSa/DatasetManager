from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                              QPushButton, QFileDialog)
from PySide6.QtCore import Qt
from .gallery_view import GalleryView
from .tag_panel import TagPanel
from .image_model import ImageModel
import os

class TagEditorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.image_model = ImageModel()
        self.pending_changes = []
        self.tags_to_remove = set()  # Track tags pending removal
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Top controls
        top_layout = QHBoxLayout()
        
        # Path input
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter folder path...")
        
        # Buttons
        self.browse_btn = QPushButton("Browse")
        self.load_btn = QPushButton("Load")
        self.unload_btn = QPushButton("Unload")
        self.save_btn = QPushButton("Save All Changes")
        self.save_btn.setEnabled(False)
        
        top_layout.addWidget(self.path_input)
        top_layout.addWidget(self.browse_btn)
        top_layout.addWidget(self.load_btn)
        top_layout.addWidget(self.unload_btn)
        top_layout.addWidget(self.save_btn)
        
        layout.addLayout(top_layout)

        # Split view
        split_layout = QHBoxLayout()
        
        # Gallery view
        self.gallery = GalleryView()
        split_layout.addWidget(self.gallery, 1)
        
        # Tag panel
        self.tag_panel = TagPanel()
        split_layout.addWidget(self.tag_panel, 1)
        
        layout.addLayout(split_layout)

        # Connect signals
        self.browse_btn.clicked.connect(self.browse_folder)
        self.load_btn.clicked.connect(self.load_folder)
        self.unload_btn.clicked.connect(self.unload_folder)
        self.save_btn.clicked.connect(self.save_changes)
        self.tag_panel.filter_changed.connect(self.apply_filters)
        self.tag_panel.tags_removal_queued.connect(self.queue_tag_removal)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.path_input.setText(folder)

    def load_folder(self):
        folder = self.path_input.text()
        if not folder or not os.path.exists(folder):
            return

        # Load images and their tags
        self.image_model.load_directory(folder)
        
        # Update gallery
        self.gallery.display_images(self.image_model.images)
        
        # Update tag panel with tag frequencies
        tag_counts = self.image_model.get_tag_frequencies()
        self.tag_panel.update_tags(tag_counts)
        
        # Update counter
        self.tag_panel.update_counter(len(self.image_model.images), 
                                    len(self.image_model.images))

    def unload_folder(self):
        self.image_model.clear()
        self.gallery.clear()
        self.tag_panel.update_tags({})
        self.tag_panel.update_counter(0, 0)
        self.tags_to_remove.clear()
        self.tag_panel.clear_pending_status()
        self.save_btn.setEnabled(False)

    def apply_filters(self, tags: set, combine_logic: str, filter_logic: str):
        # Filter images based on tags and logic
        filtered_images = self.image_model.filter_images(
            tags, combine_logic, filter_logic
        )
        
        # Update gallery
        self.gallery.display_images(filtered_images)
        
        # Update counter
        self.tag_panel.update_counter(len(filtered_images), 
                                    len(self.image_model.images))

    def queue_tag_removal(self, tags):
        """Queue tags for removal"""
        self.tags_to_remove.update(tags)
        self.save_btn.setEnabled(True)

    def save_changes(self):
        """Execute all pending changes"""
        if self.tags_to_remove:
            # Create the actual removal function
            def remove_tags():
                for image_data in self.image_model.images:
                    if image_data.tags & self.tags_to_remove:
                        # Get the tag file path
                        tag_path = os.path.splitext(image_data.path)[0] + '.txt'
                        if os.path.exists(tag_path):
                            # Read current tags
                            with open(tag_path, 'r', encoding='utf-8') as f:
                                current_tags = {tag.strip() for tag in f.read().split(',') if tag.strip()}
                            
                            # Remove the specified tags
                            updated_tags = current_tags - self.tags_to_remove
                            
                            # Write back the remaining tags
                            with open(tag_path, 'w', encoding='utf-8') as f:
                                f.write(','.join(updated_tags))

            # Execute the removal
            remove_tags()
            self.tags_to_remove.clear()

        # Clear pending status
        self.tag_panel.clear_pending_status()
        self.save_btn.setEnabled(False)

        # Reload to reflect changes
        self.load_folder()
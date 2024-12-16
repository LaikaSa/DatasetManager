from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                              QPushButton, QFileDialog)
from PySide6.QtCore import Qt, Signal
from .gallery_view import GalleryView
from .tag_panel import TagPanel
from .data_model import DataModel
import os

class TagEditorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.data_model = DataModel()
        self.modified_captions = {}  # Initialize modified_captions dictionary
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Top controls
        top_layout = QHBoxLayout()
        
        # Path input and buttons
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter folder path...")
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
        
        # Gallery
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
        self.tag_panel.filter_changed.connect(self.on_filter_changed)
        self.tag_panel.tags_removal_requested.connect(self.remove_tags)
        self.tag_panel.caption_changed.connect(self.on_caption_changed)
        self.gallery.image_selected.connect(self.on_image_selected)

    def on_image_selected(self, image_path: str):
        """Handle image selection"""
        if image_path:  # Only process if an image is actually selected
            image_data = self.data_model.images.get(image_path)
            if image_data:
                caption = ', '.join(sorted(image_data.tags))
                self.tag_panel.set_caption(image_path, caption)
        else:
            self.tag_panel.clear()

    def on_caption_changed(self, image_path: str, new_caption: str):
        """Handle caption changes"""
        print(f"Caption changed for {image_path}")
        self.modified_captions[image_path] = new_caption
        self.save_btn.setEnabled(True)

    def on_filter_changed(self, tags: set, combine_logic: str, filter_logic: str):
        print(f"TagEditorTab applying filter: {len(tags)} tags")  # Debug print
        filtered_paths = self.data_model.filter_images(tags, combine_logic, filter_logic)
        filtered_images = [self.data_model.images[path] for path in filtered_paths]
        
        # Update gallery with filtered images
        self.gallery.display_images(filtered_images)
        
        # Update counter
        self.tag_panel.update_counter(len(filtered_images), len(self.data_model.images))

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.path_input.setText(folder)

    def load_folder(self):
        folder = self.path_input.text()
        if not folder or not os.path.exists(folder):
            print(f"Invalid folder path: {folder}")
            return

        print("Starting folder load...")
        self.data_model.load_directory(folder)
        print("Directory loaded, updating UI...")
        
        # Update gallery
        images = list(self.data_model.images.values())
        print(f"Displaying {len(images)} images...")
        self.gallery.display_images(images)
        
        # Update tag panel
        print(f"Updating tag panel with {len(self.data_model.tag_frequencies)} tags...")
        self.tag_panel.update_tags(self.data_model.tag_frequencies)
        
        # Update counter with total images
        self.tag_panel.update_counter(len(images), len(images))
        
        self.save_btn.setEnabled(False)
        print("Load complete")

    def unload_folder(self):
        """Unload current folder and clear all data"""
        print("Unloading folder...")  # Debug print
        self.data_model.clear()
        self.gallery.clear()
        self.tag_panel.clear()
        self.modified_captions.clear()
        self.save_btn.setEnabled(False)
        # Update counter explicitly
        self.tag_panel.update_counter(0, 0)
        print("Folder unloaded")  # Debug print

    def remove_tags(self, tags: set):
        """Handle tag removal request"""
        print(f"Removing {len(tags)} tags")
        self.data_model.remove_tags(tags)
        
        # Clear all tag selections before updating tags
        self.tag_panel.clear_all_selections()
        
        # Update tag panel with new frequencies
        self.tag_panel.update_tags(self.data_model.tag_frequencies)
        self.save_btn.setEnabled(bool(self.data_model.modified_files))
        
        # Show all images after tag removal
        self.gallery.display_images(list(self.data_model.images.values()))
        self.tag_panel.update_counter(len(self.data_model.images), len(self.data_model.images))

    def apply_filters(self, tags: set, combine_logic: str, filter_logic: str):
        print(f"Applying filters: {len(tags)} tags, {combine_logic}, {filter_logic}")
        filtered_paths = self.data_model.filter_images(tags, combine_logic, filter_logic)
        filtered_images = [self.data_model.images[path] for path in filtered_paths]
        
        # Update gallery with filtered images
        self.gallery.display_images(filtered_images)
        
        # Update counter
        self.tag_panel.update_counter(len(filtered_images), len(self.data_model.images))

    def queue_tag_removal(self, tags):
        """Queue tags for removal"""
        self.tags_to_remove.update(tags)
        self.save_btn.setEnabled(True)

    def save_changes(self):
        """Save all pending changes"""
        print("Starting save process...")  # Debug print
        
        # Process caption changes first
        for image_path, new_caption in self.modified_captions.items():
            print(f"Processing caption change for {image_path}")  # Debug print
            new_tags = {tag.strip() for tag in new_caption.split(',') if tag.strip()}
            self.data_model.update_image_tags(image_path, new_tags)
        
        # Save all changes to disk
        saved, total = self.data_model.save_changes(create_backup=True)
        print(f"Saved {saved}/{total} files")  # Debug print
        
        # Clear pending changes
        self.modified_captions.clear()
        self.save_btn.setEnabled(False)
        
        # Reload to reflect changes
        current_folder = self.path_input.text()
        if current_folder and os.path.exists(current_folder):
            self.load_folder()

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, QLabel,
                              QPushButton, QFileDialog, QMessageBox, QCheckBox, QSplitter)
from PySide6.QtCore import Qt
from .gallery_view import GalleryView
from .tag_panel import TagPanel
from .data_model import DataModel
from .loading_thread import LoadingThread 
import os
import shutil
from send2trash import send2trash

class TagEditorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.data_model = DataModel()
        self.modified_captions = {}
        self.loading_thread = None
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
        
        # Add backup captions checkbox (default off)
        self.backup_cb = QCheckBox("Backup Old Captions")
        self.backup_cb.setChecked(False)
        self.backup_cb.setToolTip("Save old caption files as .000, .001, etc. before overwriting")

        # Add parallel loading checkbox
        self.parallel_cb = QCheckBox("Parallel Loading")
        self.parallel_cb.setToolTip("Use multiple CPU cores to speed up loading (may use more memory)")
        
        top_layout.addWidget(self.path_input)
        top_layout.addWidget(self.browse_btn)
        top_layout.addWidget(self.load_btn)
        top_layout.addWidget(self.unload_btn)
        top_layout.addWidget(self.save_btn)
        top_layout.addWidget(self.backup_cb)
        top_layout.addWidget(self.parallel_cb)
        
        layout.addLayout(top_layout)

        # Add status label — fixed height so it never shifts the layout below it
        self.status_label = QLabel()
        self.status_label.setFixedHeight(18)
        self.status_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(self.status_label)

        # Split view — draggable divider between gallery and tag panel
        self.splitter = QSplitter(Qt.Horizontal)

        self.gallery = GalleryView()
        self.tag_panel = TagPanel()

        self.splitter.addWidget(self.gallery)
        self.splitter.addWidget(self.tag_panel)

        # Thumbnail is 150px wide + 5px spacing = 155px per column
        # Min: 1 column (~160px), Max: 5 columns (~780px)
        self.splitter.setCollapsible(0, False)
        self.splitter.setCollapsible(1, False)
        self.gallery.setMinimumWidth(160)
        self.gallery.setMaximumWidth(780)
        self.tag_panel.setMinimumWidth(260)

        # Give gallery a bit more space by default
        self.splitter.setSizes([600, 400])

        layout.addWidget(self.splitter)

        # Connect signals
        self.browse_btn.clicked.connect(self.browse_folder)
        self.load_btn.clicked.connect(self.load_folder)
        self.unload_btn.clicked.connect(self.unload_folder)
        self.save_btn.clicked.connect(self.save_changes)
        self.tag_panel.filter_changed.connect(self.on_filter_changed)
        self.tag_panel.tags_removal_requested.connect(self.remove_tags)
        self.tag_panel.replace_add_tags_requested.connect(self.replace_add_tags)
        self.tag_panel.caption_changed.connect(self.on_caption_changed)
        self.gallery.image_selected.connect(self.on_image_selected)
        self.tag_panel.delete_requested.connect(self.delete_files)
        self.tag_panel.move_requested.connect(self.move_files)

    def on_image_selected(self, image_path: str):
        """Handle image selection"""
        if image_path:
            image_data = self.data_model.images.get(image_path)
            if image_data:
                caption = ', '.join(image_data.tags)
                self.tag_panel.set_caption(image_path, caption)
        else:
            # Returning to grid — only clear the caption editor,
            # leave the tag list and filters untouched
            self.tag_panel.caption_tab.clear()
            self.tag_panel.tab_widget.setCurrentWidget(self.tag_panel.filter_tab)

    def on_caption_changed(self, image_path: str, new_caption: str):
        """Handle caption changes"""
        print(f"Caption changed for {image_path}")
        self.modified_captions[image_path] = new_caption
        self.save_btn.setEnabled(True)

    def on_filter_changed(self, tags: set, combine_logic: str, filter_logic: str):
        print(f"TagEditorTab applying filter: {len(tags)} tags")
        filtered_images = self.data_model.filter_images(tags, combine_logic, filter_logic)
        
        # Update gallery with filtered images directly
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
            self.status_label.setText("Invalid folder path")
            return

        # Disable controls during loading
        self.load_btn.setEnabled(False)
        self.unload_btn.setEnabled(False)
        self.parallel_cb.setEnabled(False)
        self.status_label.setText("Loading...")

        # Start loading thread
        self.loading_thread = LoadingThread(folder, self.parallel_cb.isChecked())
        self.loading_thread.progress.connect(self.update_loading_status)
        self.loading_thread.finished.connect(self.on_loading_finished)
        self.loading_thread.start()

    def update_loading_status(self, message):
        """Update status label with loading progress"""
        self.status_label.setText(message)

    def on_loading_finished(self, result):
        """Handle completion of loading thread"""
        if result is None:
            self.status_label.setText("Error loading folder")
        else:
            # Process results
            for item in result['results']:
                self.data_model.add_image(item)
            
            # Update UI
            images = list(self.data_model.images.values())
            self.gallery.display_images(images)
            self.tag_panel.update_tags(self.data_model.tag_frequencies)
            self.tag_panel.update_counter(len(images), len(images))
            
            self.status_label.setText(
                f"Loaded {len(images)} images in {result['time']:.2f} seconds"
            )

        # Re-enable controls
        self.load_btn.setEnabled(True)
        self.unload_btn.setEnabled(True)
        self.parallel_cb.setEnabled(True)

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
        
        # Clear all tag selections before updating tags (also resets FilterTab's own selected_tags)
        self.tag_panel.filter_tab.clear_filters()
        
        # Update tag panel with new frequencies
        self.tag_panel.update_tags(self.data_model.tag_frequencies)
        self.save_btn.setEnabled(bool(self.data_model.modified_files))
        
        # Show all images after tag removal
        self.gallery.display_images(list(self.data_model.images.values()))
        self.tag_panel.update_counter(len(self.data_model.images), len(self.data_model.images))
        
    def replace_add_tags(self, replace_tags: set, new_tags: list, positions: list):
            """Handle replace/add tags request"""
            print(f"Replace/add: removing {len(replace_tags)} tags, inserting {len(new_tags)} at {positions}")
            self.data_model.replace_or_add_tags(replace_tags, new_tags, positions)

            # Clear filter selections in case a replaced tag was being filtered on
            self.tag_panel.filter_tab.clear_filters()

            # Refresh tag panel and gallery
            self.tag_panel.update_tags(self.data_model.tag_frequencies)
            self.save_btn.setEnabled(bool(self.data_model.modified_files))
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

    def delete_files(self, delete_images: bool, delete_captions: bool):
        """Move files to recycle bin"""
        current_images = [img.path for img in self.gallery.get_visible_images()]
        
        deleted_images = 0
        deleted_captions = 0

        for img_path in current_images:
            if delete_images:
                try:
                    send2trash(img_path)  # Send to recycle bin instead of permanent deletion
                    deleted_images += 1
                except Exception as e:
                    print(f"Error sending image to recycle bin {img_path}: {e}")

            if delete_captions:
                caption_path = os.path.splitext(img_path)[0] + '.txt'
                try:
                    if os.path.exists(caption_path):
                        send2trash(caption_path)  # Send to recycle bin instead of permanent deletion
                        deleted_captions += 1
                except Exception as e:
                    print(f"Error sending caption to recycle bin {caption_path}: {e}")

        # Reload folder to reflect changes
        self.load_folder()
        print(f"Moved {deleted_images} images and {deleted_captions} caption files to recycle bin")

    def move_files(self, destination: str, move_images: bool, move_captions: bool):
        """Move displayed files"""
        current_images = [img.path for img in self.gallery.get_visible_images()]
        
        moved_images = 0
        moved_captions = 0

        for img_path in current_images:
            if move_images:
                try:
                    new_img_path = os.path.join(destination, os.path.basename(img_path))
                    shutil.move(img_path, new_img_path)
                    moved_images += 1
                except Exception as e:
                    print(f"Error moving image {img_path}: {e}")

            if move_captions:
                caption_path = os.path.splitext(img_path)[0] + '.txt'
                if os.path.exists(caption_path):
                    try:
                        new_caption_path = os.path.join(
                            destination, 
                            os.path.basename(caption_path)
                        )
                        shutil.move(caption_path, new_caption_path)
                        moved_captions += 1
                    except Exception as e:
                        print(f"Error moving caption {caption_path}: {e}")

        # Reload folder to reflect changes
        self.load_folder()
        print(f"Moved {moved_images} images and {moved_captions} caption files")

    def save_changes(self):
        """Save all pending changes"""
        print("Starting save process...")  # Debug print
        
        # Process caption changes first
        for image_path, new_caption in self.modified_captions.items():
            print(f"Processing caption change for {image_path}")  # Debug print
            seen = set()
            new_tags = []
            for tag in new_caption.split(','):
                tag = tag.strip()
                if tag and tag not in seen:
                    seen.add(tag)
                    new_tags.append(tag)
            self.data_model.update_image_tags(image_path, new_tags)
        
        # Save all changes to disk
        saved, total = self.data_model.save_changes(create_backup=self.backup_cb.isChecked())
        print(f"Saved {saved}/{total} files")  # Debug print
        
        # Clear pending changes
        self.modified_captions.clear()
        self.save_btn.setEnabled(False)
        
        # Reload to reflect changes
        current_folder = self.path_input.text()
        if current_folder and os.path.exists(current_folder):
            self.load_folder()

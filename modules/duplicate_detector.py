import os
import imagehash
from PIL import Image
import cv2
import numpy as np
import io
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                              QFileDialog, QMessageBox, QProgressBar, QCheckBox,
                              QSlider, QHBoxLayout, QGroupBox, QScrollArea, QLineEdit)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap, QImage
from send2trash import send2trash
from functools import lru_cache
from modules.logger import setup_logger

logger = setup_logger()
thumbnail_cache = {}

@lru_cache(maxsize=1000)  # Cache up to 1000 thumbnails
def create_cached_thumbnail(image_path):
    if image_path in thumbnail_cache:
        return thumbnail_cache[image_path]
    
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            width, height = img.size
            ratio = min(200/width, 200/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            buffer = io.BytesIO()
            img_resized.save(buffer, format='JPEG')
            qt_img = QImage.fromData(buffer.getvalue())
            pixmap = QPixmap.fromImage(qt_img)
            
            thumbnail_cache[image_path] = pixmap
            return pixmap
    except Exception as e:
        logger.error(f"Error creating thumbnail for {image_path}: {str(e)}")
        return None

class ImagePreviewGroup(QWidget):
    def __init__(self, images, similarity, method, selected_images, selection_callback):
        super().__init__()
        self.selected_images = selected_images
        self.selection_callback = selection_callback
        self.images = images
        self.containers = []
        self.init_ui(similarity, method)

    def init_ui(self, similarity, method):
        layout = QVBoxLayout()
        
        # Add similarity info
        similarity_label = QLabel(f"{method} Similarity: {similarity:.2%}")
        similarity_label.setStyleSheet("font-weight: bold; color: #2962FF;")
        layout.addWidget(similarity_label)

        # Create scroll area for images
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create widget to hold images
        image_widget = QWidget()
        image_layout = QHBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create containers for all images but load thumbnails later
        for img_path in self.images:
            try:
                img_container = ClickableImageContainer(
                    img_path,
                    img_path in self.selected_images,
                    self.on_image_clicked
                )
                image_layout.addWidget(img_container)
                self.containers.append(img_container)
                
            except Exception as e:
                logger.error(f"Error creating container for {img_path}: {str(e)}")

        image_layout.addStretch()
        scroll_area.setWidget(image_widget)
        scroll_area.setMinimumHeight(250)
        
        layout.addWidget(scroll_area)
        self.setLayout(layout)

    def on_image_clicked(self, img_path, is_selected):
        self.selection_callback(img_path, is_selected)

    def create_thumbnail(self, image_path):
        # Create a thumbnail with max size 200x200 while maintaining aspect ratio
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Calculate new dimensions
            width, height = img.size
            ratio = min(200/width, 200/height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            
            # Resize image
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to QPixmap
            buffer = io.BytesIO()
            img_resized.save(buffer, format='JPEG')
            qt_img = QImage.fromData(buffer.getvalue())
            return QPixmap.fromImage(qt_img)
        
class ClickableImageContainer(QWidget):
    def __init__(self, img_path, is_selected, callback):
        super().__init__()
        self.img_path = img_path
        self.is_selected = is_selected
        self.callback = callback
        self.thumbnail_loaded = False
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Image label with loading placeholder
        self.img_label = QLabel("Loading...")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(200, 200)

        # Resolution label
        with Image.open(self.img_path) as img:
            width, height = img.size
        self.resolution_label = QLabel(f"{width} × {height}")
        self.resolution_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.img_label)
        layout.addWidget(self.resolution_label)
        self.setLayout(layout)
        self.setFixedWidth(220)

        # Set initial style based on selection state
        self.update_style()

        # Start loading thumbnail in background
        QThread.currentThread().priority()  # Ensure we're in the main thread
        self.load_thumbnail_later()

    def load_thumbnail_later(self):
        QTimer.singleShot(10, self.load_thumbnail)

    def load_thumbnail(self):
        if not self.thumbnail_loaded:
            pixmap = create_cached_thumbnail(self.img_path)
            if pixmap:
                self.img_label.setPixmap(pixmap)
                self.thumbnail_loaded = True

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_selected = not self.is_selected
            self.update_style()
            self.callback(self.img_path, self.is_selected)

    def update_style(self):
        if self.is_selected:
            self.setStyleSheet("""
                QWidget {
                    background-color: rgba(0, 120, 215, 0.3);
                    border: 2px solid #0078D7;
                    border-radius: 5px;
                }
            """)
        else:
            self.setStyleSheet("")

class WorkerThread(QThread):
    progress = Signal(int)
    result = Signal(dict)
    finished = Signal()

    def __init__(self, folder_path, use_hash, use_hist, hash_threshold, hist_threshold):
        super().__init__()
        self.folder_path = folder_path
        self.use_hash = use_hash
        self.use_hist = use_hist
        self.hash_threshold = hash_threshold
        self.hist_threshold = hist_threshold
        self.is_running = True

    def run(self):
        def print_progress_bar(current, total, prefix='Progress:', length=50):
            filled_length = int(length * current / total)
            bar = '=' * filled_length + '-' * (length - filled_length)
            print(f'\r{prefix} [{bar}] {current}/{total}', end='', flush=True)

        image_files = []
        for file in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(file_path)

        total_files = len(image_files)
        logger.info(f"Found {total_files} images to process")
        
        # Store all images with their features
        image_features = {}
        
        # First pass: calculate features for all images
        logger.info("First pass: Calculating features")
        for idx, img_path in enumerate(image_files):
            if not self.is_running:
                break

            try:
                features = {}
                if self.use_hash:
                    with Image.open(img_path) as img:
                        features['hash'] = imagehash.average_hash(img)
                
                if self.use_hist:
                    img_cv = cv2.imread(img_path)
                    if img_cv is not None:
                        hist = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8],
                                        [0, 256, 0, 256, 0, 256])
                        features['hist'] = cv2.normalize(hist, hist).flatten()
                
                image_features[img_path] = features
                print_progress_bar(idx + 1, total_files, prefix='Processing:')

            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}", exc_info=True)

        print()  # New line after first pass
        
        # Second pass: group similar images
        logger.info("Second pass: Comparing images")
        groups = []
        processed = set()
        processed_count = 0

        for img_path, features in image_features.items():
            if img_path in processed:
                continue

            similar_images = {img_path}
            method = None
            max_similarity = 0

            # Find all similar images
            for other_path, other_features in image_features.items():
                if other_path == img_path or other_path in processed:
                    continue

                if self.use_hash and 'hash' in features and 'hash' in other_features:
                    hash_diff = features['hash'] - other_features['hash']
                    # Average hash is typically 64 bits, so normalize by that
                    similarity = 1 - (hash_diff / 64.0)
                    if similarity >= self.hash_threshold:
                        similar_images.add(other_path)
                        if similarity > max_similarity:
                            max_similarity = similarity
                            method = 'Hash'

                if self.use_hist and 'hist' in features and 'hist' in other_features:
                    hist_corr = cv2.compareHist(features['hist'], other_features['hist'], 
                                            cv2.HISTCMP_CORREL)
                    if hist_corr >= self.hist_threshold:
                        similar_images.add(other_path)
                        if hist_corr > max_similarity:
                            max_similarity = hist_corr
                            method = 'Histogram'

            # If we found similar images, create a group
            if len(similar_images) > 1:
                groups.append({
                    'images': list(similar_images),
                    'similarity': max_similarity,
                    'method': method
                })
                processed.update(similar_images)

            processed_count += 1
            print_progress_bar(processed_count, total_files, prefix='Comparing:')

        print()  # New line after second pass

        # Emit all groups
        if groups:
            logger.info(f"Found {len(groups)} groups of similar images")
            for group in groups:
                self.result.emit(group)
        else:
            logger.info("No duplicate images found")

        self.finished.emit()

    def stop(self):
        self.is_running = False

class DuplicateDetectorTab(QWidget):
    def __init__(self):
        super().__init__()
        logger.info("Initializing Duplicate Detector Tab")
        self.worker = None
        self.folder_path = None
        self.current_group_index = 0
        self.image_groups = []
        self.selected_images = set()  # Add this line
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Create folder selection layout
        folder_layout = QHBoxLayout()
        self.folder_path_input = QLineEdit()
        self.folder_path_input.setPlaceholderText("Enter or paste folder path here")
        self.browse_btn = QPushButton("Browse")
        folder_layout.addWidget(self.folder_path_input)
        folder_layout.addWidget(self.browse_btn)

        # Status label
        self.status_label = QLabel("No folder selected")

        # Add folder selection to main layout
        main_layout.addLayout(folder_layout)
        main_layout.addWidget(self.status_label)

        # Create method selection group
        method_group = QGroupBox("Detection Methods")
        method_layout = QVBoxLayout()

        # Checkbox layout (horizontal)
        checkbox_layout = QHBoxLayout()
        self.hash_checkbox = QCheckBox("Use Perceptual Hashing")
        self.hist_checkbox = QCheckBox("Use Color Histogram")
        checkbox_layout.addWidget(self.hash_checkbox)
        checkbox_layout.addWidget(self.hist_checkbox)
        checkbox_layout.addStretch()

        # Hashing controls in a collapsible widget
        self.hash_controls = QWidget()
        hash_layout = QVBoxLayout(self.hash_controls)
        
        hash_description = QLabel(
            "Hash Similarity (0-100):\n"
            "→ Slide right for stricter matching (more similar)\n"
            "← Slide left for looser matching (less similar)\n"
            "Recommended: 70-90% for best results"
        )
        hash_description.setWordWrap(True)
        
        self.hash_slider = QSlider(Qt.Horizontal)
        self.hash_slider.setRange(0, 100)
        self.hash_slider.setValue(90)
        
        self.hash_value_label = QLabel(f"Current threshold: {self.hash_slider.value()}%")
        
        hash_layout.addWidget(hash_description)
        hash_layout.addWidget(self.hash_slider)
        hash_layout.addWidget(self.hash_value_label)
        hash_layout.setContentsMargins(20, 0, 20, 0)
        self.hash_controls.setVisible(False)

        # Histogram controls in a collapsible widget
        self.hist_controls = QWidget()
        hist_layout = QVBoxLayout(self.hist_controls)
        
        hist_description = QLabel(
            "Histogram Correlation (0-100):\n"
            "← Slide left for looser matching (less similar)\n"
            "→ Slide right for stricter matching (more similar)"
        )
        hist_description.setWordWrap(True)
        
        self.hist_slider = QSlider(Qt.Horizontal)
        self.hist_slider.setRange(0, 100)
        self.hist_slider.setValue(95)
        
        self.hist_value_label = QLabel(f"Current threshold: {self.hist_slider.value()}%")
        
        hist_layout.addWidget(hist_description)
        hist_layout.addWidget(self.hist_slider)
        hist_layout.addWidget(self.hist_value_label)
        hist_layout.setContentsMargins(20, 0, 20, 0)
        self.hist_controls.setVisible(False)

        # Add all elements to method layout
        method_layout.addLayout(checkbox_layout)
        method_layout.addWidget(self.hash_controls)
        method_layout.addWidget(self.hist_controls)
        method_group.setLayout(method_layout)

        # Create control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.setEnabled(False)
        self.recycle_btn = QPushButton("Move Selected to Recycle Bin")  # Add this
        self.recycle_btn.setEnabled(False)  # Add this
        
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.recycle_btn)  # Add this

        # Connect the recycle bin button
        self.recycle_btn.clicked.connect(self.move_to_recycle_bin)

        # Preview area
        self.preview_area = QScrollArea()
        self.preview_area.setWidgetResizable(True)
        self.preview_area.setMinimumHeight(300)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous Group")
        self.next_btn = QPushButton("Next Group")
        self.group_label = QLabel("No duplicates found")
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.group_label)
        nav_layout.addWidget(self.next_btn)
        
        # Connect signals
        self.hash_slider.valueChanged.connect(self.update_hash_label)
        self.hist_slider.valueChanged.connect(self.update_hist_label)
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.prev_btn.clicked.connect(self.show_previous_group)
        self.next_btn.clicked.connect(self.show_next_group)
        self.hash_checkbox.stateChanged.connect(self.on_hash_checkbox_changed)
        self.hist_checkbox.stateChanged.connect(self.on_hist_checkbox_changed)
        
        # Initially disable navigation buttons
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        # Add all widgets to main layout
        main_layout.addWidget(method_group)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.preview_area)
        main_layout.addLayout(nav_layout)

        self.setLayout(main_layout)

        # Update connections
        self.browse_btn.clicked.connect(self.browse_folder)
        self.folder_path_input.textChanged.connect(self.on_path_changed)

    def display_current_group(self):
        if not self.image_groups:
            return
            
        group = self.image_groups[self.current_group_index]
        preview_widget = ImagePreviewGroup(
            images=group['images'],
            similarity=group['similarity'],
            method=group['method'],
            selected_images=self.selected_images,
            selection_callback=self.on_selection_changed
        )
        self.preview_area.setWidget(preview_widget)
        self.update_navigation_buttons()
        self.recycle_btn.setEnabled(len(self.selected_images) > 0)

    def on_selection_changed(self, img_path, is_selected):
        if is_selected:
            self.selected_images.add(img_path)
        else:
            self.selected_images.discard(img_path)
        self.recycle_btn.setEnabled(len(self.selected_images) > 0)

    def move_to_recycle_bin(self):
        logger.info(f"Moving {len(self.selected_images)} images to recycle bin")
        if not self.selected_images:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Move {len(self.selected_images)} selected images to recycle bin?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            from send2trash import send2trash
            failed_files = []
            
            for img_path in list(self.selected_images):  # Create a copy of the list
                try:
                    # Normalize path to handle Windows paths correctly
                    normalized_path = os.path.normpath(img_path)
                    send2trash(normalized_path)
                    self.selected_images.remove(img_path)
                    
                    # Remove the image from groups
                    for group in self.image_groups[:]:
                        group['images'] = [img for img in group['images'] if img != img_path]
                        if len(group['images']) < 2:
                            self.image_groups.remove(group)
                
                except Exception as e:
                    failed_files.append((img_path, str(e)))

            # Show results
            if failed_files:
                error_message = "Failed to move the following files to recycle bin:\n\n"
                for file_path, error in failed_files:
                    error_message += f"{os.path.basename(file_path)}: {error}\n"
                QMessageBox.warning(self, "Error", error_message)

            # Update display
            if self.image_groups:
                if self.current_group_index >= len(self.image_groups):
                    self.current_group_index = len(self.image_groups) - 1
                self.display_current_group()
            else:
                self.preview_area.setWidget(QWidget())
                self.group_label.setText("No duplicates found")
                self.current_group_index = 0

            self.update_navigation_buttons()
            self.recycle_btn.setEnabled(len(self.selected_images) > 0)

    def on_path_changed(self, path):
        logger.debug(f"Path changed to: {path}")
        path = path.strip()
        if os.path.exists(path) and os.path.isdir(path):
            logger.info(f"Valid folder path: {path}")
            self.folder_path = path
            self.status_label.setText(f"Selected folder: {path}")
            self.update_start_button()
        else:
            logger.warning(f"Invalid folder path: {path}")
            self.folder_path = None
            self.status_label.setText("Invalid folder path")
            self.update_start_button()

    def on_hash_checkbox_changed(self, state):
        """Handle hash checkbox state change"""
        self.hash_controls.setVisible(bool(state))
        self.update_start_button()

    def on_hist_checkbox_changed(self, state):
        """Handle histogram checkbox state change"""
        self.hist_controls.setVisible(bool(state))
        self.update_start_button()

    def update_hash_label(self):
        self.hash_value_label.setText(f"Current threshold: {self.hash_slider.value()}%")

    def update_hist_label(self):
        self.hist_value_label.setText(f"Current threshold: {self.hist_slider.value()}%")

    def browse_folder(self):
        logger.info("Opening folder browser")
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            logger.info(f"Selected folder: {folder_path}")
            self.folder_path_input.setText(folder_path)  # This will trigger on_path_changed

    def update_start_button(self):
        self.start_btn.setEnabled(
            (self.hash_checkbox.isChecked() or self.hist_checkbox.isChecked()) and 
            self.folder_path is not None
        )

    def start_detection(self):
        logger.info("Starting duplicate detection")
        logger.info(f"Hash detection: {self.hash_checkbox.isChecked()}")
        logger.info(f"Histogram detection: {self.hist_checkbox.isChecked()}")
        logger.info(f"Hash threshold: {self.hash_slider.value()}%")
        logger.info(f"Histogram threshold: {self.hist_slider.value()}%")
        if self.worker is not None and self.worker.isRunning():
            return

        self.image_groups = []
        self.current_group_index = 0
        self.preview_area.setWidget(QWidget())  # Clear preview area
        
        self.worker = WorkerThread(
            self.folder_path,
            self.hash_checkbox.isChecked(),
            self.hist_checkbox.isChecked(),
            self.hash_slider.value() / 100,
            self.hist_slider.value() / 100
        )
        
        self.worker.result.connect(self.update_result)
        self.worker.finished.connect(self.detection_finished)
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)  # Changed from folder_btn to browse_btn
        self.folder_path_input.setEnabled(False)  # Also disable the path input during processing
        
        self.worker.start()

    def stop_detection(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.detection_finished()

    def detection_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)  # Changed from folder_btn to browse_btn
        self.folder_path_input.setEnabled(True)  # Re-enable the path input
        
        total_groups = len(self.image_groups)
        if total_groups > 0:
            self.group_label.setText(f"Group {self.current_group_index + 1} of {total_groups}")
            self.update_navigation_buttons()
        else:
            self.group_label.setText("No duplicates found")

    def update_result(self, result_dict):
        self.add_duplicate_group(
            result_dict['images'],
            result_dict['similarity'],
            result_dict['method']
        )

    def add_duplicate_group(self, images, similarity, method):
        self.image_groups.append({
            'images': images,
            'similarity': similarity,
            'method': method
        })
        
        # If this is the first group, display it
        if len(self.image_groups) == 1:
            self.display_current_group()
            self.update_navigation_buttons()
        # Update group count
        self.group_label.setText(f"Group {self.current_group_index + 1} of {len(self.image_groups)}")

    def show_previous_group(self):
        if self.current_group_index > 0:
            self.current_group_index -= 1
            self.display_current_group()

    def show_next_group(self):
        if self.current_group_index < len(self.image_groups) - 1:
            self.current_group_index += 1
            self.display_current_group()


    def update_navigation_buttons(self):
        has_groups = len(self.image_groups) > 0
        if has_groups:
            self.prev_btn.setEnabled(self.current_group_index > 0)
            self.next_btn.setEnabled(self.current_group_index < len(self.image_groups) - 1)
            self.group_label.setText(f"Group {self.current_group_index + 1} of {len(self.image_groups)}")
        else:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.group_label.setText("No duplicates found")
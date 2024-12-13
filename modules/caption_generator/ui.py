from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                              QFileDialog, QTextEdit, QMessageBox, QCheckBox,
                              QLineEdit, QComboBox, QHBoxLayout, QSlider,
                              QSpinBox)
from PySide6.QtCore import Qt
from .models import ImageCaptioner
from .processing import CaptionGeneratorThread
import os
import multiprocessing
from modules.logger import setup_logger
logger = setup_logger()

class CaptionGeneratorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.captioner = None
        self.init_ui()
        
        # Initialize with default model after UI setup
        try:
            default_model = 'wd-eva02-large-tagger-v3'
            if self.check_model_exists(default_model):
                self.model_dropdown.setCurrentText(default_model)
                self.initialize_captioner()
        except Exception as e:
            print(f"Error initializing default model: {str(e)}")

    def init_ui(self):
        layout = QVBoxLayout()
        
        # 1. Folder Selection Section
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Enter or paste folder path here...")
        self.folder_btn = QPushButton("Browse")
        self.folder_btn.clicked.connect(self.select_folder)
        self.recursive_checkbox = QCheckBox("Recursive")
        self.recursive_checkbox.setChecked(False)
        folder_layout.addWidget(self.folder_input)
        folder_layout.addWidget(self.folder_btn)
        folder_layout.addWidget(self.recursive_checkbox)
        
        # 2. Model Selection Section
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_dropdown = QComboBox()
        self.model_dropdown.addItems([
            'wd-eva02-large-tagger-v3',
            'wd-swinv2-tagger-v3',
            'wd-convnext-tagger-v3'
        ])
        self.download_btn = QPushButton("Download Model")
        self.download_btn.clicked.connect(self.download_model)
        self.download_btn.hide()
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        model_layout.addWidget(self.download_btn)
        model_layout.addStretch()
        self.model_dropdown.currentIndexChanged.connect(self.on_model_changed)
        
        # 3. Options Checkboxes Section
        checkbox_layout = QHBoxLayout()
        self.rating_checkbox = QCheckBox("Include rating tags")
        self.underscore_checkbox = QCheckBox("Remove underscores")
        self.append_checkbox = QCheckBox("Append tags")
        self.debug_checkbox = QCheckBox("Debug")  # New debug checkbox
        
        self.rating_checkbox.setChecked(False)
        self.underscore_checkbox.setChecked(True)
        self.append_checkbox.setChecked(False)
        self.debug_checkbox.setChecked(False)
        
        self.append_checkbox.setToolTip("Append new tags to existing ones instead of replacing them")
        self.debug_checkbox.setToolTip("Show detailed debug information")
        self.debug_checkbox.stateChanged.connect(self.toggle_debug_mode)
        
        checkbox_layout.addWidget(self.rating_checkbox)
        checkbox_layout.addWidget(self.underscore_checkbox)
        checkbox_layout.addWidget(self.append_checkbox)
        checkbox_layout.addWidget(self.debug_checkbox)
        checkbox_layout.addStretch()
        
        # 4. Tags Input Section
        tags_layout = QVBoxLayout()
        
        undesired_layout = QHBoxLayout()
        undesired_label = QLabel("Undesired tags:")
        self.undesired_input = QLineEdit()
        self.undesired_input.setPlaceholderText("Enter tags to exclude (comma-separated)")
        undesired_layout.addWidget(undesired_label)
        undesired_layout.addWidget(self.undesired_input)
        
        prefix_layout = QHBoxLayout()
        prefix_label = QLabel("Prefix tags:")
        self.prefix_input = QLineEdit()
        self.prefix_input.setPlaceholderText("Enter tags to add at beginning (comma-separated)")
        prefix_layout.addWidget(prefix_label)
        prefix_layout.addWidget(self.prefix_input)
        
        tags_layout.addLayout(undesired_layout)
        tags_layout.addLayout(prefix_layout)
        
        # 5. Thresholds Section
        threshold_layout = QHBoxLayout()
        
        # Overall threshold
        thresh_layout = self._create_threshold_slider("Threshold", 35)
        self.thresh_slider = thresh_layout.itemAt(1).widget()
        self.thresh_value = thresh_layout.itemAt(2).widget()
        
        # General threshold
        general_layout = self._create_threshold_slider("General threshold", 35)
        self.general_slider = general_layout.itemAt(1).widget()
        self.general_value = general_layout.itemAt(2).widget()
        
        # Character threshold
        character_layout = self._create_threshold_slider("Character threshold", 35)
        self.character_slider = character_layout.itemAt(1).widget()
        self.character_value = character_layout.itemAt(2).widget()
        
        threshold_layout.addLayout(thresh_layout)
        threshold_layout.addLayout(general_layout)
        threshold_layout.addLayout(character_layout)
        
        # 6. Batch Processing Section
        batch_layout = QHBoxLayout()
        batch_label = QLabel("Batch size:")
        self.batch_size_input = QSpinBox()
        self.batch_size_input.setMinimum(1)
        self.batch_size_input.setMaximum(32)
        self.batch_size_input.setValue(1)
        
        worker_label = QLabel("Data loader workers:")
        self.worker_count_input = QSpinBox()
        self.worker_count_input.setMinimum(0)
        self.worker_count_input.setMaximum(multiprocessing.cpu_count())
        self.worker_count_input.setValue(2)
        
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(self.batch_size_input)
        batch_layout.addWidget(worker_label)
        batch_layout.addWidget(self.worker_count_input)
        batch_layout.addStretch()
        
        # 7. Status and Process Section
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        
        self.process_btn = QPushButton("Generate Captions")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        
        # Add all sections to main layout
        layout.addLayout(folder_layout)
        layout.addLayout(model_layout)
        layout.addLayout(checkbox_layout)
        layout.addLayout(tags_layout)
        layout.addLayout(threshold_layout)
        layout.addLayout(batch_layout)
        layout.addWidget(self.status_text)
        layout.addWidget(self.process_btn)
        
        self.setLayout(layout)
        
        # Connect signals
        self.folder_input.textChanged.connect(self.validate_folder)
        self.initialize_captioner()

    def _create_threshold_slider(self, label, default_value):
        """Helper method to create threshold slider layouts"""
        layout = QVBoxLayout()
        label_widget = QLabel(label)
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(100)
        slider.setValue(default_value)
        value_label = QLabel(f"{default_value/100:.2f}")
        slider.valueChanged.connect(
            lambda v: value_label.setText(f"{v/100:.2f}")
        )
        layout.addWidget(label_widget)
        layout.addWidget(slider)
        layout.addWidget(value_label)
        return layout

    def toggle_debug_mode(self, state):
        """Toggle debug mode for logger and captioner"""
        global logger
        logger = setup_logger(debug_mode=bool(state))
        if self.captioner:
            self.captioner.debug_mode = bool(state)

    def check_model_exists(self, model_name):
        """Check if model files exist"""
        try:
            # Get root directory (two levels up from ui.py)
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            model_info = ImageCaptioner.MODELS[model_name]
            
            # Construct paths using os.path.join
            model_dir = os.path.join(root_dir, model_info['path'])
            model_path = os.path.normpath(os.path.join(model_dir, "model.onnx"))
            tags_path = os.path.normpath(os.path.join(model_dir, "selected_tags.csv"))
            
            print(f"Checking model files:")
            print(f"Root directory: {root_dir}")
            print(f"Model directory: {model_dir}")
            print(f"Model path: {model_path}")
            print(f"Tags path: {tags_path}")
            print(f"Model exists: {os.path.exists(model_path)}")
            print(f"Tags exist: {os.path.exists(tags_path)}")
            
            exists = os.path.exists(model_path) and os.path.exists(tags_path)
            print(f"Final check result: {exists}")
            
            return exists
        except Exception as e:
            print(f"Error checking model existence: {str(e)}")
            return False

    def on_model_changed(self):
        """Handle model change event"""
        try:
            model_name = self.model_dropdown.currentText()
            print(f"Model changed to: {model_name}")  # Debug print
            
            if self.check_model_exists(model_name):
                print(f"Model files found for {model_name}")  # Debug print
                self.download_btn.hide()
                self.initialize_captioner()
            else:
                print(f"Model files not found for {model_name}")  # Debug print
                self.download_btn.show()
                self.process_btn.setEnabled(False)
                self.status_text.append(f"Model {model_name} not found. Click Download Model to download it.")
        except Exception as e:
            print(f"Error in on_model_changed: {str(e)}")  # Debug print
            self.status_text.append(f"Error changing model: {str(e)}")

    def download_model(self):
        """Download the selected model"""
        try:
            from huggingface_hub import hf_hub_download
            import os
            
            model_name = self.model_dropdown.currentText()
            model_info = ImageCaptioner.MODELS[model_name]
            repo_id = model_info['repo_id']
            
            # Get root directory
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            # Disable UI elements during download
            self.download_btn.setEnabled(False)
            self.model_dropdown.setEnabled(False)
            self.download_btn.setText("Downloading...")
            self.status_text.append(f"Downloading {model_name} from {repo_id}...")
            
            # Create model directory in root/models
            model_dir = os.path.normpath(os.path.join(root_dir, model_info['path']))
            os.makedirs(model_dir, exist_ok=True)
            
            print(f"Downloading to directory: {model_dir}")  # Debug print
            
            # Download files
            files = ["model.onnx", "selected_tags.csv"]
            for file in files:
                self.status_text.append(f"Downloading {file}...")
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=file,
                        local_dir=model_dir,
                        force_download=True
                    )
                    target_path = os.path.normpath(os.path.join(model_dir, file))
                    print(f"Downloaded to: {downloaded_path}")  # Debug print
                    print(f"Target path: {target_path}")  # Debug print
                    
                    if downloaded_path != target_path:
                        import shutil
                        shutil.move(downloaded_path, target_path)
                    self.status_text.append(f"Downloaded {file}")
                except Exception as e:
                    self.status_text.append(f"Error downloading {file}: {str(e)}")
                    raise
            
            # Verify files exist after download
            if not self.check_model_exists(model_name):
                raise Exception(f"Files not found after download in {model_dir}")
            
            # Re-enable UI elements
            self.download_btn.setEnabled(True)
            self.model_dropdown.setEnabled(True)
            self.download_btn.setText("Download Model")
            
            # Hide download button and initialize captioner
            self.download_btn.hide()
            self.status_text.append(f"Model {model_name} downloaded successfully!")
            self.initialize_captioner()
            
        except Exception as e:
            error_msg = f"Error downloading model: {str(e)}"
            self.status_text.append(error_msg)
            logger.error(error_msg)
            self.download_btn.setEnabled(True)
            self.model_dropdown.setEnabled(True)
            self.download_btn.setText("Retry Download")

    def validate_folder(self, path):
        """Validate the folder path and enable/disable process button"""
        if os.path.isdir(path):
            self.process_btn.setEnabled(True)
            self.folder_input.setStyleSheet("")
            self.status_text.append(f"Valid folder path: {path}")
        else:
            self.process_btn.setEnabled(False)
            self.folder_input.setStyleSheet("background-color: #FFE6E6;")  # Light red background for invalid path

    def initialize_captioner(self):
        try:
            model_name = self.model_dropdown.currentText()
            print(f"Initializing captioner with model: {model_name}")
            
            if not self.check_model_exists(model_name):
                print(f"Model files not found during initialization")
                self.status_text.append(f"Model {model_name} not found. Please download it first.")
                self.download_btn.show()
                self.process_btn.setEnabled(False)
                return
            
            # Pass debug state when creating captioner
            self.captioner = ImageCaptioner(
                model_name,
                debug_mode=self.debug_checkbox.isChecked()
            )
            
            if self.captioner is None or self.captioner.session is None:
                raise Exception("Failed to initialize captioner or session")
                
            self.status_text.append(f"Caption model {model_name} loaded successfully")
            logger.info(f"Caption model {model_name} loaded successfully")
            
            # Enable process button if folder is valid
            self.validate_folder(self.folder_input.text())
            
        except Exception as e:
            error_msg = f"Error loading caption model: {str(e)}"
            print(f"Initialization error: {error_msg}")
            self.status_text.append(error_msg)
            logger.error(error_msg)
            self.process_btn.setEnabled(False)
            self.captioner = None

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_input.setText(folder)  # Changed from folder_label to folder_input

    def process_undesired_tags(self, tags_input):
        """Process undesired tags input to handle both underscore and space versions"""
        if not tags_input.strip():
            return set()
        
        # Split by comma and clean up spaces
        tags = [tag.strip() for tag in tags_input.split(',')]
        
        # Create a set with both underscore and space versions of each tag
        processed_tags = set()
        for tag in tags:
            # Add the original tag
            processed_tags.add(tag)
            # Add the version with underscores replaced by spaces
            processed_tags.add(tag.replace('_', ' '))
            # Add the version with spaces replaced by underscores
            processed_tags.add(tag.replace(' ', '_'))
        
        return processed_tags
    
    def process_tag_list(self, tags_input):
        """Process comma-separated tag list and handle both underscore and space versions"""
        if not tags_input.strip():
            return []
        
        # Split by comma and clean up spaces
        return [tag.strip() for tag in tags_input.split(',') if tag.strip()]

    def start_processing(self):
        folder_path = self.folder_input.text()
        if not os.path.isdir(folder_path):
            QMessageBox.warning(self, "Error", "Please select a valid folder")
            return

        # Verify captioner is initialized
        if self.captioner is None:
            self.status_text.append("Error: Captioner not initialized. Please select a model first.")
            return

        # Process undesired and prefix tags
        undesired_tags = self.process_undesired_tags(self.undesired_input.text())
        prefix_tags = self.process_tag_list(self.prefix_input.text())
        
        try:
            # Test captioner before starting thread
            test_result = self.captioner.session is not None
            if not test_result:
                raise ValueError("Captioner session is not initialized")
        except Exception as e:
            self.status_text.append(f"Error: Captioner is not ready: {str(e)}")
            return

        self.process_btn.setEnabled(False)
        self.folder_btn.setEnabled(False)
        self.folder_input.setEnabled(False)
        self.recursive_checkbox.setEnabled(False)
        self.undesired_input.setEnabled(False)
        self.prefix_input.setEnabled(False)
        self.append_checkbox.setEnabled(False)
        
        # Get threshold values
        thresh = float(self.thresh_slider.value()) / 100
        general_threshold = float(self.general_slider.value()) / 100
        character_threshold = float(self.character_slider.value()) / 100

        # Pass all options to the worker thread
        self.worker = CaptionGeneratorThread(
            self.captioner, 
            folder_path,
            include_rating=self.rating_checkbox.isChecked(),
            remove_underscore=self.underscore_checkbox.isChecked(),
            recursive=self.recursive_checkbox.isChecked(),
            undesired_tags=undesired_tags,
            prefix_tags=prefix_tags,
            append_tags=self.append_checkbox.isChecked(),
            thresh=thresh,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            batch_size=self.batch_size_input.value(),
            worker_count=self.worker_count_input.value()
        )
        self.worker.caption_generated.connect(self.update_status)
        self.worker.process_completed.connect(self.process_completed)
        self.worker.error_occurred.connect(self.handle_error)
        
        # Basic info always shown
        logger.info(f"Starting caption generation with settings:")
        logger.info(f"Folder: {folder_path}")
        logger.info(f"Options:")
        logger.info(f"  - Append mode: {self.append_checkbox.isChecked()}")
        logger.info(f"  - Rating tags: {self.rating_checkbox.isChecked()}")
        logger.info(f"  - Remove underscores: {self.underscore_checkbox.isChecked()}")
        logger.info(f"  - Recursive: {self.recursive_checkbox.isChecked()}")

        # Detailed info only in debug mode
        if self.debug_checkbox.isChecked():
            logger.debug("Detailed settings:")
            logger.debug(f"  Thresholds:")
            logger.debug(f"    - General: {general_threshold:.3f}")
            logger.debug(f"    - Character: {character_threshold:.3f}")
            logger.debug(f"    - Overall: {thresh:.3f}")
            logger.debug(f"  Undesired tags: {undesired_tags}")
            logger.debug(f"  Prefix tags: {prefix_tags}")
            logger.debug(f"  Batch size: {self.batch_size_input.value()}")
            logger.debug(f"  Worker count: {self.worker_count_input.value()}")
        self.worker.start()

    def update_status(self, image_path, caption):
        # Show relative path if recursive, otherwise just filename
        if self.recursive_checkbox.isChecked():
            relative_path = os.path.relpath(image_path, self.folder_input.text())
            self.status_text.append(f"Generated caption for {relative_path}")
        else:
            self.status_text.append(f"Generated caption for {os.path.basename(image_path)}")

    def process_completed(self):
        self.process_btn.setEnabled(True)
        self.folder_btn.setEnabled(True)
        self.folder_input.setEnabled(True)
        self.recursive_checkbox.setEnabled(True)
        self.undesired_input.setEnabled(True)
        self.prefix_input.setEnabled(True)
        self.append_checkbox.setEnabled(True)  # Re-enable append checkbox
        self.rating_checkbox.setEnabled(True)  # Also re-enable other checkboxes
        self.underscore_checkbox.setEnabled(True)
        completion_msg = "Caption generation completed!"
        self.status_text.append(completion_msg)
        logger.info(completion_msg)

    def handle_error(self, error_message):
        self.status_text.append(f"Error: {error_message}")
        logger.error(error_message)
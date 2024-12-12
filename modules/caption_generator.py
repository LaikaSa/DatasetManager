from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout, QSlider,
                              QFileDialog, QTextEdit, QMessageBox, QCheckBox, QLineEdit, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal
import os
import pandas as pd
import numpy as np
import onnxruntime
from PIL import Image
from modules.logger import setup_logger
import sys
import multiprocessing
from collections import defaultdict

logger = setup_logger()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class ImageCaptioner:
    MODELS = {
        'wd-eva02-large-tagger-v3': {
            'path': 'models/wd-eva02-large-tagger-v3',
            'type': 'eva02-v3',
            'repo_id': 'SmilingWolf/wd-eva02-large-tagger-v3',
            'opset': 17
        },
        'wd-swinv2-tagger-v3': {
            'path': 'models/wd-swinv2-tagger-v3',
            'type': 'swinv2-v3',
            'repo_id': 'SmilingWolf/wd-swinv2-tagger-v3',
            'opset': 17
        },
        'wd-convnext-tagger-v3': {
            'path': 'models/wd-convnext-tagger-v3',
            'type': 'convnext-v3',
            'repo_id': 'SmilingWolf/wd-convnext-tagger-v3',
            'opset': 17
        }
    }

    def __init__(self, model_name='wd-eva02-large-tagger-v3'):
        print(f"Initializing ImageCaptioner with model: {model_name}")
        
        # Get root directory
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.MODELS.keys())}")

        # Get model info
        model_info = self.MODELS[model_name]
        self.model_type = model_info['type']
        
        # Construct full paths
        model_dir = os.path.join(root_dir, model_info['path'])
        model_path = os.path.join(model_dir, "model.onnx")
        tags_path = os.path.join(model_dir, "selected_tags.csv")

        # Verify paths
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        if not os.path.exists(tags_path):
            raise FileNotFoundError(f"Tags file not found at: {tags_path}")

        # Load model
        self.session = self.create_session(model_path)
        if self.session is None:
            raise Exception("Failed to create ONNX session")

        # Load tags
        tags_df = pd.read_csv(tags_path)
        self.load_tags(tags_df)
        
        # Get model input shape
        _, height, width, _ = self.session.get_inputs()[0].shape
        self.target_size = height
        print(f"Model input shape: {height}x{width}")

    def _get_model_type(self, model_path):
        """Determine the model type from the path"""
        path = model_path.lower()
        if 'swinv2' in path and 'v3' in path:
            return 'swinv2-v3'
        elif 'convnext' in path and 'v3' in path:
            return 'convnext-v3'
        else:
            return 'v2'  # default to v2 for original model

    def load_tags(self, tags_df):
        # Store the original names without any processing
        self.tag_names = tags_df["name"].tolist()
        
        # Separate tags by category
        self.rating_indexes = list(np.where(tags_df["category"] == 9)[0])
        self.general_indexes = list(np.where(tags_df["category"] == 0)[0])
        self.character_indexes = list(np.where(tags_df["category"] == 4)[0])

    def prepare_image(self, image_path):
        # Load image
        image = Image.open(image_path).convert('RGBA')
        
        # Create white background
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")
        
        # Pad image to square
        max_dim = max(image.size)
        pad_left = (max_dim - image.size[0]) // 2
        pad_top = (max_dim - image.size[1]) // 2
        
        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))
        
        # Resize if needed
        if max_dim != self.target_size:
            padded_image = padded_image.resize(
                (self.target_size, self.target_size),
                Image.BICUBIC
            )
        
        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)
        
        # Convert RGB to BGR
        image_array = image_array[:, :, ::-1]
        
        return np.expand_dims(image_array, axis=0)

    def generate_caption(self, image_path, 
                        general_threshold=0.35, 
                        character_threshold=0.85,
                        thresh=0.35,
                        remove_underscore=True,
                        undesired_tags=None,
                        always_first_tags=None,
                        caption_separator=", ",
                        include_rating=False):
        # Use thresh as default if specific thresholds aren't provided
        general_threshold = general_threshold if general_threshold != thresh else thresh
        character_threshold = character_threshold if character_threshold != thresh else thresh
        try:
            # Prepare image
            image = self.prepare_image(image_path)
            
            # Run inference
            label_name = self.session.get_outputs()[0].name
            preds = self.session.run([label_name], {self.input_name: image})[0]
            
            # Process predictions
            labels = list(zip(self.tag_names, preds[0].astype(float)))
            
            # Initialize combined_tags with prefix tags if provided
            combined_tags = []
            if always_first_tags:
                combined_tags.extend(always_first_tags)
            
            # Add rating tag only if include_rating is True
            if include_rating:
                ratings_names = [labels[i] for i in self.rating_indexes]
                rating = dict(ratings_names)
                rating_tag = max(rating.items(), key=lambda x: x[1])[0]
                if rating_tag not in combined_tags:  # Avoid duplicates
                    combined_tags.append(rating_tag)
            
            # Process character tags
            character_names = [labels[i] for i in self.character_indexes]
            character_res = [x for x in character_names if x[1] > character_threshold]
            character_res = dict(character_res)
            sorted_character = sorted(character_res.items(), key=lambda x: x[1], reverse=True)
            for tag, _ in sorted_character:
                if tag not in combined_tags:  # Avoid duplicates
                    combined_tags.append(tag)
            
            # Process general tags
            general_names = [labels[i] for i in self.general_indexes]
            general_res = [x for x in general_names if x[1] > general_threshold]
            general_res = dict(general_res)
            sorted_general = sorted(general_res.items(), key=lambda x: x[1], reverse=True)
            for tag, _ in sorted_general:
                if tag not in combined_tags:  # Avoid duplicates
                    combined_tags.append(tag)
            
            # Remove undesired tags if any
            if undesired_tags:
                combined_tags = [tag for tag in combined_tags if tag not in undesired_tags]
            
            # Handle underscore removal only after all tag processing is done
            if remove_underscore:
                # List of tags where underscore should be preserved
                kaomojis = ["0_0", "(o)_(o)", "+_+", "+_-", "._.", "<o>_<o>", "<|>_<|>", 
                        "=_=", ">_<", "3_3", "6_9", ">_o", "@_@", "^_^", "o_o", 
                        "u_u", "x_x", "|_|", "||_||"]
                
                # Process each tag
                combined_tags = [
                    tag if tag in kaomojis else tag.replace("_", " ")
                    for tag in combined_tags
                ]
            
            # Join tags with separator
            caption = caption_separator.join(combined_tags)
            
            return caption
            
        except Exception as e:
            print(f"Error generating caption for {image_path}: {e}")
            return "error_generating_caption"

    def convert_model_opset(self, model_path, target_opset=3):
        try:
            import onnx
            from onnx import version_converter
            
            print(f"Converting model {model_path} to opset {target_opset}")
            
            # Load the model
            model = onnx.load(model_path)
            
            # Convert the model to the target opset
            converted_model = version_converter.convert_version(model, target_opset)
            
            # Save to a temporary path
            temp_path = model_path.replace('.onnx', f'_opset{target_opset}.onnx')
            onnx.save(converted_model, temp_path)
            
            print(f"Model converted and saved to {temp_path}")
            return temp_path
            
        except Exception as e:
            print(f"Error converting model: {e}")
            return model_path

    def create_session(self, model_path):
        try:
            import onnx
            import onnxruntime as ort
            
            print(f"Loading ONNX model: {model_path}")
            model = onnx.load(model_path)
            input_name = model.graph.input[0].name
            self.input_name = input_name

            print(f"Available providers: {ort.get_available_providers()}")
            
            # Configure session options
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            # Set up providers
            providers = []
            provider_options = []
            
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.append("CUDAExecutionProvider")
                provider_options.append({
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                })
            
            # Always add CPU provider
            providers.append("CPUExecutionProvider")
            provider_options.append({})
            
            print(f"Using providers: {providers}")
            print(f"With options: {provider_options}")
            
            # Create session
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
                provider_options=provider_options
            )
            
            print(f"Session created successfully with active providers: {session.get_providers()}")
            return session

        except Exception as e:
            print(f"Error creating ONNX session: {e}")
            print(f"Exception type: {type(e)}")
            print(f"Exception args: {e.args}")
            
            try:
                print("Attempting fallback with minimal settings...")
                session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider'],
                    provider_options=[{}]
                )
                print("Successfully created basic session")
                return session
            except Exception as fallback_e:
                print(f"Fallback also failed: {fallback_e}")
                return None

class ProgressBar:
    def __init__(self, total, prefix='', length=20):
        self.total = total
        self.prefix = prefix
        self.length = length
        self.current = 0

    def update(self, current):
        self.current = current
        filled_length = int(self.length * current // self.total)
        bar = '=' * filled_length + '-' * (self.length - filled_length)
        sys.stdout.write(f'\r{self.prefix}[{bar}] {current}/{self.total}')
        sys.stdout.flush()
        if current == self.total:
            sys.stdout.write('\n')
            sys.stdout.flush()

class CaptionGeneratorThread(QThread):
    caption_generated = Signal(str, str)
    process_completed = Signal()
    error_occurred = Signal(str)

    def __init__(self, captioner, folder_path, include_rating=False, 
                 remove_underscore=True, recursive=False, 
                 undesired_tags=None, prefix_tags=None, append_tags=False,
                 thresh=0.35, general_threshold=0.35, character_threshold=0.35):
        super().__init__()
        self.captioner = captioner
        self.folder_path = folder_path
        self.include_rating = include_rating
        self.remove_underscore = remove_underscore
        self.recursive = recursive
        self.undesired_tags = undesired_tags or set()
        self.prefix_tags = prefix_tags or []
        self.append_tags = append_tags
        self.thresh = thresh
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold

    def get_image_files(self, folder_path):
        image_files = []
        if self.recursive:
            # Walk through directory and subdirectories
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_files.append(os.path.join(root, file))
        else:
            # Only get files from the main directory
            image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                         if os.path.isfile(os.path.join(folder_path, f)) and
                         f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        return image_files

    def run(self):
        try:
            # First check if captioner is properly initialized
            if self.captioner is None:
                raise ValueError("Captioner is not properly initialized")
                
            # Check if the captioner has the required methods
            if not hasattr(self.captioner, 'generate_caption'):
                raise ValueError("Captioner does not have generate_caption method")

            image_files = self.get_image_files(self.folder_path)
            if not image_files:
                self.error_occurred.emit("No image files found in the specified folder")
                return
                
            total_files = len(image_files)
            print(f"Found {total_files} images to process")
            
            progress = ProgressBar(total_files, prefix='Processing: ')

            for i, image_path in enumerate(image_files, 1):
                try:
                    print(f"Processing image {i}/{total_files}: {image_path}")
                    # Generate new caption
                    caption = self.captioner.generate_caption(
                        image_path, 
                        include_rating=self.include_rating,
                        remove_underscore=self.remove_underscore,
                        undesired_tags=self.undesired_tags,
                        always_first_tags=self.prefix_tags,
                        general_threshold=self.general_threshold,
                        character_threshold=self.character_threshold,
                        thresh=self.thresh
                    )
                    
                    if caption is None:
                        raise ValueError(f"Failed to generate caption for {image_path}")
                    
                    txt_path = os.path.splitext(image_path)[0] + '.txt'
                    
                    # Handle append mode
                    if self.append_tags and os.path.exists(txt_path):
                        try:
                            with open(txt_path, 'r', encoding='utf-8') as f:
                                existing_content = f.read().strip()
                            
                            # Split existing and new tags
                            existing_tags = [tag.strip() for tag in existing_content.split(',') if tag.strip()]
                            new_tags = [tag.strip() for tag in caption.split(',') if tag.strip()]
                            
                            # Remove duplicates while preserving order
                            combined_tags = []
                            seen = set()
                            
                            # Add existing tags first
                            for tag in existing_tags:
                                if tag not in seen:
                                    combined_tags.append(tag)
                                    seen.add(tag)
                            
                            # Add new tags
                            for tag in new_tags:
                                if tag not in seen:
                                    combined_tags.append(tag)
                                    seen.add(tag)
                            
                            # Create final caption
                            caption = ', '.join(combined_tags)
                            
                        except Exception as e:
                            print(f"Error reading existing caption for {image_path}: {e}")
                    
                    # Write caption to file
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(caption + '\n')
                    
                    self.caption_generated.emit(image_path, caption)
                    progress.update(i)
                
                except Exception as e:
                    error_msg = f"Error processing {image_path}: {str(e)}"
                    print(error_msg)
                    self.error_occurred.emit(error_msg)
                    continue

            self.process_completed.emit()

        except Exception as e:
            error_msg = f"Process error: {str(e)}"
            print(error_msg)
            self.error_occurred.emit(error_msg)

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
        
        # Create horizontal layout for folder selection and recursive option
        folder_layout = QHBoxLayout()
        
        # Folder path input box
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Enter or paste folder path here...")
        
        # Folder browse button
        self.folder_btn = QPushButton("Browse")
        self.folder_btn.clicked.connect(self.select_folder)
        
        # Recursive checkbox
        self.recursive_checkbox = QCheckBox("Recursive")
        self.recursive_checkbox.setChecked(False)
        
        # Add widgets to folder layout
        folder_layout.addWidget(self.folder_input)
        folder_layout.addWidget(self.folder_btn)
        folder_layout.addWidget(self.recursive_checkbox)
        
        # Create model selection layout
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
        self.download_btn.hide()  # Hide initially
        
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        model_layout.addWidget(self.download_btn)
        model_layout.addStretch()
        
        # Connect model change event
        self.model_dropdown.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_dropdown)
        
        # Create a horizontal layout for checkboxes
        checkbox_layout = QHBoxLayout()
        
        # Rating tag checkbox
        self.rating_checkbox = QCheckBox("Include rating tags")
        self.rating_checkbox.setChecked(False)
        
        # Underscore checkbox
        self.underscore_checkbox = QCheckBox("Remove underscores")
        self.underscore_checkbox.setChecked(True)
        
        # Append tags checkbox
        self.append_checkbox = QCheckBox("Append tags")
        self.append_checkbox.setChecked(False)
        self.append_checkbox.setToolTip("Append new tags to existing ones instead of replacing them")
        
        # Add checkboxes to horizontal layout
        checkbox_layout.addWidget(self.rating_checkbox)
        checkbox_layout.addWidget(self.underscore_checkbox)
        checkbox_layout.addWidget(self.append_checkbox)
        checkbox_layout.addStretch()
        
        # Create layout for undesired tags
        undesired_layout = QHBoxLayout()
        undesired_label = QLabel("Undesired tags:")
        self.undesired_input = QLineEdit()
        self.undesired_input.setPlaceholderText("Enter tags to exclude (comma-separated)")
        undesired_layout.addWidget(undesired_label)
        undesired_layout.addWidget(self.undesired_input)
        
        # Create layout for prefix tags
        prefix_layout = QHBoxLayout()
        prefix_label = QLabel("Prefix tags:")
        self.prefix_input = QLineEdit()
        self.prefix_input.setPlaceholderText("Enter tags to add at beginning (comma-separated)")
        prefix_layout.addWidget(prefix_label)
        prefix_layout.addWidget(self.prefix_input)
        
        # Create threshold sliders layout
        threshold_layout = QHBoxLayout()
        
        # Overall threshold slider
        thresh_layout = QVBoxLayout()
        thresh_label = QLabel("Threshold")
        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setMinimum(0)
        self.thresh_slider.setMaximum(100)
        self.thresh_slider.setValue(35)  # Default 0.35
        self.thresh_value = QLabel("0.35")
        self.thresh_slider.valueChanged.connect(
            lambda v: self.thresh_value.setText(f"{v/100:.2f}")
        )
        thresh_layout.addWidget(thresh_label)
        thresh_layout.addWidget(self.thresh_slider)
        thresh_layout.addWidget(self.thresh_value)
        
        # General threshold slider
        general_layout = QVBoxLayout()
        general_label = QLabel("General threshold")
        self.general_slider = QSlider(Qt.Horizontal)
        self.general_slider.setMinimum(0)
        self.general_slider.setMaximum(100)
        self.general_slider.setValue(35)  # Default 0.35
        self.general_value = QLabel("0.35")
        self.general_slider.valueChanged.connect(
            lambda v: self.general_value.setText(f"{v/100:.2f}")
        )
        general_layout.addWidget(general_label)
        general_layout.addWidget(self.general_slider)
        general_layout.addWidget(self.general_value)
        
        # Character threshold slider
        character_layout = QVBoxLayout()
        character_label = QLabel("Character threshold")
        self.character_slider = QSlider(Qt.Horizontal)
        self.character_slider.setMinimum(0)
        self.character_slider.setMaximum(100)
        self.character_slider.setValue(35)  # Default 0.35
        self.character_value = QLabel("0.35")
        self.character_slider.valueChanged.connect(
            lambda v: self.character_value.setText(f"{v/100:.2f}")
        )
        character_layout.addWidget(character_label)
        character_layout.addWidget(self.character_slider)
        character_layout.addWidget(self.character_value)
        
        # Add all threshold layouts
        threshold_layout.addLayout(thresh_layout)
        threshold_layout.addLayout(general_layout)
        threshold_layout.addLayout(character_layout)
        
        # Add threshold layout to main layout
        layout.addLayout(threshold_layout)

        # Status display
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        
        # Process button
        self.process_btn = QPushButton("Generate Captions")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setEnabled(False)
        
        # Add all layouts to main layout
        layout.addLayout(folder_layout)
        layout.addLayout(model_layout)
        layout.addLayout(checkbox_layout)
        layout.addLayout(undesired_layout)
        layout.addLayout(prefix_layout)
        layout.addWidget(self.status_text)
        layout.addWidget(self.process_btn)
        
        self.setLayout(layout)
        
        # Connect folder input changes to validation
        self.folder_input.textChanged.connect(self.validate_folder)
        
        # Initialize captioner with default model
        self.initialize_captioner()

    def check_model_exists(self, model_name):
        """Check if model files exist"""
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_info = ImageCaptioner.MODELS[model_name]
        model_dir = os.path.join(root_dir, model_info['path'])
        model_path = os.path.join(model_dir, "model.onnx")
        tags_path = os.path.join(model_dir, "selected_tags.csv")
        
        return os.path.exists(model_path) and os.path.exists(tags_path)

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
            repo_id = model_info['repo_id']  # This should work now
            
            # Disable UI elements during download
            self.download_btn.setEnabled(False)
            self.model_dropdown.setEnabled(False)
            self.download_btn.setText("Downloading...")
            self.status_text.append(f"Downloading {model_name} from {repo_id}...")
            
            # Create model directory
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(root_dir, model_info['path'])
            os.makedirs(model_dir, exist_ok=True)
            
            # Download files
            files = ["model.onnx", "selected_tags.csv"]
            for file in files:
                self.status_text.append(f"Downloading {file}...")
                try:
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=file,
                        local_dir=model_dir,  # Changed from cache_dir to local_dir
                        force_download=True
                    )
                    # Move file to correct location if needed
                    target_path = os.path.join(model_dir, file)
                    if downloaded_path != target_path:
                        import shutil
                        shutil.move(downloaded_path, target_path)
                    self.status_text.append(f"Downloaded {file}")
                except Exception as e:
                    self.status_text.append(f"Error downloading {file}: {str(e)}")
                    raise
            
            # Re-enable UI elements
            self.download_btn.setEnabled(True)
            self.model_dropdown.setEnabled(True)
            self.download_btn.setText("Download Model")
            
            # Hide download button and initialize captioner
            self.download_btn.hide()
            self.status_text.append(f"Model {model_name} downloaded successfully!")
            self.initialize_captioner()
            
        except Exception as e:
            self.status_text.append(f"Error downloading model: {str(e)}")
            self.download_btn.setEnabled(True)
            self.model_dropdown.setEnabled(True)
            self.download_btn.setText("Retry Download")
            logger.error(f"Error downloading model: {str(e)}")

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
            print(f"Initializing captioner with model: {model_name}")  # Debug print
            
            if not self.check_model_exists(model_name):
                print(f"Model files not found during initialization")  # Debug print
                self.status_text.append(f"Model {model_name} not found. Please download it first.")
                self.download_btn.show()
                self.process_btn.setEnabled(False)
                return
            
            # Create new captioner instance
            self.captioner = ImageCaptioner(model_name)
            
            if self.captioner is None or self.captioner.session is None:
                raise Exception("Failed to initialize captioner or session")
                
            self.status_text.append(f"Caption model {model_name} loaded successfully")
            logger.info(f"Caption model {model_name} loaded successfully")
            
            # Enable process button if folder is valid
            self.validate_folder(self.folder_input.text())
            
        except Exception as e:
            error_msg = f"Error loading caption model: {str(e)}"
            print(f"Initialization error: {error_msg}")  # Debug print
            self.status_text.append(error_msg)
            logger.error(error_msg)
            self.process_btn.setEnabled(False)
            self.captioner = None  # Reset captioner on failure

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
            character_threshold=character_threshold
        )
        self.worker.caption_generated.connect(self.update_status)
        self.worker.process_completed.connect(self.process_completed)
        self.worker.error_occurred.connect(self.handle_error)
        
        logger.info(f"Starting caption generation for folder: {folder_path}")
        logger.info(f"Include rating tags: {self.rating_checkbox.isChecked()}")
        logger.info(f"Remove underscores: {self.underscore_checkbox.isChecked()}")
        logger.info(f"Recursive processing: {self.recursive_checkbox.isChecked()}")
        logger.info(f"Undesired tags: {undesired_tags}")
        logger.info(f"Prefix tags: {prefix_tags}")
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

    def check_model_exists(self, model_name):
        """Check if model files exist"""
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_info = ImageCaptioner.MODELS[model_name]
            model_dir = os.path.join(root_dir, model_info['path'])
            model_path = os.path.join(model_dir, "model.onnx")
            tags_path = os.path.join(model_dir, "selected_tags.csv")
            
            print(f"Checking model files:")  # Debug prints
            print(f"Model path: {model_path}")
            print(f"Tags path: {tags_path}")
            print(f"Model exists: {os.path.exists(model_path)}")
            print(f"Tags exist: {os.path.exists(tags_path)}")
            
            return os.path.exists(model_path) and os.path.exists(tags_path)
        except Exception as e:
            print(f"Error checking model existence: {str(e)}")  # Debug print
            return False
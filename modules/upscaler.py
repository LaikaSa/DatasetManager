import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import datetime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                              QFileDialog, QTabWidget, QHBoxLayout, 
                              QDoubleSpinBox, QLineEdit, QScrollArea, QListWidget)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
import requests
from modules.logger import setup_logger

logger = setup_logger()

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # Upsampling
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.body(feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat

        # Upsampling
        feat = self.lrelu(self.conv_up1(self.upsample(feat)))
        feat = self.lrelu(self.conv_up2(self.upsample(feat)))
        feat = self.lrelu(self.conv_hr(feat))
        feat = self.conv_last(feat)

        return feat

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch):
        super(RRDB, self).__init__()
        self.rdb1 = RDB(num_feat, num_grow_ch)
        self.rdb2 = RDB(num_feat, num_grow_ch)
        self.rdb3 = RDB(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class RDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch):
        super(RDB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class UpscaleWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()

    def __init__(self, input_paths, model_path, scale_factor):
        super().__init__()
        self.input_paths = input_paths if isinstance(input_paths, list) else [input_paths]
        self.model_path = model_path
        self.scale_factor = scale_factor
        self.is_running = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tile_size = 512
        self.tile_pad = 32

    def print_progress_bar(self, current, total, prefix='Progress:', length=50):
        filled_length = int(length * current / total)
        bar = '=' * filled_length + '-' * (length - filled_length)
        # Move up one line and clear it
        print('\033[1A\033[K', end='')
        print(f'{prefix} [{bar}] {current}/{total}')

    def load_model(self):
        if self.model is None:
            state_dict = torch.load(self.model_path, map_location=self.device)
            if 'params_ema' in state_dict:
                state_dict = state_dict['params_ema']

            # Count the number of RRDB blocks
            block_count = 0
            for key in state_dict.keys():
                if key.startswith('body.'):
                    parts = key.split('.')
                    if len(parts) > 2 and parts[1].isdigit():
                        block_num = int(parts[1])
                        block_count = max(block_count, block_num + 1)

            self.status.emit(f"Detected {block_count} blocks in model")
            
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=block_count,
                num_grow_ch=32
            )
            
            model.load_state_dict(state_dict)
            model.eval()
            self.model = model.to(self.device)
        return self.model

    def process_tile(self, tile, scale):
        # Convert tile to tensor
        tile_np = np.array(tile)
        tile_tensor = torch.from_numpy(tile_np).float() / 255.0
        tile_tensor = tile_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tile_tensor)
            if scale != 4:
                output = torch.nn.functional.interpolate(
                    output,
                    scale_factor=scale/4,
                    mode='bicubic',
                    align_corners=False
                )

        # Convert back to PIL Image
        output = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(output)

    def process_image(self, img_path):
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Calculate output size aligned to 8 pixels
            dest_w = int((img.width * self.scale_factor) // 8 * 8)
            dest_h = int((img.height * self.scale_factor) // 8 * 8)

            # Calculate tile dimensions
            tile_w = min(self.tile_size, img.width)
            tile_h = min(self.tile_size, img.height)

            # If image is small enough, process it directly
            if img.width <= self.tile_size and img.height <= self.tile_size:
                output_img = self.process_tile(img, self.scale_factor)
            else:
                # Calculate total tiles for progress
                total_tiles = ((img.height + tile_h - self.tile_pad - 1) // (tile_h - self.tile_pad)) * \
                                ((img.width + tile_w - self.tile_pad - 1) // (tile_w - self.tile_pad))
                current_tile = 0
                
                # Print initial tile progress bar
                print('')  # Empty line for progress bar
                
                # Process image in tiles
                output_img = Image.new('RGB', (dest_w, dest_h))
                for y in range(0, img.height, tile_h - self.tile_pad):
                    for x in range(0, img.width, tile_w - self.tile_pad):
                        if not self.is_running:
                            return False
                        
                        # Extract and process tile
                        right = min(x + tile_w, img.width)
                        bottom = min(y + tile_h, img.height)
                        tile = img.crop((x, y, right, bottom))
                        processed_tile = self.process_tile(tile, self.scale_factor)
                        
                        # Paste tile
                        paste_x = int(x * self.scale_factor)
                        paste_y = int(y * self.scale_factor)
                        output_img.paste(processed_tile, (paste_x, paste_y))

                        # Update tile progress
                        current_tile += 1
                        filled_length = int(50 * current_tile / total_tiles)
                        bar = '=' * filled_length + '-' * (50 - filled_length)
                        print(f'\033[1A\033[K' + f'Tiles: [{bar}] {current_tile}/{total_tiles}')
                
                print()  # New line after tiles complete

            # Save the result
            output_path = os.path.join(
                os.path.dirname(img_path),
                'upscaled',
                os.path.basename(img_path)
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_img.save(output_path)
            
            return True

        except Exception as e:
            self.status.emit(f"Error processing {img_path}: {str(e)}")
            return False

    def run(self):
        try:
            logger.info("Loading model...")
            self.load_model()
            
            total_files = len(self.input_paths)
            processed = 0
            
            print('')  # Empty line for progress bar
            self.print_progress_bar(0, total_files, prefix='Upscaling:')
            
            start_time = datetime.datetime.now()
            
            for img_path in self.input_paths:
                if not self.is_running:
                    logger.info("Process stopped by user")
                    self.status.emit("Process stopped by user")
                    break
                
                logger.info(f"Processing: {os.path.basename(img_path)}")
                if self.process_image(img_path):
                    processed += 1
                    self.print_progress_bar(processed, total_files, prefix='Upscaling:')
            
            end_time = datetime.datetime.now()
            duration = end_time - start_time
            
            print()  # New line after progress bar
            finish_msg = f"Finished processing {processed} images in {duration.total_seconds():.1f} seconds"
            logger.info(finish_msg)
            self.status.emit(finish_msg)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg)
            self.status.emit(error_msg)
        
        self.finished.emit()

    def stop(self):
        self.is_running = False

    def clear_gpu_memory(self):
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

class DragDropMixin:
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

class SingleImageTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Input selection
        input_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Select an image file or drag & drop anywhere")
        self.browse_btn = QPushButton("Browse")
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(self.browse_btn)
        
        self.browse_btn.clicked.connect(self.browse_file)
        self.input_path.textChanged.connect(lambda: self.parent.check_input(self.input_path.text()))
        
        layout.addLayout(input_layout)
        layout.addStretch()
        self.setLayout(layout)

    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.input_path.setText(path)

    def get_input_paths(self):
        path = self.input_path.text().strip()
        return [path] if path else []

    def handle_dropped_files(self, files):
        if files:
            self.input_path.setText(files[0])  # Take only the first file for single image tab

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:  # Only accept one file for single image tab
                path = urls[0].toLocalFile()
                if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    event.accept()
                    self.drop_label.setStyleSheet("""
                        QLabel {
                            border: 2px dashed #4CAF50;
                            border-radius: 5px;
                            padding: 20px;
                            background: #E8F5E9;
                        }
                    """)
                    return
        event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 20px;
                background: #f0f0f0;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            url = event.mimeData().urls()[0]
            path = url.toLocalFile()
            if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.input_path.setText(path)
                event.accept()
        
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 20px;
                background: #f0f0f0;
            }
        """)

class MultipleImagesTab(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.init_ui()
        self.selected_paths = []
        self.is_folder = False

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Directory input layout
        dir_layout = QHBoxLayout()
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Enter directory path or drag & drop anywhere")
        self.dir_input.textChanged.connect(self.handle_dir_input)
        dir_layout.addWidget(self.dir_input)
        
        # Buttons layout
        button_layout = QHBoxLayout()
        self.browse_files_btn = QPushButton("Select Files")
        self.browse_folder_btn = QPushButton("Select Folder")
        button_layout.addWidget(self.browse_files_btn)
        button_layout.addWidget(self.browse_folder_btn)
        
        # Selected files info
        self.info_label = QLabel("No files selected")
        self.info_label.setAlignment(Qt.AlignCenter)
        
        # File list widget
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(200)
        self.file_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #aaa;
                border-radius: 4px;
                background-color: #ffffff;
                padding: 5px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #eee;
            }
            QListWidget::item:selected {
                background-color: #e0e0e0;
                color: black;
            }
            QListWidget::item:hover {
                background-color: #f0f0f0;
            }
        """)
        
        # Connect buttons
        self.browse_files_btn.clicked.connect(self.browse_files)
        self.browse_folder_btn.clicked.connect(self.browse_folder)
        
        # Add widgets to layout
        layout.addLayout(dir_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.info_label)
        layout.addWidget(self.file_list)
        
        self.setLayout(layout)

    def handle_dir_input(self):
        path = self.dir_input.text().strip()
        if os.path.exists(path):
            if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.selected_paths = [path]
                self.is_folder = False
                self.update_file_list()
                self.parent.check_input(path)
            elif os.path.isdir(path):
                self.process_directory(path)

    def process_directory(self, directory):
        files = []
        # Only process files in the root directory, not in subfolders
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                files.append(file_path)
        
        if files:
            self.selected_paths = files
            self.is_folder = True
            self.update_file_list()
            self.parent.check_input(files[0])

    def update_file_list(self):
        self.file_list.clear()
        for path in self.selected_paths:
            item_text = f"{os.path.basename(path)} ({os.path.dirname(path)})"
            self.file_list.addItem(item_text)
        self.info_label.setText(f"{len(self.selected_paths)} files selected")

    def handle_dropped_files(self, files):
        if files:
            if len(files) == 1 and os.path.isdir(files[0]):
                self.dir_input.setText(files[0])
            self.selected_paths = files
            self.update_file_list()
            self.parent.check_input(files[0])

    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if files:
            self.selected_paths = files
            self.is_folder = False
            self.update_file_list()
            self.parent.check_input(files[0])

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.dir_input.setText(folder)
            self.process_directory(folder)

    def get_input_paths(self):
        return self.selected_paths

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
            self.drop_label.setStyleSheet("""
                QLabel {
                    border: 2px dashed #4CAF50;
                    border-radius: 5px;
                    padding: 20px;
                    background: #E8F5E9;
                }
            """)
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 20px;
                background: #f0f0f0;
            }
        """)

    def dropEvent(self, event: QDropEvent):
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                files.append(path)
            elif os.path.isdir(path):
                # If it's a folder, set it in the directory input
                self.dir_input.setText(path)
                # Process directory will handle adding the files (root folder only)
                self.process_directory(path)
                event.accept()
                return
        
        if files:
            self.selected_paths = files
            self.update_file_list()
            self.parent.check_input(files[0])
            event.accept()
        
        self.drop_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 20px;
                background: #f0f0f0;
            }
        """)

class UpscalerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.model_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Create sub-tabs
        self.tabs = QTabWidget()
        self.single_tab = SingleImageTab(self)
        self.multiple_tab = MultipleImagesTab(self)
        
        self.tabs.addTab(self.single_tab, "Single Image")
        self.tabs.addTab(self.multiple_tab, "Multiple Images")

        # Scale selection
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale factor:"))
        self.scale_spin = QDoubleSpinBox()  # Change to QDoubleSpinBox
        self.scale_spin.setRange(0.1, 16.0)  # Allow values from 0.1 to 16.0
        self.scale_spin.setValue(4.0)  # Default to 4.0
        self.scale_spin.setSingleStep(0.1)  # Set step to 0.1
        self.scale_spin.setDecimals(1)  # Show one decimal place
        self.scale_spin.setSuffix('x')
        scale_layout.addWidget(self.scale_spin)
        scale_layout.addStretch()

        # Model download section
        model_layout = QHBoxLayout()
        self.model_status = QLabel("Model not downloaded")
        self.download_btn = QPushButton("Download Model")
        model_layout.addWidget(self.model_status)
        model_layout.addWidget(self.download_btn)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Upscaling")
        self.stop_btn = QPushButton("Stop")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)

        # Status area
        self.status_area = QScrollArea()
        self.status_area.setWidgetResizable(True)
        self.status_text = QLabel()
        self.status_text.setAlignment(Qt.AlignTop)
        self.status_text.setWordWrap(True)
        self.status_area.setWidget(self.status_text)
        self.status_area.setMinimumHeight(200)

        # Add all layouts
        layout.addWidget(self.tabs)
        layout.addLayout(scale_layout)
        layout.addLayout(model_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_area)

        # Connect signals
        self.download_btn.clicked.connect(self.download_model)
        self.start_btn.clicked.connect(self.start_upscale)
        self.stop_btn.clicked.connect(self.stop_upscale)

        self.setLayout(layout)
        
        # Check if model exists
        self.check_model()

    def check_model(self):
        # Get project root directory (two levels up from the module file)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(root_dir, "models")
        model_filename = "RealESRGAN_x4plus_anime_6B.pth"
        self.model_path = os.path.join(model_dir, model_filename)
        
        if os.path.exists(self.model_path):
            self.model_status.setText("Model ready")
            self.download_btn.setEnabled(False)
            self.check_input("")  # Initialize with empty string
        else:
            self.model_status.setText("Model not downloaded")
            self.download_btn.setEnabled(True)
            self.start_btn.setEnabled(False)

    def check_input(self, path):
        if os.path.exists(self.model_path):
            current_tab = self.tabs.currentWidget()
            input_paths = current_tab.get_input_paths()
            self.start_btn.setEnabled(len(input_paths) > 0)
        else:
            self.start_btn.setEnabled(False)

    def download_model(self):
        self.download_btn.setEnabled(False)
        self.model_status.setText("Downloading model...")
        
        try:
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_dir = os.path.join(root_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(self.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            self.model_status.setText("Model ready")
            self.check_input("")
            
        except Exception as e:
            self.model_status.setText(f"Download failed: {str(e)}")
            self.download_btn.setEnabled(True)

    def start_upscale(self):
        if self.worker is not None and self.worker.isRunning():
            return

        current_tab = self.tabs.currentWidget()
        input_paths = current_tab.get_input_paths()
        
        if not input_paths:
            self.update_status("No input files selected")
            return

        scale_factor = self.scale_spin.value()

        self.worker = UpscaleWorker(
            input_paths, 
            self.model_path, 
            scale_factor
        )
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.upscale_finished)

        # Disable UI controls
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.tabs.setEnabled(False)
        self.scale_spin.setEnabled(False)
        self.status_text.setText("")
        
        self.worker.start()

    def stop_upscale(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.upscale_finished()

    def upscale_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self.input_path.setEnabled(True)
        self.scale_spin.setEnabled(True)  # Re-enable scale selection

    def stop_upscale(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.worker.clear_gpu_memory()  # Make sure to clear memory when stopping
            self.upscale_finished()

    def update_status(self, text):
        if "Finished" in text:
            formatted_text = f"<p style='color: green; font-weight: bold;'>{text}</p>"
        elif "Error" in text:
            formatted_text = f"<p style='color: red; font-weight: bold;'>{text}</p>"
        else:
            formatted_text = f"<p>{text}</p>"

        current_text = self.status_text.text()
        self.status_text.setText(current_text + formatted_text if current_text else formatted_text)

    def upscale_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.tabs.setEnabled(True)
        self.scale_spin.setEnabled(True)

    def stop_upscale(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.worker.clear_gpu_memory()  # Now this method exists
            self.upscale_finished()
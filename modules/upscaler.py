import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import datetime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                              QFileDialog, QProgressBar, QHBoxLayout, 
                              QSpinBox, QLineEdit, QScrollArea)
from PySide6.QtCore import Qt, QThread, Signal
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

    def __init__(self, input_path, model_path, scale_factor, is_folder=False):
        super().__init__()
        self.input_path = input_path
        self.model_path = model_path
        self.scale_factor = scale_factor
        self.is_folder = is_folder
        self.is_running = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tile_size = 512  # Default tile size
        self.tile_pad = 32    # Default tile padding

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
        def print_progress_bar(current, total, prefix='Progress:', length=50):
            filled_length = int(length * current / total)
            bar = '=' * filled_length + '-' * (length - filled_length)
            # Move up one line and clear it
            print('\033[1A\033[K', end='')
            print(f'{prefix} [{bar}] {current}/{total}')

        try:
            logger.info("Loading model...")
            self.load_model()
            
            if self.is_folder:
                image_files = []
                for file in os.listdir(self.input_path):
                    file_path = os.path.join(self.input_path, file)
                    if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_files.append(file_path)
                
                total_files = len(image_files)
                logger.info(f"Found {total_files} images to process")
                processed = 0
                
                # Print initial progress bar
                print('')  # Empty line for progress bar
                print_progress_bar(0, total_files, prefix='Upscaling:')
                
                start_time = datetime.datetime.now()
                
                for img_path in image_files:
                    if not self.is_running:
                        logger.info("Process stopped by user")
                        self.status.emit("Process stopped by user")
                        break
                    
                    logger.info(f"Processing: {os.path.basename(img_path)}")
                    if self.process_image(img_path):
                        processed += 1
                        print_progress_bar(processed, total_files, prefix='Upscaling:')
                
                end_time = datetime.datetime.now()
                duration = end_time - start_time
                
                print()  # New line after progress bar
                finish_msg = f"Finished processing {processed} images in {duration.total_seconds():.1f} seconds"
                logger.info(finish_msg)
                self.status.emit(finish_msg)
                
            else:
                logger.info(f"Processing single image: {os.path.basename(self.input_path)}")
                start_time = datetime.datetime.now()
                
                if self.process_image(self.input_path):
                    end_time = datetime.datetime.now()
                    duration = end_time - start_time
                    finish_msg = f"Finished processing image in {duration.total_seconds():.1f} seconds"
                    logger.info(finish_msg)
                    self.status.emit(finish_msg)
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg)
            self.status.emit(error_msg)
        
        self.finished.emit()

    def stop(self):
        self.is_running = False

class UpscalerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.model_path = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input selection
        input_layout = QHBoxLayout()
        self.input_path = QLineEdit()
        self.input_path.setPlaceholderText("Enter path to image or folder")
        self.browse_btn = QPushButton("Browse")
        input_layout.addWidget(self.input_path)
        input_layout.addWidget(self.browse_btn)

        # Scale selection
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale factor:"))
        self.scale_spin = QSpinBox()
        self.scale_spin.setRange(2, 16)  # Allow scaling from 2x to 16x
        self.scale_spin.setValue(4)  # Default to 4x
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

        # Add scale layout to main layout
        layout.addLayout(input_layout)
        layout.addLayout(scale_layout)  # Add the scale selection
        layout.addLayout(model_layout)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_area)

        # Connect signals
        self.browse_btn.clicked.connect(self.browse_input)
        self.download_btn.clicked.connect(self.download_model)
        self.start_btn.clicked.connect(self.start_upscale)
        self.stop_btn.clicked.connect(self.stop_upscale)
        self.input_path.textChanged.connect(self.check_input)

        self.setLayout(layout)
        
        # Check if model exists
        self.check_model()

    def check_model(self):
        # Create models directory in the same directory as the script
        model_filename = "RealESRGAN_x4plus_anime_6B.pth"
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        self.model_path = os.path.join(model_dir, model_filename)
        
        if os.path.exists(self.model_path):
            self.model_status.setText("Model ready")
            self.download_btn.setEnabled(False)
            self.check_input(self.input_path.text())
        else:
            self.model_status.setText("Model not downloaded")
            self.download_btn.setEnabled(True)
            self.start_btn.setEnabled(False)

    def download_model(self):
        self.download_btn.setEnabled(False)
        self.model_status.setText("Downloading model...")
        
        try:
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(model_dir, exist_ok=True)
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(self.model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            self.model_status.setText("Model ready")
            self.check_input(self.input_path.text())
            
        except Exception as e:
            self.model_status.setText(f"Download failed: {str(e)}")
            self.download_btn.setEnabled(True)

    def browse_input(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.input_path.setText(path)

    def check_input(self, path):
        path = path.strip()
        if os.path.exists(path) and os.path.exists(self.model_path):
            self.start_btn.setEnabled(True)
        else:
            self.start_btn.setEnabled(False)

    def start_upscale(self):
        if self.worker is not None and self.worker.isRunning():
            return

        input_path = self.input_path.text().strip()
        is_folder = os.path.isdir(input_path)
        scale_factor = self.scale_spin.value()

        self.worker = UpscaleWorker(
            input_path, 
            self.model_path, 
            scale_factor,
            is_folder
        )
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.upscale_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)
        self.input_path.setEnabled(False)
        self.scale_spin.setEnabled(False)  # Disable scale selection during processing
        self.status_text.setText("")
        
        self.worker.start()

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
            # Add some styling to the finish message
            formatted_text = f"<p style='color: green; font-weight: bold;'>{text}</p>"
        elif "Error" in text:
            # Add styling to error messages
            formatted_text = f"<p style='color: red; font-weight: bold;'>{text}</p>"
        else:
            formatted_text = f"<p>{text}</p>"

        current_text = self.status_text.text()
        self.status_text.setText(current_text + formatted_text if current_text else formatted_text)

    def upscale_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self.input_path.setEnabled(True)
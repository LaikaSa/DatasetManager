import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import datetime
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, QSpinBox,
                              QFileDialog, QTabWidget, QHBoxLayout, QCheckBox,
                              QDoubleSpinBox, QLineEdit, QScrollArea, QListWidget,
                              QListWidgetItem, QComboBox)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent
import requests
from modules import settings
from modules.logger import setup_logger
from modules.Upscaler.seedvr2_upscaler import (
    SeedVR2UpscaleWorker,
    SeedVR2DownloadWorker,
    are_seedvr2_models_downloaded,
)

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

class ModelDownloadWorker(QThread):
    """Downloads the RealESRGAN weights off the GUI thread into a .part file,
    then renames it into place, so an interrupted download can never leave a
    partial file behind that looks like a valid model."""

    status = Signal(str)
    finished_ok = Signal(bool)

    def __init__(self, url, dest_path):
        super().__init__()
        self.url = url
        self.dest_path = dest_path
        self.is_running = True

    def run(self):
        part_path = self.dest_path + '.part'
        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()
            total = int(response.headers.get('content-length', 0))
            downloaded = 0
            with open(part_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if not self.is_running:
                        break
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                    if total:
                        self.status.emit(
                            f"Downloading model... {downloaded >> 20} / {total >> 20} MB"
                        )
            if self.is_running:
                os.replace(part_path, self.dest_path)
                self.finished_ok.emit(True)
            else:
                self.finished_ok.emit(False)
        except Exception as e:
            self.status.emit(f"Download failed: {str(e)}")
            self.finished_ok.emit(False)
        finally:
            if os.path.exists(part_path):
                try:
                    os.remove(part_path)
                except OSError:
                    pass


class UpscaleWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()
    model_loaded = Signal(object)  # lets the tab cache the model between runs

    def __init__(self, input_paths, model_path, scale_factor, device=None,
                 model=None, owns_model=True):
        super().__init__()
        self.input_paths = input_paths if isinstance(input_paths, list) else [input_paths]
        self.model_path = model_path
        self.scale_factor = scale_factor
        self.is_running = True
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model  # preloaded (tab-cached) model, if any
        self.owns_model = owns_model  # False when the tab keeps the model
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
            self.model_loaded.emit(self.model)
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
            dest_w = max(8, int((img.width * self.scale_factor) // 8 * 8))
            dest_h = max(8, int((img.height * self.scale_factor) // 8 * 8))

            # Calculate tile dimensions
            tile_w = min(self.tile_size, img.width)
            tile_h = min(self.tile_size, img.height)

            # If image is small enough, process it directly
            if img.width <= self.tile_size and img.height <= self.tile_size:
                output_img = self.process_tile(img, self.scale_factor)
                if output_img.size != (dest_w, dest_h):
                    output_img = output_img.resize((dest_w, dest_h), Image.Resampling.LANCZOS)
            else:
                # Tiled path: accumulate into a float buffer and blend the
                # overlap region with linear ramps (plain paste() left visible
                # seams at tile borders).
                out = np.zeros((dest_h, dest_w, 3), dtype=np.float32)
                weight = np.zeros((dest_h, dest_w), dtype=np.float32)

                step_w = max(1, tile_w - self.tile_pad)
                step_h = max(1, tile_h - self.tile_pad)
                total_tiles = ((img.height - 1) // step_h + 1) * ((img.width - 1) // step_w + 1)
                current_tile = 0

                pad_w_out = int(self.tile_pad * self.scale_factor)
                pad_h_out = int(self.tile_pad * self.scale_factor)

                # Print initial tile progress bar
                print('')  # Empty line for progress bar

                for y in range(0, img.height, step_h):
                    for x in range(0, img.width, step_w):
                        if not self.is_running:
                            return False

                        # Extract and process tile
                        right = min(x + tile_w, img.width)
                        bottom = min(y + tile_h, img.height)
                        tile = img.crop((x, y, right, bottom))
                        processed_tile = self.process_tile(tile, self.scale_factor)

                        # Target region in output space, at its exact size so
                        # non-integer scale factors can't leave 1px gaps.
                        tx = int(round(x * self.scale_factor))
                        ty = int(round(y * self.scale_factor))
                        tw = min(int(round((right - x) * self.scale_factor)), dest_w - tx)
                        th = min(int(round((bottom - y) * self.scale_factor)), dest_h - ty)
                        if processed_tile.size != (tw, th):
                            processed_tile = processed_tile.resize((tw, th), Image.Resampling.LANCZOS)

                        # Linear ramps over the overlap, disabled at the
                        # image borders where there is no neighbour tile.
                        wx = np.ones(tw, dtype=np.float32)
                        if x > 0 and pad_w_out > 0:
                            r = min(pad_w_out, tw)
                            wx[:r] = np.linspace(0.0, 1.0, r, dtype=np.float32)
                        if right < img.width and pad_w_out > 0:
                            r = min(pad_w_out, tw)
                            wx[tw - r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
                        wy = np.ones(th, dtype=np.float32)
                        if y > 0 and pad_h_out > 0:
                            r = min(pad_h_out, th)
                            wy[:r] = np.linspace(0.0, 1.0, r, dtype=np.float32)
                        if bottom < img.height and pad_h_out > 0:
                            r = min(pad_h_out, th)
                            wy[th - r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)

                        w2 = wy[:, None] * wx[None, :]
                        out[ty:ty + th, tx:tx + tw] += np.asarray(
                            processed_tile, dtype=np.float32) * w2[:, :, None]
                        weight[ty:ty + th, tx:tx + tw] += w2

                        # Update tile progress
                        current_tile += 1
                        filled_length = int(50 * current_tile / total_tiles)
                        bar = '=' * filled_length + '-' * (50 - filled_length)
                        print(f'\033[1A\033[K' + f'Tiles: [{bar}] {current_tile}/{total_tiles}')

                print()  # New line after tiles complete

                np.maximum(weight, 1e-6, out=weight)  # guard any uncovered pixel
                out /= weight[:, :, None]
                output_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

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
            self.status.emit(f"Using device: {self.device}")
            logger.info(f"Loading model on {self.device}...")
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
        # Only free the model when this worker owns it; tab-cached models
        # are kept so the next run doesn't pay the load cost again.
        if self.owns_model and self.model is not None:
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
            
class InputWidget(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.selected_paths = []
        self._size_cache = {}  # path -> (w, h); avoid re-opening files on every filter pass
        self.setAcceptDrops(True)
        self.init_ui()

    def _get_size(self, p):
        if p not in self._size_cache:
            with Image.open(p) as img:
                self._size_cache[p] = img.size
        return self._size_cache[p]

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Path input row
        path_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter image path or folder path, or drag & drop here...")
        self.browse_file_btn = QPushButton("Browse File")
        self.browse_folder_btn = QPushButton("Browse Folder")
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_file_btn)
        path_layout.addWidget(self.browse_folder_btn)

        # Info label
        self.info_label = QLabel("No input selected")
        self.info_label.setAlignment(Qt.AlignCenter)

        # File list
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(180)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_context_menu)
        self.file_list.setUniformItemSizes(True)

        self.browse_file_btn.clicked.connect(self.browse_file)
        self.browse_folder_btn.clicked.connect(self.browse_folder)
        self.path_input.textChanged.connect(self.on_path_changed)

        layout.addLayout(path_layout)
        layout.addWidget(self.info_label)
        layout.addWidget(self.file_list)

    # ── path typing / pasting ──────────────────────────────────────────────
    def on_path_changed(self, text):
        text = text.strip()
        if not text:
            self.selected_paths = []
            self.refresh_list()
            self.parent.check_input()
            return
        if os.path.isfile(text) and text.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            self.selected_paths = [text]
            self.refresh_list()
            self.parent.check_input()
        elif os.path.isdir(text):
            self.load_folder(text)

    # ── browse buttons ─────────────────────────────────────────────────────
    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.path_input.setText(path)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.path_input.setText(folder)

    # ── folder loading ─────────────────────────────────────────────────────
    def load_folder(self, folder):
        files = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        self.selected_paths = sorted(files)
        self._size_cache.clear()
        self.refresh_list()
        self.parent.check_input()

    # ── list display ───────────────────────────────────────────────────────
    def refresh_list(self):
        self.file_list.clear()
        paths = self.selected_paths

        # Apply resolution filter if active
        if self.parent.resolution_check.isChecked():
            max_res = self.parent.resolution_spin.value()
            paths = self.filter_by_resolution(paths, max_res)
            self.info_label.setText(
                f"{len(paths)}/{len(self.selected_paths)} files "
                f"(under {max_res} px)"
            )
        else:
            count = len(paths)
            if count == 1:
                self.info_label.setText(f"1 file selected: {os.path.basename(paths[0])}")
            else:
                self.info_label.setText(f"{count} files selected")

        for p in paths:
            try:
                w, h = self._get_size(p)
                res_str = f"{w}×{h}"
            except Exception:
                res_str = "?"

            # Build a row widget: filename left, resolution right
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(0)

            name_label = QLabel(os.path.basename(p))
            name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            res_label = QLabel(f"[{res_str}]")
            res_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            res_label.setStyleSheet("color: #555555;")

            row_layout.addWidget(name_label)
            row_layout.addStretch()
            row_layout.addWidget(res_label)

            item = QListWidgetItem(self.file_list)
            item.setSizeHint(row_widget.sizeHint())
            self.file_list.addItem(item)
            self.file_list.setItemWidget(item, row_widget)

    def filter_by_resolution(self, paths, max_res):
        out = []
        for p in paths:
            try:
                w, h = self._get_size(p)
                if w < max_res and h < max_res:
                    out.append(p)
            except Exception:
                pass
        return out

    # ── called by parent when resolution filter changes ────────────────────
    def update_list(self):
        self.refresh_list()

    def show_context_menu(self, pos):
        from PySide6.QtWidgets import QMenu
        item = self.file_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        open_action = menu.addAction("Open in Folder")
        action = menu.exec(self.file_list.mapToGlobal(pos))
        if action == open_action:
            self.open_in_folder(item)

    def open_in_folder(self, item):
        import ctypes
        row = self.file_list.row(item)
        paths = self.selected_paths
        if self.parent.resolution_check.isChecked():
            paths = self.filter_by_resolution(paths, self.parent.resolution_spin.value())
        if row < len(paths):
            normalized = os.path.normpath(paths[row])
            shell32 = ctypes.windll.shell32
            shell32.ILCreateFromPathW.argtypes = [ctypes.c_wchar_p]
            shell32.ILCreateFromPathW.restype = ctypes.c_void_p
            shell32.SHOpenFolderAndSelectItems.argtypes = [
                ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_ulong
            ]
            shell32.SHOpenFolderAndSelectItems.restype = ctypes.HRESULT
            shell32.ILFree.argtypes = [ctypes.c_void_p]
            shell32.ILFree.restype = None
            pidl = shell32.ILCreateFromPathW(normalized)
            if pidl:
                shell32.SHOpenFolderAndSelectItems(pidl, 0, None, 0)
                shell32.ILFree(pidl)

    # ── what the worker actually processes ────────────────────────────────
    def get_input_paths(self):
        paths = self.selected_paths
        if self.parent.resolution_check.isChecked():
            paths = self.filter_by_resolution(paths, self.parent.resolution_spin.value())
        return paths

    # ── drag & drop ────────────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            return

        # Single folder drop
        first = urls[0].toLocalFile()
        if len(urls) == 1 and os.path.isdir(first):
            self.path_input.setText(first)
            event.accept()
            return

        # One or more files
        files = [
            u.toLocalFile() for u in urls
            if os.path.isfile(u.toLocalFile())
            and u.toLocalFile().lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]
        if files:
            self._size_cache.clear()
            self.selected_paths = files
            # Show the first path (or parent folder) in the text box
            if len(files) == 1:
                self.path_input.blockSignals(True)
                self.path_input.setText(files[0])
                self.path_input.blockSignals(False)
            else:
                self.path_input.blockSignals(True)
                self.path_input.setText(os.path.dirname(files[0]))
                self.path_input.blockSignals(False)
            self.refresh_list()
            self.parent.check_input()
            event.accept()

class UpscalerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.model_path = None
        # Cached RealESRGAN model so repeated runs skip torch.load + device
        # transfer. Keyed by (model_path, device).
        self._cached_model = None
        self._cached_model_key = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Single unified input widget
        self.input_widget = InputWidget(self)
        layout.addWidget(self.input_widget)

        # Scale + resolution controls
        controls_layout = QHBoxLayout()

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale factor:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 16.0)
        self.scale_spin.setValue(4.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setDecimals(1)
        self.scale_spin.setSuffix('x')
        scale_layout.addWidget(self.scale_spin)

        resolution_layout = QHBoxLayout()
        self.resolution_check = QCheckBox("Only upscale images smaller than:")
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(1, 10000)
        self.resolution_spin.setValue(1024)
        self.resolution_spin.setSuffix(' px')
        self.resolution_spin.setEnabled(False)
        self.resolution_check.stateChanged.connect(self.toggle_resolution_filter)
        self.resolution_spin.valueChanged.connect(self.input_widget.update_list)
        resolution_layout.addWidget(self.resolution_check)
        resolution_layout.addWidget(self.resolution_spin)

        controls_layout.addLayout(scale_layout)
        controls_layout.addSpacing(20)
        controls_layout.addLayout(resolution_layout)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Model row (choose between the default RealESRGAN and SeedVR2)
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("Default (RealESRGAN anime 6B)", "realesrgan")
        self.model_combo.addItem("SeedVR2 (3B fp16)", "seedvr2")
        # NOTE: connected after seedvr2_options exists (addItem emits indexChanged)
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        self.model_status = QLabel("Model not downloaded")
        self.download_btn = QPushButton("Download Model")
        self.download_btn.clicked.connect(self.download_model)
        model_layout.addWidget(self.model_status)
        model_layout.addWidget(self.download_btn)
        layout.addLayout(model_layout)

        # SeedVR2-only options (hidden when the default model is selected)
        self.seedvr2_options = QWidget()
        seedvr2_layout = QHBoxLayout(self.seedvr2_options)
        seedvr2_layout.setContentsMargins(0, 0, 0, 0)
        seedvr2_layout.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(42)
        seedvr2_layout.addWidget(self.seed_spin)

        seedvr2_layout.addSpacing(20)
        seedvr2_layout.addWidget(QLabel("Color correction:"))
        self.color_combo = QComboBox()
        # LAB is the default: full perceptual color matching with detail preservation
        # (wavelet reconstruction + LAB histogram transfer).
        self.color_combo.addItem("LAB (perceptual, recommended)", "lab")
        self.color_combo.addItem("Wavelet", "wavelet")
        self.color_combo.addItem("Wavelet adaptive", "wavelet_adaptive")
        self.color_combo.addItem("HSV", "hsv")
        self.color_combo.addItem("AdaIN", "adain")
        self.color_combo.addItem("None", "none")
        seedvr2_layout.addWidget(self.color_combo)

        seedvr2_layout.addSpacing(20)
        self.tile_check = QCheckBox("Tile VAE (low VRAM)")
        self.tile_check.setChecked(True)
        seedvr2_layout.addWidget(self.tile_check)
        seedvr2_layout.addStretch()
        self.seedvr2_options.setVisible(False)
        layout.addWidget(self.seedvr2_options)

        # Now safe to react to model changes (all widgets exist)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        # Action buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Upscaling")
        self.stop_btn = QPushButton("Stop")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_upscale)
        self.stop_btn.clicked.connect(self.stop_upscale)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)

        # Status area
        self.status_area = QScrollArea()
        self.status_area.setWidgetResizable(True)
        self.status_text = QLabel()
        self.status_text.setAlignment(Qt.AlignTop)
        self.status_text.setWordWrap(True)
        self.status_area.setWidget(self.status_text)
        self.status_area.setMinimumHeight(200)
        layout.addWidget(self.status_area)

        self.check_model()

    def _clear_cached_model(self):
        if self._cached_model is not None:
            del self._cached_model
            self._cached_model = None
            self._cached_model_key = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def toggle_resolution_filter(self, state):
        self.resolution_spin.setEnabled(bool(state))
        self.input_widget.update_list()

    def selected_model(self):
        return self.model_combo.currentData()

    def on_model_changed(self, *_args):
        """Refresh the model row + SeedVR2 options whenever the model changes."""
        self.seedvr2_options.setVisible(self.selected_model() == "seedvr2")
        self.refresh_model_row()

    def refresh_model_row(self):
        """Recompute the ready-state of every model and update the UI."""
        self.model_ready = {}
        self.model_ready["realesrgan"] = os.path.exists(self.model_path)
        self.model_ready["seedvr2"] = are_seedvr2_models_downloaded()

        ready = self.model_ready.get(self.selected_model(), False)
        self.model_status.setText("Model ready" if ready else "Model not downloaded")
        self.download_btn.setEnabled(not ready and not self._downloading)
        self.check_input()

    def check_model(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(root_dir, "models")
        self.model_path = os.path.join(model_dir, "RealESRGAN_x4plus_anime_6B.pth")
        self.model_ready = {}
        self._downloading = False
        self.refresh_model_row()

    def check_input(self):
        ready = self.model_ready.get(self.selected_model(), False)
        self.start_btn.setEnabled(ready and len(self.input_widget.get_input_paths()) > 0)

    def download_model(self):
        model_type = self.selected_model()
        if model_type == "seedvr2":
            self._downloading = True
            self.download_btn.setEnabled(False)
            self.model_status.setText("Downloading SeedVR2 (~7GB)...")
            self.seedvr2_download = SeedVR2DownloadWorker()
            self.seedvr2_download.status.connect(self.model_status.setText)
            self.seedvr2_download.finished_ok.connect(self._seedvr2_download_done)
            self.seedvr2_download.start()
            return

        # Default RealESRGAN model (small, single file) - downloaded in a
        # worker thread so the GUI stays responsive during the transfer.
        self._downloading = True
        self.download_btn.setEnabled(False)
        self.model_status.setText("Downloading model...")
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.download_worker = ModelDownloadWorker(url, self.model_path)
        self.download_worker.status.connect(self.model_status.setText)
        self.download_worker.finished_ok.connect(self._realesrgan_download_done)
        self.download_worker.start()

    def _realesrgan_download_done(self, ok):
        self._downloading = False
        self.refresh_model_row()
        if not ok:
            self.model_status.setText("Download failed")
            self.download_btn.setEnabled(True)

    def _seedvr2_download_done(self, ok):
        self._downloading = False
        self.refresh_model_row()
        if not ok:
            self.model_status.setText("Download failed")
            self.download_btn.setEnabled(True)

    def start_upscale(self):
        if self.worker is not None and self.worker.isRunning():
            return
        input_paths = self.input_widget.get_input_paths()
        if not input_paths:
            self.update_status("No input files selected")
            return

        device = settings.to_torch_device(settings.get_selected_device_id())

        if self.selected_model() == "seedvr2":
            self.worker = SeedVR2UpscaleWorker(
                input_paths,
                self.scale_spin.value(),
                device=device,
                seed=self.seed_spin.value(),
                color_correction=self.color_combo.currentData(),
                tile_vae=self.tile_check.isChecked(),
            )
        else:
            # Reuse the cached model when model + device are unchanged
            cache_key = (self.model_path, device)
            if self._cached_model_key != cache_key:
                self._clear_cached_model()
            use_cached = self._cached_model_key == cache_key
            self.worker = UpscaleWorker(
                input_paths, self.model_path, self.scale_spin.value(),
                device=device, model=self._cached_model,
                owns_model=not use_cached,
            )
            self.worker.model_loaded.connect(self._on_model_loaded)

    def _on_model_loaded(self, model):
        """Cache a freshly loaded model for the next run."""
        device = settings.to_torch_device(settings.get_selected_device_id())
        self._cached_model = model
        self._cached_model_key = (self.model_path, device)

        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.upscale_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.input_widget.setEnabled(False)
        self.scale_spin.setEnabled(False)
        self.resolution_check.setEnabled(False)
        self.resolution_spin.setEnabled(False)
        self.seedvr2_options.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.status_text.setText("")
        self.worker.start()

    def stop_upscale(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.worker.clear_gpu_memory()
            self.upscale_finished()

    def upscale_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.input_widget.setEnabled(True)
        self.scale_spin.setEnabled(True)
        self.resolution_check.setEnabled(True)
        self.seedvr2_options.setEnabled(True)
        self.model_combo.setEnabled(True)
        if self.resolution_check.isChecked():
            self.resolution_spin.setEnabled(True)
        self.check_input()

    def update_status(self, text):
        if "Finished" in text:
            formatted = f"<p style='color:green;font-weight:bold;'>{text}</p>"
        elif "Error" in text:
            formatted = f"<p style='color:red;font-weight:bold;'>{text}</p>"
        else:
            formatted = f"<p>{text}</p>"
        current = self.status_text.text()
        self.status_text.setText(current + formatted if current else formatted)
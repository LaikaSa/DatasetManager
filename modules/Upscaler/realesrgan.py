"""RealESRGAN model definitions and upscaling workers.

The torch model (RRDBNet) and the two QThread workers (model download,
upscaling) live here so the tab module stays pure UI.
"""
import os
import math
import datetime

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from PySide6.QtCore import QThread, Signal
import requests

from modules.logger import setup_logger
from modules.utils import print_progress_bar

logger = setup_logger()

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        nb = num_block
        nf = num_feat
        ng = num_grow_ch
        conv = nn.Conv2d

        self.conv_fn = conv
        self.conv_body = nn.ModuleList()
        for i in range(nb):
            self.conv_body.append(RRDB(nf, ng))

        self.conv_head = conv(num_in_ch, nf, 3, 1, 1)
        self.conv_tail = conv(nf, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_head(x)
        for i in range(0, len(self.conv_body)):
            feat = self.conv_body[i](feat)
        out = self.conv_tail(self.lrelu(feat))
        return out

class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv5(self.lrelu(self.conv4(self.lrelu(self.conv3(self.lrelu(self.conv2(self.lrelu(self.conv1(x)))))))))
        return out * 0.2 + x

class RDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RDB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))
        x4 = self.lrelu(self.conv4(x3))
        x5 = self.conv5(x4)
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

    def __init__(self, input_paths, model_path, scale_factor, device=None,
                 min_size=0):
        super().__init__()
        self.input_paths = input_paths if isinstance(input_paths, list) else [input_paths]
        self.model_path = model_path
        self.scale_factor = scale_factor
        self.min_size = int(min_size)  # >0 -> auto per-image scale factor
        self.is_running = True
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tile_size = 512
        self.tile_pad = 32

    def _effective_scale(self, w, h):
        """Scale factor for one image.

        In auto mode (min_size > 0) this is the *smallest 0.1 step*
        (1.0, 1.1, 1.2, ...) that brings the longest side to at least
        min_size. The factor is applied strictly to both sides (no pixel
        alignment rounding), so the aspect ratio is kept exactly. Otherwise
        the user's fixed factor is used.
        """
        if self.min_size > 0:
            steps = math.ceil(self.min_size * 10 / max(w, h) - 1e-9)
            return max(1.0, steps / 10.0)
        return self.scale_factor

    def load_model(self):
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

            # Per-image scale factor (auto mode targets the minimum size)
            scale = self._effective_scale(img.width, img.height)

            # Apply the factor strictly to both sides (nearest pixel) so the
            # aspect ratio is preserved with no alignment rounding.
            dest_w = max(8, int(round(img.width * scale)))
            dest_h = max(8, int(round(img.height * scale)))

            # Calculate tile dimensions
            tile_w = min(self.tile_size, img.width)
            tile_h = min(self.tile_size, img.height)

            # If image is small enough, process it directly
            if img.width <= self.tile_size and img.height <= self.tile_size:
                output_img = self.process_tile(img, scale)
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

                pad_w_out = int(self.tile_pad * scale)
                pad_h_out = int(self.tile_pad * scale)

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
                        processed_tile = self.process_tile(tile, scale)

                        # Target region in output space, at its exact size so
                        # non-integer scale factors can't leave 1px gaps.
                        tx = int(round(x * scale))
                        ty = int(round(y * scale))
                        tw = min(int(round((right - x) * scale)), dest_w - tx)
                        th = min(int(round((bottom - y) * scale)), dest_h - ty)
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
                        print_progress_bar(current_tile, total_tiles, prefix='Tiles:')

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
            print_progress_bar(0, total_files, prefix='Upscaling:')
            
            start_time = datetime.datetime.now()
            
            for img_path in self.input_paths:
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
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg)
            self.status.emit(error_msg)
        finally:
            # Free the model + CUDA cache once the WHOLE batch is done
            # (or stopped) - never per image - so VRAM is returned as soon
            # as the operation ends.
            self.clear_gpu_memory()
            self.status.emit("GPU memory freed")

        self.finished.emit()

    def stop(self):
        self.is_running = False

    def clear_gpu_memory(self):
        """Free the model and the CUDA cache. Called once per run, after
        the whole batch finishes (or is stopped)."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
"""RealESRGAN upscaling workers (thin Qt shells).

The Qt-free pipeline (RRDBNet + RealESRGANUpscaler + download) lives in
engine.py; this module only keeps the QThread wrappers the tab drives.
"""
import os
import datetime

import torch
from PIL import Image
from PySide6.QtCore import QThread, Signal

from modules.logger import setup_logger
from modules.utils import print_progress_bar
from .engine import RealESRGANUpscaler, download_model

logger = setup_logger()


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
        # The engine owns the .part + rename logic and the canonical
        # RealESRGAN URL; the url ctor arg is kept for constructor parity.
        ok = download_model("realesrgan", self.dest_path,
                            status_cb=self.status.emit,
                            stop_check=lambda: not self.is_running)
        self.finished_ok.emit(ok)

    def stop(self):
        self.is_running = False


class UpscaleWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()

    def __init__(self, input_paths, model_path, scale_factor, device=None,
                 min_size=0):
        super().__init__()
        self.input_paths = input_paths if isinstance(input_paths, list) else [input_paths]
        self.is_running = True
        self.upscaler = RealESRGANUpscaler(
            model_path, device=device,
            scale_factor=scale_factor, min_size=min_size,
        )

    def run(self):
        try:
            self.status.emit(f"Using device: {self.upscaler.device}")
            logger.info(f"Loading model on {self.upscaler.device}...")
            self.upscaler.load_model(status_cb=self.status.emit)

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
                if self.upscaler.process_image(
                        img_path,
                        stop_check=lambda: not self.is_running,
                        tile_progress_cb=lambda c, t: print_progress_bar(
                            c, t, prefix='Tiles:'),
                        error_cb=self.status.emit):
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
            self.upscaler.clear_gpu_memory()
            self.status.emit("GPU memory freed")

        self.finished.emit()

    def stop(self):
        self.is_running = False
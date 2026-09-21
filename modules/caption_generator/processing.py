from PySide6.QtCore import QThread, Signal
import numpy as np
import os
import threading
import torch
from PIL import Image
from .utils import ProgressBar
from .data_loader import ImageLoadingPrepDataset, collate_fn_remove_corrupted
from modules.logger import setup_logger
logger = setup_logger()

class CaptionGeneratorThread(QThread):
    caption_generated = Signal(str, str)
    process_completed = Signal()
    error_occurred = Signal(str)
    stopped = Signal()

    def __init__(self, captioner, folder_path, include_rating=False, 
                 remove_underscore=True, recursive=False, 
                 undesired_tags=None, prefix_tags=None, append_tags=False,
                 thresh=0.35, general_threshold=0.35, character_threshold=0.35,
                 batch_size=1, worker_count=2):
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
        self.batch_size = batch_size
        self.worker_count = worker_count
        self._stop_event = threading.Event()

    def request_stop(self):
        self._stop_event.set()

    def _should_stop(self):
        return self._stop_event.is_set()

    def _prepare_image(self, image_path):
        """Decode + preprocess one image (thread-safe, no Qt objects)."""
        try:
            return self.captioner.prepare_image(image_path)
        except Exception as e:
            logger.error(f"Could not prepare {image_path}: {e}")
            return None

    def _apply_append(self, image_path, txt_path, caption):
        """Append mode: merge new tags into the existing caption file
        (existing tags first, de-duplicated)."""
        if not self.append_tags or not os.path.exists(txt_path):
            return caption
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                existing_content = f.read().strip()

            existing_tags = [tag.strip() for tag in existing_content.split(',') if tag.strip()]
            new_tags = [tag.strip() for tag in caption.split(',') if tag.strip()]

            if self.captioner.debug_mode:
                logger.debug(f"\nAppending tags for {os.path.basename(image_path)}:")
                logger.debug(f"  Existing tags: {existing_tags}")
                logger.debug(f"  New tags: {new_tags}")

            combined_tags = []
            seen = set()
            for tag in existing_tags + new_tags:
                if tag not in seen:
                    combined_tags.append(tag)
                    seen.add(tag)

            if self.captioner.debug_mode:
                logger.debug(f"  Final combined tags: {combined_tags}")

            return ', '.join(combined_tags)

        except Exception as e:
            logger.error(f"Error reading existing caption for {image_path}: {e}")
            return caption

    def run(self):
        try:
            image_files = self.get_image_files(self.folder_path)
            total_files = len(image_files)

            # Basic info always shown
            logger.info(f"Starting caption generation for {len(image_files)} images")
            progress = ProgressBar(total_files, prefix='Processing: ')

            from concurrent.futures import ThreadPoolExecutor
            prep_workers = max(1, self.worker_count)
            batch_size = max(1, self.batch_size)

            completed = 0
            for batch_start in range(0, total_files, batch_size):
                if self._should_stop():
                    logger.info("Caption generation stopped by user")
                    self.stopped.emit()
                    return

                batch_paths = image_files[batch_start: batch_start + batch_size]

                try:
                    # Prepare images in parallel - decoding is the I/O-bound
                    # part, and the data-loader worker count is finally used
                    # for something.
                    if len(batch_paths) == 1:
                        prepared = [self._prepare_image(batch_paths[0])]
                    else:
                        with ThreadPoolExecutor(max_workers=min(prep_workers, len(batch_paths))) as pool:
                            prepared = list(pool.map(self._prepare_image, batch_paths))

                    ok_idx = [i for i, im in enumerate(prepared) if im is not None]

                    # One batched inference for every prepared image in the batch
                    preds = None
                    if ok_idx:
                        preds = self.captioner.predict_batch([prepared[i] for i in ok_idx])

                    for j, i in enumerate(ok_idx):
                        image_path = batch_paths[i]

                        caption = self.captioner.caption_from_preds(
                            np.asarray(preds[j], dtype=float),
                            self.general_threshold,
                            self.character_threshold,
                            self.remove_underscore,
                            self.undesired_tags,
                            self.prefix_tags,
                            ", ",
                            self.include_rating,
                        )

                        # Log generated tags only in debug mode
                        if self.captioner.debug_mode and caption is not None:
                            logger.debug(f"\nGenerated caption for {os.path.basename(image_path)}:")
                            logger.debug(f"  Tags: {caption.split(', ')}")

                        # A None caption means inference failed for this image
                        # - skip the write so we never store an error string.
                        if caption is None:
                            self.error_occurred.emit(f"Caption generation failed for {image_path}")
                        else:
                            txt_path = os.path.splitext(image_path)[0] + '.txt'
                            caption = self._apply_append(image_path, txt_path, caption)
                            with open(txt_path, 'w', encoding='utf-8') as f:
                                f.write(caption + '\n')
                            self.caption_generated.emit(image_path, caption)

                        completed += 1
                        progress.update(completed)

                except Exception as e:
                    logger.error(f"Error processing batch starting at {batch_paths[0]}: {str(e)}")
                    self.error_occurred.emit(f"Error processing batch: {str(e)}")
                    completed += len(batch_paths)

            # Basic completion info always shown
            logger.info("Caption generation completed")
            self.process_completed.emit()

        except Exception as e:
            error_msg = f"Process error: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)


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


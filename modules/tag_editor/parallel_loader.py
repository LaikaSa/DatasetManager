from multiprocessing import Pool
from pathlib import Path
import os

# Cap math-library threads *before* numpy loads: workers only do
# thumbnail-sized numpy work, and OpenBLAS's default per-core thread pools
# exhaust the Windows paging file when cpu_count() workers spawn at once.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
from PIL import Image
import time

from modules.utils import IMAGE_EXTENSIONS
from modules.logger import setup_logger

logger = setup_logger()

def process_single_image(args):
    """Process a single image and its tags (runs in worker process)"""
    image_path, thumbnail_size = args
    try:
        # Use PIL instead of QImage for parallel processing
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate new size maintaining aspect ratio
            ratio = min(thumbnail_size / img.width, thumbnail_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            
            # Resize image
            thumbnail = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to numpy array for transfer
            img_array = np.array(thumbnail)

        # Read tags (preserve order, drop duplicates)
        tag_path = str(Path(image_path).with_suffix('.txt'))
        tags = []
        if os.path.exists(tag_path):
            with open(tag_path, 'r', encoding='utf-8') as f:
                seen = set()
                for tag in f.read().split(','):
                    tag = tag.strip().lower()
                    if tag and tag not in seen:
                        seen.add(tag)
                        tags.append(tag)

        return {
            'path': image_path,
            'array': img_array,
            'tags': tags
        }
    except Exception as e:
        logger.error("Error processing %s: %s", image_path, e)
        return None

class ParallelLoader:
    def __init__(self, thumbnail_size=150):
        self.thumbnail_size = thumbnail_size
        self.pool = None

    def start_pool(self):
        if self.pool is None:
            # One Pool per ParallelLoader instance. Each LoadingThread owns
            # its loader and calls stop_pool() when it finishes, so workers
            # always exit from the thread that created the pool.
            #
            # Capped at 4: on Windows each spawned worker re-imports the whole
            # app (run.py -> main -> PySide6) plus numpy/OpenBLAS, so
            # cpu_count() workers can exhaust the paging file (WinError 1455).
            self.pool = Pool(processes=min(4, os.cpu_count() or 1))

    def stop_pool(self):
        if self.pool:
            self.pool.close()
            self.pool.join()
            self.pool = None

    @staticmethod
    def array_to_pixmap(arr):
        """Build a QPixmap from an RGB numpy array on the GUI thread.

        (QPixmap is a GUI object - building it on a QThread/process worker
        isn't guaranteed. The .copy() makes the QImage own its data, since
        the numpy array is only alive inside the worker.)
        """
        from PySide6.QtGui import QImage, QPixmap
        height, width, channel = arr.shape
        q_img = QImage(
            arr.data, width, height, 3 * width, QImage.Format_RGB888
        ).copy()
        return QPixmap.fromImage(q_img)

    def load_images(self, directory, recursive=False):
        """Load images and tags in parallel"""
        try:
            logger.debug("Starting parallel loading process...")
            self.start_pool()
            
            # Get all image files
            file_scan_start = time.time()
            valid_extensions = IMAGE_EXTENSIONS
            image_paths = [
                str(p) for p in (Path(directory).rglob('*.*') if recursive
                                 else Path(directory).glob('*.*'))
                if p.suffix.lower() in valid_extensions
            ]
            file_scan_end = time.time()
            logger.debug("File scanning time: %.2f seconds", file_scan_end - file_scan_start)
            logger.info("Found %d images", len(image_paths))
            
            # Prepare arguments
            args = [(path, self.thumbnail_size) for path in image_paths]

            # Process images in parallel (chunksize keeps worker<->main IPC
            # sane on large folders instead of one task per message)
            parallel_start = time.time()
            chunksize = max(1, len(args) // (self.pool._processes * 4))
            results = self.pool.map(process_single_image, args, chunksize=chunksize)
            parallel_end = time.time()
            logger.debug("Parallel processing time: %.2f seconds", parallel_end - parallel_start)
            
            # Convert results to QPixmap in main thread
            conversion_start = time.time()
            processed_images = []
            successful = 0
            failed = 0
            
            for result in results:
                if result is None:
                    failed += 1
                    continue

                # NOTE: no QPixmap is created here. This method runs inside a
                # QThread, so GUI objects are built on the main thread by the
                # caller (ParallelLoader.array_to_pixmap).
                processed_images.append({
                    'path': result['path'],
                    'array': result['array'],
                    'tags': result['tags']
                })
                successful += 1

            conversion_end = time.time()
            logger.debug("Parallel summary: %d successful, %d failed, %.2f seconds",
                         successful, failed, conversion_end - conversion_start)

            return processed_images

        except Exception as e:
            logger.error("Error in parallel loading: %s", e)
            return []

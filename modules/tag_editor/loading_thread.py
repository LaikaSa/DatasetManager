from PySide6.QtCore import QThread, Signal, Qt
import numpy as np
from pathlib import Path
from .parallel_loader import ParallelLoader
import time

class LoadingThread(QThread):
    progress = Signal(str)  # For status updates
    finished = Signal(dict)  # For final results
    
    def __init__(self, directory, use_parallel):
        super().__init__()
        self.directory = directory
        self.use_parallel = use_parallel
        self.parallel_loader = ParallelLoader()

    def run(self):
        start_time = time.time()
        self.progress.emit("Starting load process...")

        try:
            if self.use_parallel:
                self.progress.emit("Using parallel loading...")
                results = self.parallel_loader.load_images(self.directory)
            else:
                self.progress.emit("Using sequential loading...")
                results = self.load_sequential()

            end_time = time.time()
            total_time = end_time - start_time
            
            self.progress.emit(f"Load completed in {total_time:.2f} seconds")
            self.finished.emit({
                'results': results,
                'time': total_time
            })

        except Exception as e:
            self.progress.emit(f"Error during loading: {str(e)}")
            self.finished.emit(None)
        finally:
            # The Pool is created inside this QThread, so it must be stopped
            # from the same thread. Stopping it here means workers exit
            # cleanly even if a later action replaces/destroys this thread
            # (previously the pool was GC-terminated mid-map, producing
            # BrokenPipeError tracebacks in the SpawnPoolWorker processes).
            self.parallel_loader.stop_pool()

    def load_sequential(self):
        results = []
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp'}
        files = [f for f in Path(self.directory).glob("*.*")
                if f.suffix.lower() in valid_extensions]

        total_files = len(files)
        self.progress.emit(f"Found {total_files} files to process")

        from PIL import Image

        for i, file_path in enumerate(files):
            if i % 10 == 0:  # Update progress every 10 files
                self.progress.emit(f"Processing {i}/{total_files}...")

            image_path = str(file_path)
            tag_path = file_path.with_suffix('.txt')

            try:
                # Load tags (preserve order, drop duplicates)
                tags = []
                if tag_path.exists():
                    with open(tag_path, 'r', encoding='utf-8') as f:
                        seen = set()
                        for tag in f.read().split(','):
                            tag = tag.strip().lower()
                            if tag and tag not in seen:
                                tags.append(tag)

                # Decode to an RGB array (same shape as the parallel path).
                # QPixmap creation happens on the main thread afterwards -
                # GUI objects shouldn't be built on worker threads.
                with Image.open(image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    ratio = min(150 / img.width, 150 / img.height)
                    new_size = (max(1, int(img.width * ratio)), max(1, int(img.height * ratio)))
                    img_array = np.array(img.resize(new_size, Image.Resampling.LANCZOS))

                results.append({
                    'path': image_path,
                    'array': img_array,
                    'tags': tags
                })

            except Exception as e:
                self.progress.emit(f"Error processing {image_path}: {str(e)}")

        return results

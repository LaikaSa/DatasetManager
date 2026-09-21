import os
from PIL import Image
from ..logger import setup_logger
import sys
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

logger = setup_logger()

class ImageConverter:
    def __init__(self):
        self.supported_formats = {
            'png': 'PNG',
            'jpeg': 'JPEG',
            'jpg': 'JPEG',
            'bmp': 'BMP',
            'webp': 'WEBP'
        }
        self.image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

    def convert_folder(self, folder_path, target_format, recursive=False, use_parallel=False, stop_check=None):
        """Convert all images in a folder to the target format."""
        try:
            # Collect files to convert
            files_to_convert = []
            
            if recursive:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith(self.image_extensions):
                            if not file.lower().endswith(f'.{target_format}'):
                                files_to_convert.append((root, file))
            else:
                for file in os.listdir(folder_path):
                    if file.lower().endswith(self.image_extensions):
                        if not file.lower().endswith(f'.{target_format}'):
                            files_to_convert.append((folder_path, file))

            total_files = len(files_to_convert)
            if total_files == 0:
                print("No files to convert")
                return

            processed_files = 0
            
            def update_progress():
                nonlocal processed_files
                processed_files += 1
                self.print_progress_bar(processed_files, total_files)

            # Convert files
            if use_parallel:
                # Submit in chunks so a Stop is honored between chunks instead
                # of after the entire (possibly huge) backlog is queued.
                chunk = 64
                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    for start in range(0, total_files, chunk):
                        if stop_check and stop_check():
                            print("\nConversion stopped by user")
                            break
                        chunk_paths = [
                            os.path.join(root, file)
                            for root, file in files_to_convert[start:start + chunk]
                        ]
                        futures = [
                            executor.submit(self._convert_image, p, target_format)
                            for p in chunk_paths
                        ]
                        # Wait for this chunk to complete
                        for future in futures:
                            try:
                                future.result()
                            except Exception as e:
                                logger.error(f"Error in parallel conversion: {str(e)}")
                            update_progress()
            else:
                # Sequential processing
                for root, file in files_to_convert:
                    if stop_check and stop_check():
                        print("\nConversion stopped by user")
                        break
                    file_path = os.path.join(root, file)
                    try:
                        self._convert_image(file_path, target_format)
                        update_progress()
                    except Exception as e:
                        logger.error(f"Error converting {file}: {str(e)}")
                        update_progress()

            print()  # New line after progress bar completes

        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            raise

    def print_progress_bar(self, current, total, bar_length=50):
        """Print a progress bar to the terminal."""
        progress = float(current) / total
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\r[{bar}] {current}/{total}')
        sys.stdout.flush()

    def _convert_image(self, image_path, target_format):
        """Convert a single image to the target format."""
        try:
            with Image.open(image_path) as img:
                new_path = os.path.splitext(image_path)[0] + f'.{target_format}'
                
                # Handle different image modes for JPEG conversion
                if target_format.lower() == 'jpeg':
                    if img.mode in ('RGBA', 'LA'):
                        img = img.convert('RGB')
                    elif img.mode == 'P':  # Handle palette images
                        img = img.convert('RGB')
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')
                
                # quality=95 matches the resizer tab; PIL's default (75) was
                # visibly softer than the rest of the app's output
                save_kwargs = {'quality': 95} if target_format.lower() in ('jpeg', 'png') else {}
                img.save(new_path, self.supported_formats[target_format], **save_kwargs)
                os.remove(image_path)
                
        except Exception as e:
            logger.error(f"Error converting {image_path}: {str(e)}")
            raise
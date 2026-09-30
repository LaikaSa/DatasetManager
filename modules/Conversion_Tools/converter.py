import os
import shutil
import sys
from pathlib import Path
from PIL import Image
from ..logger import setup_logger
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


class IccProfileFixer:
    """Strip embedded ICC profiles from images.

    Strips *every* embedded profile (not just broken ones): libpng warns for
    any iCCP chunk that is not byte-identical to the canonical sRGB profile,
    so checking with buildTransform() would miss the ones that trigger the
    warning. Non-RGB/RGBA modes are normalized to RGB at the same time.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

    def fix_folder(self, folder_path, recursive=True, backup=True, dry_run=False, stop_check=None):
        """Strip ICC profiles from all images in a folder.

        Returns a dict with 'fixed', 'skipped' and 'errors' counts.
        """
        files = []
        if recursive:
            for root, _, filenames in os.walk(folder_path):
                for file in filenames:
                    if Path(file).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                        files.append(Path(root) / file)
        else:
            for file in os.listdir(folder_path):
                path = Path(folder_path) / file
                if path.is_file() and path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    files.append(path)

        # Never touch our own backup files
        files = [p for p in files if not p.name.endswith(".bak")]

        total_files = len(files)
        if total_files == 0:
            print("No image files found")
            return {"fixed": 0, "skipped": 0, "errors": 0}

        counts = {"fixed": 0, "skipped": 0, "errors": 0}

        for i, path in enumerate(files, 1):
            if stop_check and stop_check():
                break

            if dry_run:
                try:
                    with Image.open(path) as img:
                        if img.info.get("icc_profile") or img.mode not in ("RGB", "RGBA"):
                            counts["fixed"] += 1
                        else:
                            counts["skipped"] += 1
                except Exception as e:
                    counts["errors"] += 1
                    logger.error(f"Error checking {path}: {str(e)}")
            else:
                try:
                    if self._fix_image(path, backup) == "fixed":
                        counts["fixed"] += 1
                    else:
                        counts["skipped"] += 1
                except Exception as e:
                    counts["errors"] += 1
                    logger.error(f"Error fixing {path}: {str(e)}")

            self.print_progress_bar(i, total_files)

        print()  # New line after progress bar completes
        return counts

    def _fix_image(self, image_path, backup):
        """Strip the embedded ICC profile from a single image.

        Returns 'fixed' or 'skip'.
        """
        img = Image.open(image_path)

        icc = img.info.get("icc_profile")
        mode_needs_conversion = img.mode not in ("RGB", "RGBA")

        if not icc and not mode_needs_conversion:
            return "skip"

        # Back up original
        if backup:
            bak = image_path.with_suffix(image_path.suffix + ".bak")
            if not bak.exists():
                shutil.copy2(image_path, bak)

        # Strip profile / normalize mode
        img.load()                          # fully read data before overwriting
        img.info.pop("icc_profile", None)   # this removes the iCCP chunk
        if mode_needs_conversion:
            img = img.convert("RGB")

        # Save (no icc_profile passed -> chunk is gone)
        save_kwargs = {}
        suffix = image_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            save_kwargs["quality"] = 95
            save_kwargs["subsampling"] = 0
        elif suffix == ".png":
            save_kwargs["compress_level"] = 6

        img.save(image_path, **save_kwargs)
        return "fixed"

    def print_progress_bar(self, current, total, bar_length=50):
        """Print a progress bar to the terminal."""
        progress = float(current) / total
        filled_length = int(bar_length * progress)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        sys.stdout.write(f'\r[{bar}] {current}/{total}')
        sys.stdout.flush()
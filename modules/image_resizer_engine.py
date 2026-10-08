"""Qt-free resize engine behind modules.image_resizer (see engine contract).

ResizeWorker builds params from the widgets, passes progress/stop closures,
and keeps the GUI signal formats it always had. dry_run is engine-side only
(the GUI has no such checkbox): it scans and reports without writing.
"""

import os
import shutil

from PIL import Image

from modules.utils import IMAGE_EXTENSIONS
from modules.logger import setup_logger

logger = setup_logger()


def run(params, progress_cb=None, stop_check=None):
    """Resize every image in a folder that exceeds a maximum resolution.

    params:
      folder: path of the folder to scan
      max_resolution: int; images with either side over it are resized
          proportionally into a 'resized' subfolder of their own directory
      recursive: include images from subfolders (default False)
      dry_run: scan and report only, no writes (default False)
      in_place: overwrite resized images over their originals instead of
          writing into a 'resized' subfolder (default False)
      backup: with in_place, first copy each overwritten original to
          <name>.<ext>.bak (default True; an existing .bak is kept)
    progress_cb(current, total, message) - called per file; message carries
    the per-file status line (empty when nothing new to report).
    stop_check() -> bool - checked between files; True aborts the run and
    the summary reports partial results with 'stopped': True.
    Returns the summary dict (also the source for the GUI status lines).
    """
    folder = params['folder']
    max_resolution = int(params['max_resolution'])
    recursive = bool(params.get('recursive', False))
    dry_run = bool(params.get('dry_run', False))
    in_place = bool(params.get('in_place', False))
    backup = bool(params.get('backup', True))

    image_files = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            # Never descend into our own output folders (a re-run with a
            # lower max resolution would otherwise nest resized/resized/...)
            dirs[:] = [d for d in dirs if d != 'resized']
            for file in files:
                if file.lower().endswith(IMAGE_EXTENSIONS):
                    image_files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder):
            full = os.path.join(folder, file)
            if os.path.isfile(full) and file.lower().endswith(IMAGE_EXTENSIONS):
                image_files.append(full)

    total_files = len(image_files)
    logger.info(f"Resizing {total_files} images in {folder} "
                f"(max {max_resolution}, recursive={recursive}, dry_run={dry_run})")
    processed = 0
    resized = 0
    skipped = 0
    errors = 0
    stopped = False
    items = []

    for img_path in image_files:
        if stop_check and stop_check():
            stopped = True
            break

        try:
            with Image.open(img_path) as img:
                width, height = img.size
                needs_resize = width > max_resolution or height > max_resolution

                if needs_resize:
                    # Calculate new dimensions
                    ratio = min(max_resolution / width, max_resolution / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)

                    if dry_run:
                        action = 'dry-run'
                        reason = f'would resize {width}x{height} to {new_width}x{new_height}'
                        message = (
                            f"Would resize: {os.path.basename(img_path)}\n"
                            f"Original: {width}x{height} → New: {new_width}x{new_height}"
                        )
                    else:
                        if in_place:
                            output_path = img_path
                            if backup:
                                bak = img_path + '.bak'
                                if not os.path.exists(bak):
                                    shutil.copy2(img_path, bak)
                        else:
                            # Create resized subfolder if it doesn't exist
                            output_dir = os.path.join(os.path.dirname(img_path), 'resized')
                            os.makedirs(output_dir, exist_ok=True)
                            output_path = os.path.join(output_dir, os.path.basename(img_path))

                        # Resize and save
                        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        resized_img.save(output_path, quality=95)
                        resized += 1
                        action = 'processed'
                        reason = f'resized {width}x{height} to {new_width}x{new_height}'
                        message = (
                            f"Resized: {os.path.basename(img_path)}\n"
                            f"Original: {width}x{height} → New: {new_width}x{new_height}"
                        )

                    items.append({'path': img_path, 'action': action,
                                  'reason': reason,
                                  'original_size': [width, height],
                                  'new_size': [new_width, new_height]})
                else:
                    skipped += 1
                    items.append({'path': img_path, 'action': 'skipped',
                                  'reason': f'within {max_resolution}px limit ({width}x{height})'})
                    message = f"Skipped: {os.path.basename(img_path)} ({width}x{height})"

                processed += 1
                if progress_cb:
                    progress_cb(processed, total_files, message)

        except Exception as e:
                errors += 1
                logger.error(f"Error processing {img_path}: {str(e)}")
                items.append({'path': img_path, 'action': 'error', 'reason': str(e)})
                if progress_cb:
                    progress_cb(processed, total_files,
                                f"Error processing {img_path}: {str(e)}")

    if progress_cb:
        progress_cb(processed, total_files,
                    f"\nCompleted: {processed} images processed, {resized} images resized")

    return {'processed': processed, 'skipped': skipped, 'errors': errors,
            'items': items, 'stopped': stopped, 'total': total_files,
            'dry_run': dry_run}

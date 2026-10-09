"""Qt-free engine for the Conversion Tools feature.

Wraps the two file operations (format conversion and ICC profile
stripping) behind the standard run() API so both the GUI workers and the
CLI can drive them. Terminal presentation (progress bars) lives with the
caller: this module logs, it never prints.
"""

import os
import shutil
import multiprocessing
from pathlib import Path
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

from ..logger import setup_logger

logger = setup_logger()


def _normalize_ext(ext):
    """Normalize an extra extension (--ext) to lowercase '.ext' form."""
    ext = ext.lower()
    return ext if ext.startswith('.') else '.' + ext


def _new_summary(total):
    return {"processed": 0, "skipped": 0, "errors": 0, "stopped": False,
            "total": total, "items": []}


def _count_items(summary):
    for item in summary["items"]:
        if item["action"] in ("processed", "dry-run"):
            summary["processed"] += 1
        elif item["action"] == "skipped":
            summary["skipped"] += 1
        elif item["action"] == "error":
            summary["errors"] += 1


def run(params, progress_cb=None, stop_check=None):
    """Run one Conversion Tools operation.

    params covers every knob the GUI exposes, as plain data:
        operation: "convert" or "icc_fix" (required).
        folder_path: folder to process (required).
        recursive: scan subfolders (defaults False for "convert", True
            for "icc_fix", matching each operation's own default).
        target_format: conversion target ("convert" only), a key of
            ImageConverter.supported_formats.
        use_parallel: thread-pool the conversion ("convert" only; the
            GUI passes the app-wide cog setting here).
        backup: copy each modified file to <name>.ext.bak before
            overwriting ("icc_fix" only).
        dry_run: scan and report only, write nothing.
        extensions: optional extra extensions to scan for (the CLI's
            --ext equivalent), added to the operation's own default set.
    progress_cb(current: int, total: int, message: str = "") - called on
    progress, may be None.
    stop_check() -> bool - True means abort ASAP between files, may be
    None (never abort).
    Returns a summary dict:
        {"processed": N, "skipped": N, "errors": N, "stopped": bool,
         "total": N, "items": [{"path": ..., "action": ..., "reason": ...}]}
    Item actions are "processed|skipped|error|dry-run"; a dry-run
    "would fix" entry counts toward "processed" with action "dry-run".
    """
    operation = params.get("operation")
    if operation == "convert":
        return ImageConverter().convert_folder(
            params["folder_path"],
            params["target_format"],
            recursive=params.get("recursive", False),
            use_parallel=params.get("use_parallel", False),
            dry_run=params.get("dry_run", False),
            progress_cb=progress_cb,
            stop_check=stop_check,
            extensions=params.get("extensions"),
        )
    elif operation == "icc_fix":
        return IccProfileFixer().fix_folder(
            params["folder_path"],
            recursive=params.get("recursive", True),
            backup=params.get("backup", True),
            dry_run=params.get("dry_run", False),
            progress_cb=progress_cb,
            stop_check=stop_check,
            extensions=params.get("extensions"),
        )
    raise ValueError(f"Unknown operation: {operation!r}")


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

    def _convert_image(self, file_path, target_format):
        """Convert one image, writing the result next to the original
        (same basename, extension = target_format). The original is left
        in place; removing sources is the caller's decision (cli recycle).
        Refuses to overwrite an existing file so two sources sharing a
        basename can never clobber each other. Raises on failure.
        """
        fmt = self.supported_formats[target_format]
        out_path = os.path.splitext(file_path)[0] + '.' + target_format
        if os.path.exists(out_path):
            raise FileExistsError(f"refusing to overwrite {out_path}")
        with Image.open(file_path) as img:
            if fmt == 'JPEG' and img.mode not in ('1', 'L', 'RGB', 'CMYK'):
                # JPEG stores no alpha: flatten alpha/palette onto white.
                rgba = img.convert('RGBA')
                flat = Image.new('RGBA', rgba.size, (255, 255, 255, 255))
                flat.alpha_composite(rgba)
                img = flat.convert('RGB')
            save_kwargs = {}
            if fmt == 'JPEG':
                save_kwargs['quality'] = 95
                save_kwargs['subsampling'] = 0
            elif fmt == 'PNG':
                save_kwargs['compress_level'] = 6
            elif fmt == 'WEBP':
                save_kwargs['quality'] = 95
            img.save(out_path, **save_kwargs)
        return out_path

    def convert_folder(self, folder_path, target_format, recursive=False, use_parallel=False, dry_run=False, progress_cb=None, stop_check=None, extensions=None):
        """Convert all images in a folder to the target format.

        dry_run reports what would be written (including which outputs
        already exist) without touching anything.

        Returns a summary dict (see run()).
        """
        scan_exts = self.image_extensions + tuple(
            _normalize_ext(e) for e in extensions or ()
        )
        try:
            # Collect files to convert
            files_to_convert = []
            
            if recursive:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.lower().endswith(scan_exts):
                            if not file.lower().endswith(f'.{target_format}'):
                                files_to_convert.append((root, file))
            else:
                for file in os.listdir(folder_path):
                    if file.lower().endswith(scan_exts):
                        if not file.lower().endswith(f'.{target_format}'):
                            files_to_convert.append((folder_path, file))

            total_files = len(files_to_convert)
            summary = _new_summary(total_files)
            if total_files == 0:
                return summary

            processed_files = 0

            def update_progress():
                nonlocal processed_files
                processed_files += 1
                if progress_cb:
                    progress_cb(processed_files, total_files)

            # Convert files
            if use_parallel and not dry_run:
                # Submit in chunks so a Stop is honored between chunks instead
                # of after the entire (possibly huge) backlog is queued.
                chunk = 64
                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    for start in range(0, total_files, chunk):
                        if stop_check and stop_check():
                            summary["stopped"] = True
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
                        for path, future in zip(chunk_paths, futures):
                            try:
                                future.result()
                                summary["items"].append(
                                    {"path": path, "action": "processed", "reason": None}
                                )
                            except Exception as e:
                                logger.error(f"Error in parallel conversion: {str(e)}")
                                summary["items"].append(
                                    {"path": path, "action": "error", "reason": str(e)}
                                )
                            update_progress()
            else:
                # Sequential processing
                for root, file in files_to_convert:
                    if stop_check and stop_check():
                        summary["stopped"] = True
                        break
                    file_path = os.path.join(root, file)
                    if dry_run:
                        out_path = os.path.splitext(file_path)[0] + '.' + target_format
                        reason = (f"would skip: {out_path} exists"
                                  if os.path.exists(out_path)
                                  else f"would convert to .{target_format}")
                        summary["items"].append(
                            {"path": file_path, "action": "dry-run", "reason": reason}
                        )
                        update_progress()
                        continue
                    try:
                        self._convert_image(file_path, target_format)
                        summary["items"].append(
                            {"path": file_path, "action": "processed", "reason": None}
                        )
                    except Exception as e:
                        logger.error(f"Error converting {file}: {str(e)}")
                        summary["items"].append(
                            {"path": file_path, "action": "error", "reason": str(e)}
                        )
                    update_progress()

            _count_items(summary)
            return summary

        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            raise

class IccProfileFixer:
    """Strip embedded ICC profiles from images.

    Strips *every* embedded profile (not just broken ones): libpng warns for
    any iCCP chunk that is not byte-identical to the canonical sRGB profile,
    so checking with buildTransform() would miss the ones that trigger the
    warning. Non-RGB/RGBA modes are normalized to RGB at the same time.
    """

    SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp"}

    def fix_folder(self, folder_path, recursive=True, backup=True, dry_run=False, progress_cb=None, stop_check=None, extensions=None):
        """Strip ICC profiles from all images in a folder.

        Returns a summary dict (see run()).
        """
        scan_exts = self.SUPPORTED_EXTENSIONS | {
            _normalize_ext(e) for e in extensions or ()
        }
        files = []
        if recursive:
            for root, _, filenames in os.walk(folder_path):
                for file in filenames:
                    if Path(file).suffix.lower() in scan_exts:
                        files.append(Path(root) / file)
        else:
            for file in os.listdir(folder_path):
                path = Path(folder_path) / file
                if path.is_file() and path.suffix.lower() in scan_exts:
                    files.append(path)

        # Never touch our own backup files
        files = [p for p in files if not p.name.endswith(".bak")]

        total_files = len(files)
        summary = _new_summary(total_files)
        if total_files == 0:
            return summary

        for i, path in enumerate(files, 1):
            if stop_check and stop_check():
                summary["stopped"] = True
                break

            if dry_run:
                try:
                    with Image.open(path) as img:
                        icc = img.info.get("icc_profile")
                        mode_bad = img.mode not in ("RGB", "RGBA")
                        reason = ("embedded ICC profile" if icc
                                  else f"mode {img.mode} outside RGB/RGBA")
                        if icc or mode_bad:
                            summary["items"].append(
                                {"path": str(path), "action": "dry-run", "reason": reason}
                            )
                        else:
                            summary["items"].append(
                                {"path": str(path), "action": "skipped",
                                 "reason": "no embedded ICC profile, mode already RGB/RGBA"}
                            )
                except Exception as e:
                    logger.error(f"Error checking {path}: {str(e)}")
                    summary["items"].append(
                        {"path": str(path), "action": "error", "reason": str(e)}
                    )
            else:
                try:
                    if self._fix_image(path, backup) == "fixed":
                        summary["items"].append(
                            {"path": str(path), "action": "processed", "reason": None}
                        )
                    else:
                        summary["items"].append(
                            {"path": str(path), "action": "skipped",
                             "reason": "no embedded ICC profile, mode already RGB/RGBA"}
                        )
                except Exception as e:
                    logger.error(f"Error fixing {path}: {str(e)}")
                    summary["items"].append(
                        {"path": str(path), "action": "error", "reason": str(e)}
                    )

            if progress_cb:
                progress_cb(i, total_files)

        _count_items(summary)
        return summary

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

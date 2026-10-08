"""Qt-free entry point for the duplicate detector.

Wraps the pure detection primitives (detection.py) behind the shared
run(params, progress_cb, stop_check) engine contract so the GUI worker
and the CLI drive one implementation. Detection only reads files unless
params["delete"] asks to recycle the losers of every duplicate group
(keeper policy documented on run()); dry_run then reports without moving.
"""
import os
from modules.logger import setup_logger
from modules.utils import IMAGE_EXTENSIONS

from .detection import collect_image_files, extract_features, group_images
from PIL import Image, ImageStat

logger = setup_logger()


def _keeper_key(path, size, brightness):
    """Sort key for keeper selection: most pixels first, then higher mean
    brightness, then path (fully deterministic)."""
    pixels = int(size[0]) * int(size[1]) if size else 0
    return (-pixels, -(brightness if brightness is not None else -1.0), path)


def run(params, progress_cb=None, stop_check=None):
    """Detect duplicate image groups in a folder.

    params:
        folder: directory to scan (required).
        recursive: include subfolders (default False).
        use_hash: compute perceptual hashes (default False).
        use_hist: compute color histograms (default False).
        hash_threshold: min hash similarity 0-1 (default 0.9, the GUI
            slider default).
        hist_threshold: min histogram correlation 0-1 (default 0.95,
            the GUI slider default).
        extensions: file extensions to consider (default
            modules.utils.IMAGE_EXTENSIONS).
        delete: when True, send every group loser to the recycle bin
            (send2trash); the keeper is the group member with the most
            pixels, ties broken by higher mean brightness, then by path.
            Default False keeps the run report-only (GUI behavior).
        dry_run: with delete, report what would be recycled and move
            nothing (default False).
    progress_cb(current, total, message=""): called on per-file progress
    during the feature pass and periodically during the grouping pass
    (message is 'processing' or 'grouping'); may be None.
    stop_check() -> bool: True means abort ASAP between files; may be
    None (never abort). A stop mid-feature-pass still groups what was
    collected, matching the GUI.

    Returns the summary dict: {"processed": N, "skipped": N,
    "errors": N, "items": [per-file dicts], "groups": [...],
    "stopped": bool}. items entries are
    {"path": ..., "action": "processed|skipped|error", "reason": ...,
    "group": group index or None}; groups hold the same group dicts
    detection.group_images returns.
    """
    folder = params['folder']
    recursive = params.get('recursive', False)
    use_hash = params.get('use_hash', False)
    use_hist = params.get('use_hist', False)
    hash_threshold = params.get('hash_threshold', 0.9)
    hist_threshold = params.get('hist_threshold', 0.95)
    extensions = tuple(params.get('extensions', IMAGE_EXTENSIONS))
    delete = bool(params.get('delete', False))
    dry_run = bool(params.get('dry_run', False))

    processed = 0
    skipped = 0
    errors = 0
    items = []
    stopped = False

    image_files = collect_image_files(folder, extensions=extensions,
                                      recursive=recursive)

    total_files = len(image_files)
    logger.info(f"Found {total_files} images to process")

    # Store all images with their features
    image_features = {}
    sizes = {}  # path -> (width, height), collected here so the preview UI
    # doesn't have to re-decode every image on the main thread

    # First pass: calculate features for all images
    logger.info("First pass: Calculating features")
    for idx, image_path in enumerate(image_files):
        if stop_check is not None and stop_check():
            stopped = True
            break

        try:
            features, size = extract_features(image_path, use_hash, use_hist)
        except Exception as e:
            errors += 1
            logger.error(f"Error processing {image_path}: {str(e)}", exc_info=True)
            items.append({"path": image_path, "action": "error",
                          "reason": str(e), "group": None})
            continue

        image_features[image_path] = features
        if size is not None:
            sizes[image_path] = size

        if features:
            processed += 1
            reason = None
        else:
            # Nothing to compare with (e.g. both methods off, or every
            # requested feature failed to compute for this file).
            skipped += 1
            reason = "no features extracted"
        items.append({"path": image_path, "action": "processed" if features else "skipped",
                      "reason": reason, "group": None})

        if progress_cb is not None:
            progress_cb(idx + 1, total_files, "processing")

    # Second pass: group similar images (vectorized)
    logger.info("Second pass: Comparing images")
    groups = group_images(
        image_features, sizes,
        use_hash, use_hist,
        hash_threshold, hist_threshold,
        progress_cb=(lambda done, total: progress_cb(done, total, "grouping")
                     if progress_cb is not None else None),
        stop_check=stop_check,
    )

    if groups:
        logger.info(f"Found {len(groups)} groups of similar images")
    else:
        logger.info("No duplicate images found")

    # Point each item at its group (items for ungrouped files keep group=None).
    by_path = {item["path"]: item for item in items}
    for gidx, group in enumerate(groups):
        for path in group['images']:
            item = by_path.get(path)
            if item is not None:
                item["group"] = gidx

    deleted = 0
    would_delete = 0
    brightness = {}  # path -> mean L; only decoded for delete candidates
    if delete and groups:
        from send2trash import send2trash
        for group in groups:
            members = [p for p in group['images'] if p in by_path]
            if len(members) < 2:
                continue
            for path in members:
                try:
                    with Image.open(path) as img:
                        brightness[path] = float(
                            ImageStat.Stat(img.convert('L')).mean[0])
                except Exception:
                    brightness[path] = None
            keeper = min(members, key=lambda p: _keeper_key(
                p, sizes.get(p), brightness.get(p)))
            by_path[keeper]["keeper"] = True
            for path in members:
                if path == keeper:
                    continue
                item = by_path[path]
                item["keeper"] = False
                if dry_run:
                    would_delete += 1
                    item["action"] = "dry-run"
                    item["reason"] = f"would recycle (keeper: {keeper})"
                else:
                    try:
                        send2trash(os.path.normpath(path))
                        deleted += 1
                        item["action"] = "deleted"
                        item["reason"] = f"recycled (keeper: {keeper})"
                    except Exception as e:
                        errors += 1
                        logger.error(f"Recycle failed for {path}: {e}")
                        item["action"] = "error"
                        item["reason"] = f"recycle failed: {e}"
    if not stopped and stop_check is not None and stop_check():
        # Grouping broke out early on the flag rather than running to the end.
        stopped = True

    return {"processed": processed, "skipped": skipped, "errors": errors,
            "items": items, "groups": groups, "stopped": stopped,
            "deleted": deleted, "would_delete": would_delete}

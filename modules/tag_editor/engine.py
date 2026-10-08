"""Qt-free entry point for the tag editor.

Implements the on-disk sidecar operations (scan/dump/add/remove/replace/
clear) so the CLI and the GUI drive one implementation: data_model.py's
save path and per-image tag math delegate to the shared helpers below.
This module must stay free of PySide6 imports.
"""
from pathlib import Path

from modules.logger import setup_logger
from modules.utils import IMAGE_EXTENSIONS

logger = setup_logger()

# The operations run() accepts; 'dump' only reads, the rest write sidecars.
OPERATIONS = ('dump', 'add', 'remove', 'replace', 'clear')


def parse_tags(text, lowercase=True):
    """Parse comma-separated tag text exactly like the GUI's loaders.

    Order is preserved, whitespace stripped, duplicates dropped. The loader
    path lower-cases (lowercase=True); the caption-edit save path keeps the
    text as typed (lowercase=False).
    """
    tags = []
    seen = set()
    for tag in text.split(','):
        tag = tag.strip()
        if lowercase:
            tag = tag.lower()
        if tag and tag not in seen:
            seen.add(tag)
            tags.append(tag)
    return tags


def serialize_tags(tags):
    """Render a tag list as sidecar text (comma + space, no trailing newline)."""
    return ', '.join(tags)


def read_sidecar(txt_path):
    """Read and parse a sidecar .txt the way the GUI's loaders do."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        return parse_tags(f.read())


def backup_sidecar(txt_path):
    """Rename an existing sidecar to the next free .000/.001/... slot.

    Matches the GUI backup convention in data_model.save_changes: the old
    file is renamed (not copied) and the counter increments until the slot
    is free. Returns the backup path.
    """
    txt_path = Path(txt_path)
    backup_num = 0
    while True:
        backup_path = txt_path.with_suffix(f'.{backup_num:03d}')
        if not backup_path.exists():
            txt_path.rename(backup_path)
            return backup_path
        backup_num += 1


def write_sidecar(txt_path, tags, backup=False):
    """Write a tag list to a sidecar, optionally backing the old one up first."""
    txt_path = Path(txt_path)
    if backup and txt_path.exists():
        backup_sidecar(txt_path)
    txt_path.write_text(serialize_tags(tags), encoding='utf-8')
    return txt_path


def apply_remove(tags, tags_to_remove):
    """Remove the given tags from one tag list (data_model.remove_tags logic).

    Returns the new list, or None when nothing matched (the GUI leaves such
    files unmodified).
    """
    drop = set(tags_to_remove)
    if not set(tags) & drop:
        return None
    return [t for t in tags if t not in drop]


def apply_replace_add(tags, tags_to_replace, new_tags, positions):
    """Apply data_model.replace_or_add_tags' logic to a single tag list.

    tags_to_replace are removed first (empty = pure add); new_tags are then
    inserted at each of positions ('top', 'middle', 'bottom' and/or
    ('custom', n)); several positions clone the insert. JSON-style lists
    (['custom', n]) are accepted alongside tuples. Returns (new_list,
    changed); changed is True whenever anything happened - including a
    redundant re-insert of already-present tags - matching the GUI.
    """
    if not new_tags:
        return list(tags), False

    tags = list(tags)
    changed = False

    # Remove tags marked for replacement
    if tags_to_replace and (set(tags) & set(tags_to_replace)):
        drop = set(tags_to_replace)
        tags = [t for t in tags if t not in drop]
        changed = True

    # Also strip any new_tags that already exist in the list -
    # they will be re-inserted at the desired position (move, not duplicate)
    new_tags_set = set(new_tags)
    existing_new = new_tags_set & set(tags)
    if existing_new:
        tags = [t for t in tags if t not in existing_new]
        changed = True

    base_len = len(tags)
    indices = []
    for pos in positions:
        if pos == 'top':
            idx = 0
        elif pos == 'middle':
            idx = base_len // 2
        elif pos == 'bottom':
            idx = base_len
        elif isinstance(pos, (tuple, list)) and pos[0] == 'custom':
            idx = max(0, min(pos[1] - 1, base_len))
        else:
            continue
        indices.append(idx)

    if not indices:
        indices = [base_len]  # default to bottom if nothing selected

    # Insert highest index first so earlier indices stay valid
    for idx in sorted(indices, reverse=True):
        tags[idx:idx] = list(new_tags)
        changed = True

    return tags, changed


def scan_folder(folder, recursive=False, extensions=IMAGE_EXTENSIONS):
    """List the images the GUI would load, sorted for deterministic output.

    Same extension filter as the loaders; the GUI's Recursive checkbox maps
    to the recursive flag.
    """
    root = Path(folder)
    globber = root.rglob('*.*') if recursive else root.glob('*.*')
    return sorted(p for p in globber if p.suffix.lower() in extensions)


def run(params, progress_cb=None, stop_check=None):
    """Run one tag-editor operation over a folder of image/sidecar pairs.

    params:
      folder (str): folder to scan. Required.
      op (str): one of OPERATIONS. 'dump' only reads; 'add', 'remove',
        'replace' and 'clear' write sidecars.
      recursive (bool): scan subfolders too (the GUI's Recursive checkbox).
      extensions: image extensions to scan; defaults to IMAGE_EXTENSIONS.
      tags (list): operation selection - tags to remove for 'remove', tags
        to replace for 'replace' (the GUI's checked tags / the dialog's
        'Tags to replace' field). Lower-cased, as the GUI dialog produces.
      new_tags (list): tags to insert for 'add' and 'replace' (the dialog's
        'New tags' field, also lower-cased there).
      positions (list): insertion positions for 'add'/'replace': 'top',
        'middle', 'bottom' and/or ['custom', n] (1-based).
      captions (dict): 'clear' only - image path -> new caption text (the
        GUI's pending-caption store). Paths outside the scanned set are
        skipped; when no map is given, 'caption' applies to every scan hit.
      caption (str): caption text for 'clear' without a captions map
        (default '' - i.e. clear every scanned image).
      dry_run (bool): report the would-write items without touching disk.
      backup (bool): rename old sidecars to .000 etc. before overwriting
        (the GUI's Backup checkbox). No effect under dry_run or on 'dump'.

    progress_cb(current, total, message) and stop_check=None follow the
    shared engine contract; both may be None.

    Returns {"processed","skipped","errors","items","stopped"} with one item
    per file touched: {"path","action","reason","tags"} (reason mandatory
    for skipped/error; dry-run writes count under 'processed' with action
    'dry-run'). 'dump' also returns the full {image path: [tags]} map under
    the 'dump' key.
    """
    folder = params['folder']
    op = params.get('op', 'dump')
    if op not in OPERATIONS:
        raise ValueError(f"unknown op {op!r}; expected one of {OPERATIONS}")
    recursive = params.get('recursive', False)
    extensions = tuple(params.get('extensions', IMAGE_EXTENSIONS))
    dry_run = params.get('dry_run', False)
    backup = params.get('backup', False)
    # The GUI dialog lower-cases tag inputs; mirror that for the set ops.
    tags_sel = {t.lower() for t in params.get('tags', [])}
    new_tags = [t.lower() for t in params.get('new_tags', [])]
    positions = [tuple(p) if isinstance(p, list) else p
                 for p in params.get('positions', [])]
    captions = params.get('captions')
    default_caption = params.get('caption', '')

    files = scan_folder(folder, recursive=recursive, extensions=extensions)
    total = len(files)
    logger.info(f"Tag engine op '{op}': found {total} images to process")

    processed = skipped = errors = 0
    items = []
    stopped = False
    dump_map = {}

    for idx, path in enumerate(files):
        if stop_check is not None and stop_check():
            stopped = True
            break

        txt_path = path.with_suffix('.txt')

        try:
            # Current on-disk tags, parsed with the loaders' lower-casing.
            has_sidecar = txt_path.exists()
            current = read_sidecar(txt_path) if has_sidecar else []

            if op == 'dump':
                dump_map[str(path)] = current
                items.append({"path": str(path), "action": "processed",
                              "reason": None, "tags": current})
                processed += 1
                if progress_cb is not None:
                    progress_cb(idx + 1, total, "scanning")
                continue

            if op == 'remove':
                new = apply_remove(current, tags_sel)
                skip_reason = 'no matching tags'
            elif op in ('add', 'replace'):
                if not new_tags:
                    # GUI guard: the dialog never runs with empty new tags.
                    new, skip_reason = None, 'no new tags specified'
                else:
                    replace_set = tags_sel if op == 'replace' else set()
                    new, _ = apply_replace_add(current, replace_set,
                                              new_tags, positions)
                    skip_reason = 'no change'
            else:  # 'clear' - the edit-caption write path
                skip_reason = 'no change'
                if captions is not None:
                    caption_text = captions.get(str(path))
                    if caption_text is None and path in captions:
                        caption_text = captions[path]
                    if caption_text is None:
                        new = None
                        skip_reason = 'not in caption targets'
                    else:
                        # Parsed as typed, no lower-casing - matching the
                        # GUI's save path for user-entered captions.
                        new = parse_tags(caption_text, lowercase=False)
                else:
                    new = parse_tags(default_caption, lowercase=False)

            if new is not None and serialize_tags(new) == serialize_tags(current):
                # Contents would not change: skip the write (and any backup).
                # Without a sidecar on disk this means an empty caption:
                # leave no empty file where none existed.
                new, skip_reason = None, ('no change' if has_sidecar
                                          else 'no sidecar')

            if new is None:
                skipped += 1
                items.append({"path": str(path), "action": "skipped",
                              "reason": skip_reason, "tags": current})
            elif dry_run:
                processed += 1
                items.append({"path": str(path), "action": "dry-run",
                              "reason": f"would write: {serialize_tags(new)}",
                              "tags": new})
            else:
                write_sidecar(txt_path, new, backup=backup)
                processed += 1
                items.append({"path": str(path), "action": "processed",
                              "reason": None, "tags": new})
        except Exception as e:
            errors += 1
            logger.error(f"Error processing {path}: {str(e)}")
            items.append({"path": str(path), "action": "error",
                          "reason": str(e), "tags": None})
            continue

        if progress_cb is not None:
            progress_cb(idx + 1, total, op)

    logger.info(f"Tag engine op '{op}': processed {processed}, "
                f"skipped {skipped}, errors {errors}"
                + (", stopped early" if stopped else ""))

    summary = {"processed": processed, "skipped": skipped, "errors": errors,
               "items": items, "stopped": stopped}
    if op == 'dump':
        summary["dump"] = dump_map
    return summary

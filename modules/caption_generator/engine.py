"""
Pure-Python core of the caption generator (no Qt imports; callable from the
CLI). Both caption modes live here:

- tagger: WD-tagger ONNX models via models.ImageCaptioner (batched inference),
- natural_language: an OpenAI-compatible vision LLM endpoint via
  local_llm_captioner.LocalLLMCaptioner.

Single entry point: run(params, progress_cb, stop_check). The Qt shells in
processing.py (and the future cli.py) build a plain-data params dict and call
it; GUI-side conveniences (ProgressBar prints, signal emits) stay on the
caller side, wired through the optional params["on_caption"] /
params["on_error"] callbacks. Logging goes through modules.logger exactly
where the old workers logged.
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from modules import config as app_config
from modules.logger import setup_logger
from modules.utils import IMAGE_EXTENSIONS

from .local_llm_captioner import LocalLLMCaptioner, LocalLLMCancelled

logger = setup_logger()

# Modes accepted in params["mode"] (the GUI's model combo maps the "Natural
# Language" entry onto natural_language; anything else is a WD-tagger model).
MODE_TAGGER = "tagger"
MODE_NATURAL_LANGUAGE = "natural_language"

# Where natural-language runs park the original Danbooru-tag .txt files
# (mirroring subfolders when recursive) so the freed name gets the caption.
TAG_CAPTIONS_DIRNAME = "Tag Captions"

# Defaults mirroring the GUI widgets.
DEFAULT_THRESH = 0.35
DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.35
DEFAULT_BATCH_SIZE = 1
DEFAULT_WORKER_COUNT = 2
DEFAULT_REQUEST_TIMEOUT = 300
DEFAULT_LOCAL_LLM_URL = "http://127.0.0.1:8890"
DEFAULT_MODEL_NAME = "wd-eva02-large-tagger-v3"


def run(params: dict, progress_cb=None, stop_check=None) -> dict:
    """Generate captions for every image under params["folder"].

    params covers every knob the GUI exposes as plain data:
      common:   folder, recursive, extensions (default IMAGE_EXTENSIONS),
                debug_mode, dry_run, mode (tagger|natural_language),
                captioner (optional pre-built object, e.g. the GUI's),
                on_caption(path, caption) / on_error(message) callbacks.
      tagger:   model_name, device_id, include_rating, remove_underscore,
                append_tags, undesired_tags, prefix_tags, thresh (accepted
                for GUI parity; the pipeline uses the per-category ones, as
                before), general_threshold, character_threshold, batch_size,
                worker_count.
      natural_language: api_url, api_key (fallback: modules.config's
                load_api_key), system_prompt_key (config.yaml key; accepts
                the short forms "character"/"style"), caption_prefix,
                request_timeout.

    progress_cb(current, total, message="") â€” called on progress, may be None.
    stop_check() -> bool â€” True means abort ASAP between files (batches in
    tag mode); may be None (never abort).

    Returns the summary dict: {"processed", "skipped", "errors", "items",
    "stopped", "dry_run", "would_write", "mode"}; items entries are
    {"path", "action": "processed|skipped|error|dry-run", "reason",
    "caption"} with reason mandatory for skipped/error. dry_run performs no
    writes and moves no files (Would-write counts only).
    """
    mode = params.get("mode") or MODE_TAGGER
    if mode == MODE_NATURAL_LANGUAGE:
        return _run_natural_language(params, progress_cb, stop_check)
    if mode == MODE_TAGGER:
        return _run_tagger(params, progress_cb, stop_check)
    raise ValueError(f"Unknown mode: {mode!r} (expected {MODE_TAGGER!r} or "
                     f"{MODE_NATURAL_LANGUAGE!r})")


def _new_summary(params, mode):
    return {
        "processed": 0,
        "skipped": 0,
        "errors": 0,
        "items": [],
        "stopped": False,
        "dry_run": bool(params.get("dry_run", False)),
        "would_write": 0,
        "mode": mode,
    }


def _add_item(summary, path, action, reason=None, caption=None):
    summary["items"].append({"path": path, "action": action,
                             "reason": reason, "caption": caption})


def _progress(progress_cb, current, total):
    if progress_cb:
        progress_cb(current, total)


def _scan_images(folder_path, recursive, extensions, exclude_dirname=None):
    extensions = tuple(extensions)
    image_files = []
    if recursive:
        for root, dirs, files in os.walk(folder_path):
            if exclude_dirname:
                # Never descend into our own relocated-tags folder.
                dirs[:] = [d for d in dirs if d != exclude_dirname]
            for file in files:
                if file.lower().endswith(extensions):
                    image_files.append(os.path.join(root, file))
    else:
        image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)
                       if os.path.isfile(os.path.join(folder_path, f))
                       and f.lower().endswith(extensions)]
    return image_files


def _get_captioner(params, builder):
    """Use the caller's captioner when one was injected (the GUI owns and
    releases its own), otherwise build one from params."""
    if params.get("captioner") is not None:
        return params["captioner"]
    return builder(params)


# ---------------------------------------------------------------------------
# Tagger mode (WD-tagger ONNX models)
# ---------------------------------------------------------------------------

def _build_tag_captioner(params):
    # Deferred import: models.py drags in onnxruntime/huggingface_hub, which
    # natural-language runs never need.
    from . import models

    return models.ImageCaptioner(
        params.get("model_name") or DEFAULT_MODEL_NAME,
        debug_mode=bool(params.get("debug_mode", False)),
        device_id=params.get("device_id"),
    )


def _prepare_image(captioner, image_path):
    """Decode + preprocess one image (thread-safe, no Qt objects)."""
    try:
        return captioner.prepare_image(image_path)
    except Exception as e:
        logger.error(f"Could not prepare {image_path}: {e}")
        return None


def _apply_append(captioner, image_path, txt_path, caption, append):
    """Append mode: merge new tags into the existing caption file
    (existing tags first, de-duplicated)."""
    if not append or not os.path.exists(txt_path):
        return caption
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            existing_content = f.read().strip()

        existing_tags = [tag.strip() for tag in existing_content.split(',') if tag.strip()]
        new_tags = [tag.strip() for tag in caption.split(',') if tag.strip()]

        if captioner.debug_mode:
            logger.debug(f"\nAppending tags for {os.path.basename(image_path)}:")
            logger.debug(f"  Existing tags: {existing_tags}")
            logger.debug(f"  New tags: {new_tags}")

        combined_tags = []
        seen = set()
        for tag in existing_tags + new_tags:
            if tag not in seen:
                combined_tags.append(tag)
                seen.add(tag)

        if captioner.debug_mode:
            logger.debug(f"  Final combined tags: {combined_tags}")

        return ', '.join(combined_tags)

    except Exception as e:
        logger.error(f"Error reading existing caption for {image_path}: {e}")
        return caption


def _run_tagger(params, progress_cb, stop_check):
    folder_path = params.get("folder")
    if not folder_path:
        raise ValueError("params['folder'] is required")
    recursive = bool(params.get("recursive", False))
    extensions = tuple(params.get("extensions") or IMAGE_EXTENSIONS)
    append_tags = bool(params.get("append_tags", False))
    include_rating = bool(params.get("include_rating", False))
    remove_underscore = bool(params.get("remove_underscore", True))
    undesired_tags = set(params.get("undesired_tags") or ())
    prefix_tags = list(params.get("prefix_tags") or [])
    general_threshold = float(params.get("general_threshold", DEFAULT_GENERAL_THRESHOLD))
    character_threshold = float(params.get("character_threshold", DEFAULT_CHARACTER_THRESHOLD))
    # params["thresh"] (the GUI's "Overall threshold") is accepted for 1:1
    # parity but unused here, exactly as in the old worker: the pipeline
    # decides via the per-category thresholds.
    batch_size = max(1, int(params.get("batch_size", DEFAULT_BATCH_SIZE)))
    prep_workers = max(1, int(params.get("worker_count", DEFAULT_WORKER_COUNT)))
    dry_run = bool(params.get("dry_run", False))
    on_caption = params.get("on_caption")

    summary = _new_summary(params, MODE_TAGGER)
    captioner = _get_captioner(params, _build_tag_captioner)

    image_files = _scan_images(folder_path, recursive, extensions)
    total_files = len(image_files)

    # Basic info always shown
    logger.info(f"Starting caption generation for {total_files} images")

    completed = 0
    for batch_start in range(0, total_files, batch_size):
        if stop_check and stop_check():
            logger.info("Caption generation stopped by user")
            summary["stopped"] = True
            return summary

        batch_paths = image_files[batch_start: batch_start + batch_size]

        try:
            # Prepare images in parallel - decoding is the I/O-bound
            # part, and the data-loader worker count is finally used
            # for something.
            if len(batch_paths) == 1:
                prepared = [_prepare_image(captioner, batch_paths[0])]
            else:
                with ThreadPoolExecutor(max_workers=min(prep_workers, len(batch_paths))) as pool:
                    prepared = list(pool.map(lambda p: _prepare_image(captioner, p), batch_paths))
            ok_idx = [i for i, im in enumerate(prepared) if im is not None]
            # Failed-prep files are itemised as errors (they were silently
            # dropped before); only the prepared ones go through inference.
            failed_idx = [i for i in range(len(batch_paths)) if i not in ok_idx]

            preds = None
            if ok_idx:
                preds = captioner.predict_batch([prepared[i] for i in ok_idx])

            for j, i in enumerate(ok_idx):
                image_path = batch_paths[i]

                caption = captioner.caption_from_preds(
                    np.asarray(preds[j], dtype=float),
                    general_threshold,
                    character_threshold,
                    remove_underscore,
                    undesired_tags,
                    prefix_tags,
                    ", ",
                    include_rating,
                )

                # Log generated tags only in debug mode
                if captioner.debug_mode and caption is not None:
                    logger.debug(f"\nGenerated caption for {os.path.basename(image_path)}:")
                    logger.debug(f"  Tags: {caption.split(', ')}")

                # A None caption means inference failed for this image
                # - skip the write so we never store an error string.
                if caption is None:
                    summary["errors"] += 1
                    _add_item(summary, image_path, "error",
                              reason="caption generation failed")
                    _report_error(params, f"Caption generation failed for {image_path}")
                else:
                    txt_path = os.path.splitext(image_path)[0] + '.txt'
                    if dry_run:
                        summary["would_write"] += 1
                        _add_item(summary, image_path, "dry-run",
                                  reason=f"would write caption to {txt_path}",
                                  caption=caption)
                    else:
                        caption = _apply_append(captioner, image_path, txt_path, caption, append_tags)
                        with open(txt_path, 'w', encoding='utf-8') as f:
                            f.write(caption + '\n')
                        summary["processed"] += 1
                        _add_item(summary, image_path, "processed", caption=caption)
                    if on_caption and caption is not None:
                        on_caption(image_path, caption)

                completed += 1
                _progress(progress_cb, completed, total_files)

            for i in failed_idx:
                summary["errors"] += 1
                _add_item(summary, batch_paths[i], "error",
                          reason="could not prepare image (see log)")
                completed += 1
                _progress(progress_cb, completed, total_files)

        except Exception as e:
            logger.error(f"Error processing batch starting at {batch_paths[0]}: {str(e)}")
            for image_path in batch_paths:
                summary["errors"] += 1
                _add_item(summary, image_path, "error", reason=str(e))
            on_error = params.get("on_error")
            if on_error:
                on_error(f"Error processing batch: {str(e)}")
            completed += len(batch_paths)

    # Basic completion info always shown
    logger.info("Caption generation completed")
    return summary


def _report_error(params, message):
    """Log at the engine's existing error call sites and hand the message to
    the caller's on_error callback (the GUI re-logs/emits through its own
    handler), mirroring the old worker behaviour."""
    logger.error(message)
    on_error = params.get("on_error")
    if on_error:
        on_error(message)


# ---------------------------------------------------------------------------
# Natural language mode (OpenAI-compatible vision LLM endpoint)
# ---------------------------------------------------------------------------

def _build_llm_captioner(params):
    key = params.get("system_prompt_key") or app_config.SYSTEM_PROMPT_CONFIG_KEY_CHARACTER
    # Accept the short forms so CLI callers do not need to know the config
    # key spelling; full config keys pass through untouched.
    key = {
        "character": app_config.SYSTEM_PROMPT_CONFIG_KEY_CHARACTER,
        "style": app_config.SYSTEM_PROMPT_CONFIG_KEY_STYLE,
    }.get(key, key)
    api_key = params.get("api_key")
    if api_key is None:
        api_key = app_config.load_api_key()
    return LocalLLMCaptioner(
        params.get("api_url") or DEFAULT_LOCAL_LLM_URL,
        api_key=api_key,
        debug_mode=bool(params.get("debug_mode", False)),
        request_timeout=int(params.get("request_timeout", DEFAULT_REQUEST_TIMEOUT)),
        system_prompt_key=key,
    )


def _apply_prefix(caption_prefix, caption):
    """Prepend the user's prefix text to the generated caption.

    Mirrors the tag-mode "Prefix tags" behavior: the prefix is put at
    the very beginning, and ", " is inserted between it and the caption
    unless the prefix already ends with a comma (so users can control
    the exact separator themselves)."""
    prefix = caption_prefix
    if not prefix:
        return caption
    if not caption:
        return prefix
    if prefix.endswith(","):
        return prefix + " " + caption
    return prefix + ", " + caption


def _read_tags(folder_path, image_path, tag_captions_dir):
    """Read the Danbooru-tag .txt file for this image without moving
    anything. Returns None if there are no existing tags.

    The 'Tag Captions' copy takes priority: it only ever exists because
    an earlier run relocated the real tags there, while a file beside
    the image with the same name is then a generated caption, not tags."""
    image_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    original_txt = os.path.join(image_dir, base_name + '.txt')

    relative_dir = os.path.relpath(image_dir, folder_path)
    mirrored_dir = os.path.normpath(os.path.join(tag_captions_dir, relative_dir))
    mirrored_txt = os.path.join(mirrored_dir, base_name + '.txt')

    if os.path.exists(mirrored_txt):
        # Already relocated by an earlier run.
        with open(mirrored_txt, 'r', encoding='utf-8') as f:
            tags_text = f.read().strip()
        return tags_text if tags_text else None

    if os.path.exists(original_txt):
        with open(original_txt, 'r', encoding='utf-8') as f:
            tags_text = f.read().strip()
        return tags_text if tags_text else None

    return None


def _relocate_tags(folder_path, image_path, tag_captions_dir):
    """Move the Danbooru-tag .txt file beside this image into the 'Tag
    Captions' folder (mirroring subfolder structure when recursive),
    freeing <base>.txt for the natural-language caption. Called only
    after a caption has been received, so a failed request leaves the
    tag file in place. No-op when there is no tag file to move, or when
    an earlier run already relocated it (the file beside the image is
    then a generated caption, not tags)."""
    image_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    original_txt = os.path.join(image_dir, base_name + '.txt')
    if not os.path.exists(original_txt):
        return

    relative_dir = os.path.relpath(image_dir, folder_path)
    mirrored_dir = os.path.normpath(os.path.join(tag_captions_dir, relative_dir))
    mirrored_txt = os.path.join(mirrored_dir, base_name + '.txt')
    if os.path.exists(mirrored_txt):
        return  # tags already relocated; original is a generated caption

    os.makedirs(mirrored_dir, exist_ok=True)
    os.replace(original_txt, mirrored_txt)


def _run_natural_language(params, progress_cb, stop_check):
    folder_path = params.get("folder")
    if not folder_path:
        raise ValueError("params['folder'] is required")
    recursive = bool(params.get("recursive", False))
    extensions = tuple(params.get("extensions") or IMAGE_EXTENSIONS)
    caption_prefix = (params.get("caption_prefix") or "").strip()
    dry_run = bool(params.get("dry_run", False))
    skip_existing = bool(params.get("skip_existing", False))
    retry = bool(params.get("retry", False))
    retry_interval = max(1, int(params.get("retry_interval", 30)))
    max_passes = max(0, int(params.get("max_passes", 0)))  # 0 = unlimited
    on_caption = params.get("on_caption")

    summary = _new_summary(params, MODE_NATURAL_LANGUAGE)
    captioner = _get_captioner(params, _build_llm_captioner)

    image_files = _scan_images(folder_path, recursive, extensions,
                               exclude_dirname=TAG_CAPTIONS_DIRNAME)
    total_files = len(image_files)
    logger.info(f"Starting natural language caption generation for {total_files} images")

    tag_captions_dir = os.path.join(folder_path, TAG_CAPTIONS_DIRNAME)

    completed = 0
    pending = list(image_files)
    last_errors = {}
    pass_num = 0
    while pending:
        pass_num += 1
        if max_passes and pass_num > max_passes:
            break
        if pass_num > 1:
            logger.info(
                f"{len(pending)} image(s) failed (LLM server down?); "
                f"pass {pass_num}: retrying in {retry_interval}s")
            # Sleep in 1s increments so a Stop click interrupts the wait.
            for _ in range(retry_interval):
                if stop_check and stop_check():
                    logger.info("Natural language captioning stopped by user")
                    summary["stopped"] = True
                    return summary
                time.sleep(1)

        still_pending = []
        for image_path in pending:
            if stop_check and stop_check():
                logger.info("Natural language captioning stopped by user")
                summary["stopped"] = True
                return summary

            if skip_existing:
                # Resume support: an image is "already captioned" when its
                # tag file was relocated to 'Tag Captions' AND a caption .txt
                # now sits beside the image. Skip it without calling the LLM.
                # (Images that never had a tag file can't be detected this
                # way and are re-captioned on a re-run.)
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                mirrored_txt = os.path.normpath(os.path.join(
                    tag_captions_dir,
                    os.path.relpath(os.path.dirname(image_path), folder_path),
                    base_name + '.txt'))
                if (os.path.exists(mirrored_txt)
                        and os.path.exists(os.path.splitext(image_path)[0]
                                           + '.txt')):
                    summary["skipped"] += 1
                    _add_item(summary, image_path, "skipped",
                              reason="already captioned")
                    completed += 1
                    _progress(progress_cb, completed, total_files)
                    continue

            try:
                tags_text = _read_tags(folder_path, image_path, tag_captions_dir)

                caption = captioner.caption_image(
                    image_path,
                    tags_text=tags_text,
                    should_stop=stop_check,
                )
                caption = _apply_prefix(caption_prefix, caption)

                # Only after a caption has been received: move the tag file
                # into 'Tag Captions' (which frees <base>.txt), then write the
                # natural-language caption into the freed name. If the request
                # fails, the tag file stays in place so a retry needs no manual
                # cleanup.
                if dry_run:
                    summary["would_write"] += 1
                    _add_item(summary, image_path, "dry-run",
                              reason="would move tag file and write caption",
                              caption=caption)
                else:
                    _relocate_tags(folder_path, image_path, tag_captions_dir)

                    txt_path = os.path.splitext(image_path)[0] + '.txt'
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(caption + '\n')
                    summary["processed"] += 1
                    _add_item(summary, image_path, "processed", caption=caption)
                if on_caption:
                    on_caption(image_path, caption)

            except LocalLLMCancelled:
                logger.info("Natural language captioning stopped by user")
                summary["stopped"] = True
                return summary
            except Exception as e:
                # Failed (usually the LLM server is unreachable). Keep the
                # image pending so a later pass can retry it; the error is
                # recorded once, at the end, so an image that recovers on a
                # later pass counts as processed, not as an error.
                last_errors[image_path] = str(e)
                _report_error(params, f"Error processing {image_path}: {str(e)}")
                still_pending.append(image_path)
                continue

            completed += 1
            _progress(progress_cb, completed, total_files)

        pending = still_pending
        if pending and not retry:
            break

    # Record the images that were never captioned. Reached only when the
    # loop finished without a user stop (a stop returns early above).
    for image_path in pending:
        summary["errors"] += 1
        _add_item(summary, image_path, "error",
                  reason=last_errors.get(image_path, "not processed"))
        completed += 1
        _progress(progress_cb, completed, total_files)

    logger.info("Natural language caption generation completed")
    return summary

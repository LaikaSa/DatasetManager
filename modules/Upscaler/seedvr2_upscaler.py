"""Self-contained SeedVR2 image upscaler.

Wraps the vendored ``modules.Upscaler.seedvr2`` package (numz/ComfyUI-SeedVR2_VideoUpscaler,
ByteDance SeedVR2) so the app can upscale a set of *images* with the
``seedvr2_ema_3b_fp16`` DiT + ``ema_vae_fp16`` VAE. Video-only concerns from the
upstream repo are ignored: each image is treated as a single-frame clip.

The Qt-free pipeline (SeedVR2Upscaler) lives in engine.py; this module keeps
the constants, the HF-cache helpers and the thin QThread shells the tab
drives.

Public surface used by ``modules.Upscaler.upscaler``:
    SEEDVR2_DIT_MODEL / SEEDVR2_VAE_MODEL  - weight file names
    get_seedvr2_model_dir()                - where weights live on disk
    are_seedvr2_models_downloaded()        - quick existence check
    download_seedvr2_models(progress_cb)   - download both weights (thread-safe)
    SeedVR2UpscaleWorker                   - QThread running the 4-phase pipeline

Reference implementation:
    https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler
    https://huggingface.co/numz/SeedVR2_comfyUI
"""

import os
import datetime

# Reduce CUDA memory fragmentation ("reserved but unallocated") - must be set
# before the first CUDA allocation, hence at import time of this module.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PySide6.QtCore import QThread, Signal

from modules.logger import setup_logger

logger = setup_logger()

# The heavy weights are pulled from this HuggingFace repo and stored in the
# shared HuggingFace cache (~/.cache/huggingface/hub) - the same mechanism the
# caption generator uses, so nothing is copied into the project folder.
SEEDVR2_HF_REPO = "numz/SeedVR2_comfyUI"
SEEDVR2_DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"
SEEDVR2_VAE_MODEL = "ema_vae_fp16.safetensors"
# Cache keys for the vendored code's process-wide GlobalModelCache - the
# DiT + VAE are stored there for the life of the process unless removed
# explicitly (see clear_gpu_memory).
SEEDVR2_DIT_CACHE_ID = "seedvr2_dit"
SEEDVR2_VAE_CACHE_ID = "seedvr2_vae"


def _hf_try_cache(repo_id, filename):
    """Locate a file in the HF cache without any network call (None if absent)."""
    try:
        from huggingface_hub import hf_hub_try_to_load_from_cache as _try
    except ImportError:  # huggingface_hub 1.x renamed it (drops the hf_ prefix)
        from huggingface_hub import try_to_load_from_cache as _try
    return _try(repo_id, filename)


def are_seedvr2_models_downloaded():
    """True when both weights are already in the HF cache (no network)."""
    return (
        _hf_try_cache(SEEDVR2_HF_REPO, SEEDVR2_DIT_MODEL) is not None
        and _hf_try_cache(SEEDVR2_HF_REPO, SEEDVR2_VAE_MODEL) is not None
    )


def get_seedvr2_model_dir():
    """Directory holding the cached SeedVR2 weights (the repo's snapshot dir).

    Falls back to the HF cache root if the files aren't present yet (the caller
    should download first).
    """
    cached = _hf_try_cache(SEEDVR2_HF_REPO, SEEDVR2_DIT_MODEL) or \
             _hf_try_cache(SEEDVR2_HF_REPO, SEEDVR2_VAE_MODEL)
    if cached:
        return os.path.dirname(cached)
    from huggingface_hub import constants as hf_constants
    return hf_constants.HF_HUB_CACHE


def download_seedvr2_models(progress_cb=None):
    """Download the DiT + VAE weights into the shared HuggingFace cache.

    ``progress_cb(text)`` is called with human-readable status lines so the UI can
    mirror progress. Returns True on success.
    """
    from huggingface_hub import hf_hub_download

    for filename in (SEEDVR2_DIT_MODEL, SEEDVR2_VAE_MODEL):
        if progress_cb:
            progress_cb(f"Downloading {filename} to HuggingFace cache...")
        hf_hub_download(repo_id=SEEDVR2_HF_REPO, filename=filename)
        if progress_cb:
            progress_cb(f"Downloaded {filename}")
    return True


class SeedVR2DownloadWorker(QThread):
    """Downloads the SeedVR2 weights off the GUI thread (they total ~7GB)."""

    status = Signal(str)
    finished_ok = Signal(bool)

    def __init__(self):
        super().__init__()

    def run(self):
        try:
            ok = download_seedvr2_models(progress_cb=self.status.emit)
        except Exception as e:
            self.status.emit(f"Download error: {e}")
            logger.exception("SeedVR2 download failed")
            ok = False
        self.finished_ok.emit(bool(ok))


class _SilentDebug:
    """Minimal stand-in for the upstream Debug object (logging disabled).

    The SeedVR2 pipeline expects a ``Debug`` instance; we pass a lightweight
    one that swallows log calls so it doesn't spam the console. Method calls
    are no-ops, and the handful of data attributes the pipeline reads/writes
    (tile-boundary lists, timer dict) are real containers so ``.append`` /
    ``.get`` work.
    """

    enabled = False

    # If these exist, the VAE records tile boundaries and the post-process step
    # draws them as a coloured grid + numbers on the output. We don't want the
    # debug overlay, so report them as absent (hasattr -> False, getattr -> None).
    _ABSENT = ("encode_tile_boundaries", "decode_tile_boundaries")

    def __init__(self):
        # Read via .get() elsewhere; keep it a real dict so that works.
        self.timer_durations = {}

    def __getattr__(self, name):
        if name in self._ABSENT:
            raise AttributeError(name)
        # Any other attribute is treated as a callable no-op.
        def _noop(*args, **kwargs):
            return None
        return _noop

    def log(self, message, *args, **kwargs):
        # Keep a quiet channel for warnings/errors only.
        level = kwargs.get("level", "INFO")
        if level in ("WARNING", "ERROR"):
            logger.warning("SeedVR2: %s", message)


class SeedVR2UpscaleWorker(QThread):
    """QThread that upscales a list of images with SeedVR2 (single-frame mode).

    Thin shell around engine.SeedVR2Upscaler; all pipeline logic lives there
    so the CLI can drive the same code.

    seed: a fixed value (>=0) applies to every image; -1 draws a fresh random
    seed per image (ComfyUI-style randomize).
    """

    progress = Signal(int)
    status = Signal(str)
    finished = Signal()

    def __init__(self, input_paths, scale_factor, device="cpu", seed=42,
                 color_correction="wavelet", tile_vae=True, min_size=0):
        super().__init__()
        self.input_paths = input_paths if isinstance(input_paths, list) else [input_paths]
        self.is_running = True
        from .engine import SeedVR2Upscaler
        self.engine = SeedVR2Upscaler(
            scale_factor, device, seed, color_correction, tile_vae, min_size,
        )

    # ── thread entry ──────────────────────────────────────────────────────
    def run(self):
        try:
            self.engine._setup_runner(status_cb=self.status.emit)

            total = len(self.input_paths)
            processed = 0
            start = datetime.datetime.now()

            for idx, img_path in enumerate(self.input_paths, start=1):
                if not self.is_running:
                    self.status.emit("Process stopped by user")
                    break

                self.status.emit(f"Upscaling [{idx}/{total}]: {os.path.basename(img_path)}")
                logger.info("SeedVR2 processing: %s", os.path.basename(img_path))
                try:
                    seed = self.engine._resolve_seed()
                    if self.engine.seed == -1:
                        self.status.emit(f"Seed: {seed}")
                    out_img = self.engine._process_one(img_path, seed)
                    self.engine._save(img_path, out_img)
                except Exception as e:
                    self.status.emit(f"Error processing {img_path}: {e}")
                    logger.exception("SeedVR2 image failed: %s", img_path)
                    continue

                processed += 1
                self.progress.emit(processed)

            duration = (datetime.datetime.now() - start).total_seconds()
            self.status.emit(
                f"Finished processing {processed} images in {duration:.1f} seconds"
            )
            logger.info("SeedVR2 done: %d images in %.1fs", processed, duration)

        except Exception as e:
            self.status.emit(f"Error: {e}")
            logger.exception("SeedVR2 run failed")
        finally:
            # Free DiT + VAE + CUDA cache once the WHOLE batch is done
            # (or stopped) - never per image.
            self.engine.clear_gpu_memory()
            self.status.emit("GPU memory freed")
            self.finished.emit()

    def stop(self):
        self.is_running = False

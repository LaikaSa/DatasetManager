"""Self-contained SeedVR2 image upscaler.

Wraps the vendored ``modules.Upscaler.seedvr2`` package (numz/ComfyUI-SeedVR2_VideoUpscaler,
ByteDance SeedVR2) so the app can upscale a set of *images* with the
``seedvr2_ema_3b_fp16`` DiT + ``ema_vae_fp16`` VAE. Video-only concerns from the
upstream repo are ignored: each image is treated as a single-frame clip.

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
import math
import datetime

# Reduce CUDA memory fragmentation ("reserved but unallocated") - must be set
# before the first CUDA allocation, hence at import time of this module.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from PIL import Image
from PySide6.QtCore import QThread, Signal

from modules.logger import setup_logger

logger = setup_logger()

# The heavy weights are pulled from this HuggingFace repo and stored in the
# shared HuggingFace cache (~/.cache/huggingface/hub) - the same mechanism the
# caption generator uses, so nothing is copied into the project folder.
SEEDVR2_HF_REPO = "numz/SeedVR2_comfyUI"
SEEDVR2_DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"
SEEDVR2_VAE_MODEL = "ema_vae_fp16.safetensors"


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
    """QThread that upscales a list of images with SeedVR2 (single-frame mode)."""

    progress = Signal(int)
    status = Signal(str)
    finished = Signal()

    def __init__(self, input_paths, scale_factor, device="cpu", seed=42,
                 color_correction="wavelet", tile_vae=True, min_size=0):
        super().__init__()
        self.input_paths = input_paths if isinstance(input_paths, list) else [input_paths]
        self.scale_factor = float(scale_factor)
        self.min_size = int(min_size)  # >0 -> auto per-image scale factor
        self.device = device
        self.seed = int(seed)
        self.color_correction = color_correction
        # Tile BOTH VAE encode and decode to keep peak VRAM low on large images.
        self.tile_vae = tile_vae
        self.is_running = True
        self.runner = None
        self.ctx = None
        self.debug = _SilentDebug()

    # ── model / runner lifecycle ───────────────────────────────────────────
    def _setup_runner(self):
        from modules.Upscaler.seedvr2.src.utils.constants import get_script_directory
        from modules.Upscaler.seedvr2.src.core.generation_utils import (
            setup_generation_context,
            prepare_runner,
            load_text_embeddings,
        )

        if not are_seedvr2_models_downloaded():
            raise FileNotFoundError(
                f"SeedVR2 weights not found in {get_seedvr2_model_dir()}. "
                "Download the model first."
            )

        self.status.emit(f"Loading SeedVR2 (device: {self.device})...")
        logger.info("SeedVR2: loading models on %s", self.device)

        # Offload target = the compute device, so "cache the model between
        # images" keeps DiT + VAE resident on the GPU (not offloaded to CPU)
        # and a batch of images reuses the loaded weights instead of the VAE
        # being torn down (runner.vae -> None) after each image.
        self.ctx = setup_generation_context(
            dit_device=self.device,
            vae_device=self.device,
            dit_offload_device=self.device,
            vae_offload_device=self.device,
            debug=self.debug,
        )

        self.runner, cache_context = prepare_runner(
            dit_model=SEEDVR2_DIT_MODEL,
            vae_model=SEEDVR2_VAE_MODEL,
            model_dir=get_seedvr2_model_dir(),
            debug=self.debug,
            ctx=self.ctx,
            dit_cache=True,
            vae_cache=True,
            dit_id="seedvr2_dit",
            vae_id="seedvr2_vae",
            block_swap_config=None,
            encode_tiled=self.tile_vae,
            encode_tile_size=(512, 512),
            encode_tile_overlap=(64, 64),
            decode_tiled=self.tile_vae,
            decode_tile_size=(512, 512),
            decode_tile_overlap=(64, 64),
            attention_mode="sdpa",
        )
        self.ctx["cache_context"] = cache_context

        self.ctx["text_embeds"] = load_text_embeddings(
            get_script_directory(), self.ctx["dit_device"],
            self.ctx["compute_dtype"], self.debug,
        )
        self.status.emit("SeedVR2 models loaded")
        logger.info("SeedVR2: models loaded")

    def clear_gpu_memory(self):
        from modules.Upscaler.seedvr2.src.optimization.memory_manager import complete_cleanup

        if self.runner is not None:
            try:
                complete_cleanup(self.runner, debug=self.debug,
                                 dit_cache=False, vae_cache=False)
            except Exception as e:
                logger.warning("SeedVR2 cleanup: %s", e)
            self.runner = None
        self.ctx = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── per-image processing ───────────────────────────────────────────────
    def _load_frame(self, img_path):
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        frames = torch.from_numpy(arr[None, ...])  # [1, H, W, C] in [0, 1]
        return frames, (img.height, img.width)

    def _process_one(self, img_path):
        from modules.Upscaler.seedvr2.src.core.generation_phases import (
            encode_all_batches,
            upscale_all_batches,
            decode_all_batches,
            postprocess_all_batches,
        )

        frames, (h, w) = self._load_frame(img_path)
        # SeedVR2 resizes the *shortest edge* to `resolution` (upscale only),
        # padding to a multiple of 16 internally.
        if self.min_size > 0:
            # Auto mode: smallest 0.1 step (1.0, 1.1, ...) that brings the
            # longest side to at least min_size, applied strictly to both
            # sides so the aspect ratio is kept exactly.
            steps = math.ceil(self.min_size * 10 / max(w, h) - 1e-9)
            scale = max(1.0, steps / 10.0)
            resolution = max(16, int(round(min(h, w) * scale)))
        else:
            resolution = int(round(min(h, w) * self.scale_factor))
            resolution = max(resolution, 16)

        # Reset per-run ctx state so batches don't leak across images.
        self.ctx["all_latents"] = []
        self.ctx["all_upscaled_latents"] = []
        self.ctx["batch_samples"] = []
        self.ctx["final_video"] = None
        self.ctx["video_transform"] = None
        self.ctx.pop("true_target_dims", None)

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.ctx = encode_all_batches(
            self.runner, ctx=self.ctx, images=frames, debug=self.debug,
            batch_size=1, uniform_batch_size=False, seed=self.seed,
            progress_callback=None, temporal_overlap=0,
            resolution=resolution, max_resolution=0,
            input_noise_scale=0.0, color_correction=self.color_correction,
        )
        # cache_model=True: keep DiT/VAE on the GPU so the next image in a
        # batch reuses them (otherwise the VAE is freed after each image).
        self.ctx = upscale_all_batches(
            self.runner, ctx=self.ctx, debug=self.debug, progress_callback=None,
            seed=self.seed, latent_noise_scale=0.0, cache_model=True,
        )
        self.ctx = decode_all_batches(
            self.runner, ctx=self.ctx, debug=self.debug,
            progress_callback=None, cache_model=True,
        )
        self.ctx = postprocess_all_batches(
            ctx=self.ctx, debug=self.debug, progress_callback=None,
            color_correction=self.color_correction, prepend_frames=0,
            temporal_overlap=0, batch_size=1,
        )

        sample = self.ctx["final_video"]
        if torch.is_tensor(sample):
            if sample.is_cuda or sample.is_mps:
                sample = sample.cpu()
            sample = sample.to(torch.float32)

        out = sample[0]  # [H', W', C]
        out = (out.numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def _save(self, img_path, out_img):
        output_path = os.path.join(
            os.path.dirname(img_path), "upscaled", os.path.basename(img_path)
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_img.save(output_path)
        return output_path

    # ── thread entry ───────────────────────────────────────────────────────
    def run(self):
        try:
            self._setup_runner()

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
                    out_img = self._process_one(img_path)
                    self._save(img_path, out_img)
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
            self.clear_gpu_memory()
            self.finished.emit()

    def stop(self):
        self.is_running = False

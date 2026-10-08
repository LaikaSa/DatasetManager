"""Qt-free engine for the Upscaler feature (RealESRGAN + SeedVR2).

Both pipelines are lifted from the QThread workers (realesrgan.py /
seedvr2_upscaler.py), which are now thin Qt shells around these classes.
Terminal presentation stays with the caller: this module logs, it never
prints. Progress/stop follow the shared engine contract:
    progress_cb(current, total, message="")
    stop_check() -> bool
"""
import os
import math
import random
import datetime
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from modules.logger import setup_logger
from modules.utils import IMAGE_EXTENSIONS

logger = setup_logger()

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
REAL_ESRGAN_MODEL_NAME = "RealESRGAN_x4plus_anime_6B.pth"
REAL_ESRGAN_MODEL_PATH = ROOT_DIR / "models" / REAL_ESRGAN_MODEL_NAME
REAL_ESRGAN_MODEL_URL = ("https://github.com/xinntao/Real-ESRGAN/releases/"
                         "download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth")


# ── RealESRGAN network (moved from realesrgan.py; torch-only) ─────────────
class RRDBNet(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.conv_head = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_tail = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_head(x)
        feat = self.body(feat)
        feat = self.conv_body(feat)
        out = self.conv_tail(self.lrelu(feat))
        return out


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        out = self.conv5(self.lrelu(self.conv4(self.lrelu(self.conv3(self.lrelu(self.conv2(self.lrelu(self.conv1(x)))))))))
        return out * 0.2 + x


class RDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RDB, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.lrelu(x1)
        x1 = self.conv2(x1)
        x1 = self.lrelu(x1)
        x1 = self.conv3(x1)
        x1 = self.lrelu(x1)
        x1 = self.conv4(x1)
        x1 = self.lrelu(x1)
        x1 = self.conv5(x1)
        return x1 * 0.2 + x


class RealESRGANUpscaler:
    """One loaded RealESRGAN model + per-image tiled inference.

    Output convention (unchanged from the old worker): results are written
    to <image dir>/upscaled/<basename>. The model is freed once per run,
    after the whole batch (or stop) - never per image.
    """

    def __init__(self, model_path, device=None, scale_factor=4.0, min_size=0,
                 tile_size=512, tile_pad=32):
        self.model_path = model_path
        self.scale_factor = float(scale_factor)
        self.min_size = int(min_size)  # >0 -> auto per-image scale factor
        if device:
            self.device = device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.tile_size = tile_size
        self.tile_pad = tile_pad

    def _effective_scale(self, w, h):
        """Scale factor for one image.

        In auto mode (min_size > 0) this is the *smallest 0.1 step*
        (1.0, 1.1, 1.2, ...) that brings the longest side to at least
        min_size. The factor is applied strictly to both sides (no pixel
        alignment rounding), so the aspect ratio is kept exactly. Otherwise
        the user's fixed factor is used.
        """
        if self.min_size > 0:
            steps = math.ceil(self.min_size * 10 / max(w, h) - 1e-9)
            return max(1.0, steps / 10.0)
        return self.scale_factor

    def load_model(self, status_cb=None):
        def _status(msg):
            if status_cb:
                status_cb(msg)

        state_dict = torch.load(self.model_path, map_location=self.device)
        if 'params_ema' in state_dict:
            state_dict = state_dict['params_ema']

        # Count the number of RRDB blocks
        block_count = 0
        for key in state_dict.keys():
            if key.startswith('body.'):
                parts = key.split('.')
                if len(parts) > 2 and parts[1].isdigit():
                    block_num = int(parts[1])
                    block_count = max(block_count, block_num + 1)

        _status(f"Detected {block_count} blocks in model")

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=block_count,
            num_grow_ch=32
        )

        model.load_state_dict(state_dict)
        model.eval()
        self.model = model.to(self.device)
        return self.model

    def process_tile(self, tile, scale):
        # Convert tile to tensor
        tile_np = np.array(tile)
        tile_tensor = torch.from_numpy(tile_np).float() / 255.0
        tile_tensor = tile_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(tile_tensor)
            if scale != 4:
                output = torch.nn.functional.interpolate(
                    output,
                    scale_factor=scale/4,
                    mode='bicubic',
                    align_corners=False
                )

        # Convert back to PIL Image
        output = output.squeeze().permute(1, 2, 0).cpu().numpy()
        output = (output * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(output)

    def process_image(self, img_path, stop_check=None, tile_progress_cb=None,
                      error_cb=None, in_place=False, backup=True):
        """Upscale one image into <dir>/upscaled/<name> (or in place).

        Returns True on success, False on stop/error.
        """
        try:
            img = Image.open(img_path).convert('RGB')

            # Per-image scale factor (auto mode targets the minimum size)
            scale = self._effective_scale(img.width, img.height)

            # Apply the factor strictly to both sides (nearest pixel) so the
            # aspect ratio is preserved with no alignment rounding.
            dest_w = max(8, int(round(img.width * scale)))
            dest_h = max(8, int(round(img.height * scale)))

            # Calculate tile dimensions
            tile_w = min(self.tile_size, img.width)
            tile_h = min(self.tile_size, img.height)

            # If image is small enough, process it directly
            if img.width <= self.tile_size and img.height <= self.tile_size:
                output_img = self.process_tile(img, scale)
                if output_img.size != (dest_w, dest_h):
                    output_img = output_img.resize((dest_w, dest_h), LANCZOS)
            else:
                # Tiled path: accumulate into a float buffer and blend the
                # overlap region with linear ramps (plain paste() left visible
                # seams at tile borders).
                out = np.zeros((dest_h, dest_w, 3), dtype=np.float32)
                weight = np.zeros((dest_h, dest_w), dtype=np.float32)

                step_w = max(1, tile_w - self.tile_pad)
                step_h = max(1, tile_h - self.tile_pad)
                total_tiles = ((img.height - 1) // step_h + 1) * ((img.width - 1) // step_w + 1)
                current_tile = 0

                pad_w_out = int(self.tile_pad * scale)
                pad_h_out = int(self.tile_pad * scale)

                for y in range(0, img.height, step_h):
                    for x in range(0, img.width, step_w):
                        if stop_check is not None and stop_check():
                            return False

                        # Extract and process tile
                        right = min(x + tile_w, img.width)
                        bottom = min(y + tile_h, img.height)
                        tile = img.crop((x, y, right, bottom))
                        processed_tile = self.process_tile(tile, scale)

                        # Target region in output space, at its exact size so
                        # non-integer scale factors can't leave 1px gaps.
                        tx = int(round(x * scale))
                        ty = int(round(y * scale))
                        tw = min(int(round((right - x) * scale)), dest_w - tx)
                        th = min(int(round((bottom - y) * scale)), dest_h - ty)
                        if processed_tile.size != (tw, th):
                            processed_tile = processed_tile.resize((tw, th), LANCZOS)

                        # Linear ramps over the overlap, disabled at the
                        # image borders where there is no neighbour tile.
                        wx = np.ones(tw, dtype=np.float32)
                        if x > 0 and pad_w_out > 0:
                            r = min(pad_w_out, tw)
                            wx[:r] = np.linspace(0.0, 1.0, r, dtype=np.float32)
                        if right < img.width and pad_w_out > 0:
                            r = min(pad_w_out, tw)
                            wx[tw - r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
                        wy = np.ones(th, dtype=np.float32)
                        if y > 0 and pad_h_out > 0:
                            r = min(pad_h_out, th)
                            wy[:r] = np.linspace(0.0, 1.0, r, dtype=np.float32)
                        if bottom < img.height and pad_h_out > 0:
                            r = min(pad_h_out, th)
                            wy[th - r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)

                        w2 = wy[:, None] * wx[None, :]
                        out[ty:ty + th, tx:tx + tw] += np.asarray(
                            processed_tile, dtype=np.float32) * w2[:, :, None]
                        weight[ty:ty + th, tx:tx + tw] += w2

                        # Update tile progress
                        current_tile += 1
                        if tile_progress_cb is not None:
                            tile_progress_cb(current_tile, total_tiles)

                np.maximum(weight, 1e-6, out=weight)  # guard any uncovered pixel
                out /= weight[:, :, None]
                output_img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

            # Save the result (subfolder by default; in_place overwrites
            # the source after a .bak copy unless backup is off).
            if in_place:
                output_path = img_path
                if backup:
                    bak = img_path + '.bak'
                    if not os.path.exists(bak):
                        shutil.copy2(img_path, bak)
            else:
                output_path = os.path.join(
                    os.path.dirname(img_path),
                    'upscaled',
                    os.path.basename(img_path)
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            output_img.save(output_path)

            return True

        except Exception as e:
            logger.exception("Upscale failed: %s", img_path)
            if error_cb:
                error_cb(f"Error processing {img_path}: {str(e)}")
            return False

    def clear_gpu_memory(self):
        """Free the model and the CUDA cache. Called once per run, after
        the whole batch finishes (or is stopped)."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class SeedVR2Upscaler:
    """One SeedVR2 runner + per-image 4-phase pipeline (single-frame mode).

    seed: a fixed value (>=0) applies to every image; -1 draws a fresh
    random seed per image (ComfyUI-style randomize).
    """

    def __init__(self, scale_factor, device="cpu", seed=42,
                 color_correction="wavelet", tile_vae=True, min_size=0):
        self.scale_factor = float(scale_factor)
        self.min_size = int(min_size)  # >0 -> auto per-image scale factor
        self.device = device
        self.seed = int(seed)
        self.color_correction = color_correction
        # Tile BOTH VAE encode and decode to keep peak VRAM low on large images.
        self.tile_vae = tile_vae
        self.runner = None
        self.ctx = None
        from modules.Upscaler.seedvr2_upscaler import _SilentDebug
        self.debug = _SilentDebug()

    # ── model / runner lifecycle ──────────────────────────────────────────
    def _setup_runner(self, status_cb=None):
        from modules.Upscaler.seedvr2_upscaler import (
            SEEDVR2_DIT_MODEL, SEEDVR2_VAE_MODEL,
            SEEDVR2_DIT_CACHE_ID, SEEDVR2_VAE_CACHE_ID,
            are_seedvr2_models_downloaded, get_seedvr2_model_dir,
        )
        from modules.Upscaler.seedvr2.src.utils.constants import get_script_directory
        from modules.Upscaler.seedvr2.src.core.generation_utils import (
            setup_generation_context,
            prepare_runner,
            load_text_embeddings,
        )

        def _status(msg):
            if status_cb:
                status_cb(msg)

        if not are_seedvr2_models_downloaded():
            raise FileNotFoundError(
                f"SeedVR2 weights not found in {get_seedvr2_model_dir()}. "
                "Download the model first."
            )

        _status(f"Loading SeedVR2 (device: {self.device})...")
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
            dit_id=SEEDVR2_DIT_CACHE_ID,
            vae_id=SEEDVR2_VAE_CACHE_ID,
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
        _status("SeedVR2 models loaded")
        logger.info("SeedVR2: models loaded")

    def clear_gpu_memory(self):
        from modules.Upscaler.seedvr2_upscaler import (
            SEEDVR2_DIT_CACHE_ID, SEEDVR2_VAE_CACHE_ID,
        )
        from modules.Upscaler.seedvr2.src.optimization.memory_manager import complete_cleanup
        from modules.Upscaler.seedvr2.src.core.model_cache import get_global_cache

        if self.runner is not None:
            try:
                complete_cleanup(self.runner, debug=self.debug,
                                 dit_cache=False, vae_cache=False)
            except Exception as e:
                logger.warning("SeedVR2 cleanup: %s", e)
            self.runner = None
        self.ctx = None
        # complete_cleanup only drops the runner's OWN references. The
        # vendored code also keeps DiT + VAE in a process-wide singleton
        # (GlobalModelCache) - left alone, the ~6GB DiT + VAE stay in VRAM
        # until the app exits. Remove our entries so VRAM is fully freed.
        try:
            cache = get_global_cache()
            cache.remove_dit({"node_id": SEEDVR2_DIT_CACHE_ID}, debug=self.debug)
            cache.remove_vae({"node_id": SEEDVR2_VAE_CACHE_ID}, debug=self.debug)
        except Exception as e:
            logger.warning("SeedVR2 cache cleanup: %s", e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── per-image processing ──────────────────────────────────────────────
    def _load_frame(self, img_path):
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0
        frames = torch.from_numpy(arr[None, ...])  # [1, H, W, C] in [0, 1]
        return frames, (img.height, img.width)

    def _resolve_seed(self):
        """Per-image seed: self.seed when fixed, uniform random when -1.

        Mirrors ComfyUI's randomize (uniform draw; its frontend caps at
        2**53-1 only because of JS float precision). We cap at 2**32-1:
        numpy's legacy seed() rejects more - same max the vendored node
        widget uses.
        """
        if self.seed >= 0:
            return self.seed
        return random.randint(0, 2**32 - 1)

    def _process_one(self, img_path, seed):
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

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.ctx = encode_all_batches(
            self.runner, ctx=self.ctx, images=frames, debug=self.debug,
            batch_size=1, uniform_batch_size=False, seed=seed,
            progress_callback=None, temporal_overlap=0,
            resolution=resolution, max_resolution=0,
            input_noise_scale=0.0, color_correction=self.color_correction,
        )
        # cache_model=True: keep DiT/VAE on the GPU so the next image in a
        # batch reuses them (otherwise the VAE is freed after each image).
        self.ctx = upscale_all_batches(
            self.runner, ctx=self.ctx, debug=self.debug, progress_callback=None,
            seed=seed, latent_noise_scale=0.0, cache_model=True,
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

    def _save(self, img_path, out_img, in_place=False, backup=True):
        if in_place:
            output_path = img_path
            if backup:
                bak = img_path + '.bak'
                if not os.path.exists(bak):
                    shutil.copy2(img_path, bak)
        else:
            output_path = os.path.join(
                os.path.dirname(img_path), "upscaled", os.path.basename(img_path)
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_img.save(output_path)
        return output_path


# ── shared scan / summary helpers ─────────────────────────────────────────
def scan_images(folder, recursive=False, extensions=IMAGE_EXTENSIONS):
    """List the images to upscale, sorted for deterministic output.

    The engine's own 'upscaled' output subfolders are excluded so a
    re-run never upscales its own output.
    """
    found = []
    if recursive:
        for root, dirs, files in os.walk(folder):
            dirs[:] = [d for d in dirs if d != 'upscaled']
            for name in files:
                if os.path.splitext(name)[1].lower() in extensions:
                    found.append(os.path.join(root, name))
    else:
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if os.path.isfile(path) and os.path.splitext(name)[1].lower() in extensions:
                found.append(path)
    return sorted(found)


def _new_summary():
    return {"processed": 0, "skipped": 0, "errors": 0, "stopped": False,
            "items": []}


def _meets_min_size(path, min_size):
    """True if the image's longest side already reaches min_size; auto mode
    would render such images at scale 1.0, so the run loops skip them."""
    try:
        with Image.open(path) as img:
            return max(img.size) >= min_size
    except Exception:
        return False


def run(params, progress_cb=None, stop_check=None):
    """Upscale every image under params["folder"].

    params:
      folder: directory to scan (required).
      recursive: include subfolders (default False).
      model: "realesrgan" (default) or "seedvr2".
      model_path: RealESRGAN weights path (default: <app>/models/
          RealESRGAN_x4plus_anime_6B.pth).
      scale_factor: fixed scale (default 4.0, the GUI spin default).
      min_size: >0 switches to auto mode - smallest 0.1 step that brings
          the longest side to at least min_size (GUI resolution spin).
      device: "cuda"/"cpu" (default: auto-detect, the old worker behavior).
      seed: SeedVR2 only (default -1 = random per image, the GUI default).
      color_correction: SeedVR2 only (default "wavelet").
      tile_vae: SeedVR2 only (default True).
      dry_run: scan and report only, load no model, write nothing.
      in_place: overwrite each processed image at its own path instead of
          writing into <dir>/upscaled/ (default False, GUI behavior).
      backup: with in_place, first copy the original to <name>.<ext>.bak
          (default True; an existing .bak is never overwritten).
      extensions: image extensions to scan (default IMAGE_EXTENSIONS).

    Returns {"processed", "skipped", "errors", "stopped", "items"} with one
    item per file: {"path", "action": "processed|skipped|error|dry-run",
    "reason"}.
    """
    folder = params["folder"]
    model = params.get("model", "realesrgan")
    recursive = params.get("recursive", False)
    dry_run = params.get("dry_run", False)
    in_place = bool(params.get("in_place", False))
    backup = bool(params.get("backup", True))
    extensions = tuple(params.get("extensions", IMAGE_EXTENSIONS))

    files = scan_images(folder, recursive=recursive, extensions=extensions)
    total = len(files)
    summary = _new_summary()
    logger.info("Upscale: %d images found (model=%s, dry_run=%s)",
                total, model, dry_run)

    if dry_run:
        for path in files:
            try:
                with Image.open(path) as img:
                    w, h = img.size
            except Exception as e:
                summary["errors"] += 1
                summary["items"].append({"path": path, "action": "error",
                                         "reason": str(e)})
                continue
            if params.get("min_size", 0) > 0 and max(w, h) >= params["min_size"]:
                summary["skipped"] += 1
                summary["items"].append({"path": path, "action": "skipped",
                                         "reason": f"already {max(w, h)}px "
                                                  ">= min_size"})
                if progress_cb:
                    progress_cb(summary["processed"] + summary["skipped"]
                                + summary["errors"], total)
                continue
            if params.get("min_size", 0) > 0:
                steps = math.ceil(params["min_size"] * 10 / max(w, h) - 1e-9)
                scale = max(1.0, steps / 10.0)
            else:
                scale = float(params.get("scale_factor", 4.0))
            summary["processed"] += 1
            summary["items"].append({
                "path": path, "action": "dry-run",
                "reason": f"would upscale {w}x{h} -> "
                          f"{int(round(w * scale))}x{int(round(h * scale))} "
                          f"(x{scale:g}, {model})",
            })
            if progress_cb:
                progress_cb(summary["processed"] + summary["errors"], total)
        return summary

    if model == "realesrgan":
        model_path = params.get("model_path") or str(REAL_ESRGAN_MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"RealESRGAN weights not found at {model_path}. "
                "Download the model first (cli.py download-model).")
        upscaler = RealESRGANUpscaler(
            model_path,
            device=params.get("device"),
            scale_factor=params.get("scale_factor", 4.0),
            min_size=params.get("min_size", 0),
        )
        logger.info("Upscale: using device %s", upscaler.device)
        try:
            upscaler.load_model()
            for idx, path in enumerate(files, start=1):
                if stop_check is not None and stop_check():
                    summary["stopped"] = True
                    break
                logger.info("Upscaling [%d/%d]: %s", idx, total,
                            os.path.basename(path))
                if params.get("min_size", 0) > 0 and _meets_min_size(path, params["min_size"]):
                    summary["skipped"] += 1
                    summary["items"].append({"path": path, "action": "skipped",
                                             "reason": "already >= min_size"})
                    if progress_cb:
                        progress_cb(idx, total)
                    continue
                if upscaler.process_image(path, stop_check=stop_check,
                                          in_place=in_place, backup=backup):
                    summary["processed"] += 1
                    summary["items"].append({"path": path, "action": "processed",
                                             "reason": None})
                else:
                    summary["errors"] += 1
                    summary["items"].append({"path": path, "action": "error",
                                             "reason": "stop or processing failure"})
                if progress_cb:
                    progress_cb(idx, total)
        finally:
            # Free the model + CUDA cache once the WHOLE batch is done
            # (or stopped) - never per image.
            upscaler.clear_gpu_memory()
            logger.info("Upscale: GPU memory freed")
    elif model == "seedvr2":
        upscaler = SeedVR2Upscaler(
            scale_factor=params.get("scale_factor", 4.0),
            device=params.get("device") or "cpu",
            seed=params.get("seed", -1),
            color_correction=params.get("color_correction", "wavelet"),
            tile_vae=params.get("tile_vae", True),
            min_size=params.get("min_size", 0),
        )
        try:
            upscaler._setup_runner()
            for idx, path in enumerate(files, start=1):
                if stop_check is not None and stop_check():
                    summary["stopped"] = True
                    break
                logger.info("SeedVR2 processing [%d/%d]: %s", idx, total,
                            os.path.basename(path))
                if params.get("min_size", 0) > 0 and _meets_min_size(path, params["min_size"]):
                    summary["skipped"] += 1
                    summary["items"].append({"path": path, "action": "skipped",
                                             "reason": "already >= min_size"})
                    if progress_cb:
                        progress_cb(idx, total)
                    continue
                try:
                    seed = upscaler._resolve_seed()
                    out_img = upscaler._process_one(path, seed)
                    upscaler._save(path, out_img, in_place=in_place,
                                   backup=backup)
                    summary["processed"] += 1
                    summary["items"].append({"path": path, "action": "processed",
                                             "reason": None})
                except Exception as e:
                    logger.exception("SeedVR2 image failed: %s", path)
                    summary["errors"] += 1
                    summary["items"].append({"path": path, "action": "error",
                                             "reason": str(e)})
                if progress_cb:
                    progress_cb(idx, total)
        finally:
            # Free DiT + VAE + CUDA cache once the WHOLE batch is done
            # (or stopped) - never per image.
            upscaler.clear_gpu_memory()
            logger.info("SeedVR2: GPU memory freed")
    else:
        raise ValueError(f"Unknown model: {model!r} (expected 'realesrgan' "
                         f"or 'seedvr2')")
    return summary


def download_model(model="realesrgan", dest_path=None, status_cb=None,
                   stop_check=None):
    """Download a model's weights (this blocks; running it off the calling
    thread is the caller's concern). Returns True on success.

    RealESRGAN: single .pth via HTTP into a .part file, renamed into place
    so an interrupted download never leaves a partial file behind.
    SeedVR2: both weights via huggingface_hub into its cache.
    """
    def _status(msg):
        if status_cb:
            status_cb(msg)

    if model == "seedvr2":
        from modules.Upscaler.seedvr2_upscaler import (
            SEEDVR2_HF_REPO, SEEDVR2_DIT_MODEL, SEEDVR2_VAE_MODEL,
        )
        from huggingface_hub import hf_hub_download
        for filename in (SEEDVR2_DIT_MODEL, SEEDVR2_VAE_MODEL):
            if stop_check is not None and stop_check():
                return False
            _status(f"Downloading SeedVR2 weight: {filename}...")
            try:
                hf_hub_download(repo_id=SEEDVR2_HF_REPO, filename=filename)
            except Exception as e:
                _status(f"Download failed: {e}")
                return False
        return True

    # RealESRGAN
    import requests
    dest_path = dest_path or str(REAL_ESRGAN_MODEL_PATH)
    part_path = dest_path + '.part'
    try:
        response = requests.get(REAL_ESRGAN_MODEL_URL, stream=True)
        response.raise_for_status()
        total = int(response.headers.get('content-length', 0))
        downloaded = 0
        with open(part_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if stop_check is not None and stop_check():
                    break
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                if total:
                    _status(f"Downloading model... {downloaded >> 20} / "
                            f"{total >> 20} MB")
        if stop_check is not None and stop_check():
            return False
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        os.replace(part_path, dest_path)
        return True
    except Exception as e:
        _status(f"Download failed: {e}")
        return False
    finally:
        if os.path.exists(part_path):
            try:
                os.remove(part_path)
            except OSError:
                pass
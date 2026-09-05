"""
Natural-language captioning via a local, OpenAI-compatible LLM server
(LM Studio by default: http://127.0.0.1:1234, but any server exposing the
same /v1/chat/completions endpoint will work - text-generation-webui,
koboldcpp, Ollama's OpenAI-compat endpoint, etc.).

Design notes (see the conversation this was built from for the full spec):
- No system prompt is ever sent from here. The instructions for how to turn
  tags into a caption (and what to do when there are no tags at all) are
  expected to already be configured as the model's system prompt on the
  server itself (e.g. in LM Studio's per-model settings). We only send the
  image, plus the existing Danbooru tags as plain text if there are any.
- Every image is sent as a brand new request (no chat history is kept
  between images) to avoid burning context on unrelated prior turns.
- Sampling settings (temperature, top_p, etc.) are intentionally NOT sent -
  those are expected to already be configured on the server/model itself.
- Streaming is used so a Stop button can abort mid-generation. Closing the
  HTTP connection while it is still streaming causes LM Studio (and most
  other llama.cpp-based OpenAI-compatible servers) to stop generating on
  their end too, rather than just abandoning the response on our side.
- Any inline "thinking" the model emits (<think>...</think> and similar
  wrapper tags some reasoning models use) is stripped before saving - only
  the final answer is written to the .txt file.
"""

import base64
import io
import json
import os
import re
import threading

import requests
from PIL import Image
from PySide6.QtCore import QThread, Signal

from modules.logger import setup_logger

logger = setup_logger()

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

# Strips <think>...</think>, <thinking>...</thinking>, <reasoning>...</reasoning>,
# and <reflection>...</reflection> blocks some local reasoning models emit
# inline before their final answer.
_REASONING_TAG_PATTERN = re.compile(
    r"<\s*(think|thinking|reasoning|reflection)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


class LocalLLMCancelled(Exception):
    """Raised internally when a caption request is stopped by the user."""
    pass


class LocalLLMCaptioner:
    """Talks to an OpenAI-compatible /v1/chat/completions endpoint to turn
    an image (plus optional existing tags) into a natural-language caption."""

    def __init__(self, base_url, debug_mode=False, request_timeout=300):
        self.base_url = (base_url or "").rstrip('/')
        self.debug_mode = debug_mode
        self.request_timeout = request_timeout
        self._active_response = None
        self._lock = threading.Lock()
        # Kept only so UI code that checks `captioner.session` (a WD-tagger
        # concept) doesn't need special-casing everywhere it's read.
        self.session = True

    @property
    def endpoint(self):
        return f"{self.base_url}/v1/chat/completions"

    def _encode_image(self, image_path):
        """Re-encode any supported image as JPEG for broad compatibility
        with vision-capable local models, returned as a base64 data URL."""
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"

    def _build_messages(self, image_path, tags_text):
        content = [
            {"type": "image_url", "image_url": {"url": self._encode_image(image_path)}}
        ]
        # Only ever add the tags as plain text. Nothing else goes in the
        # user turn - the system prompt already configured on the server
        # is what tells the model what to do with (or without) them.
        if tags_text:
            content.append({"type": "text", "text": tags_text})
        return [{"role": "user", "content": content}]

    def stop_current_request(self):
        """Abort the in-flight HTTP request, if any. Closing the connection
        while LM Studio is still streaming causes it to stop generating
        server-side too (confirmed behavior: LM Studio logs "Client
        disconnected. Stopping generation..." when this happens)."""
        with self._lock:
            if self._active_response is not None:
                try:
                    self._active_response.close()
                except Exception:
                    pass

    def caption_image(self, image_path, tags_text=None, should_stop=None):
        """Generate a caption for a single image via a brand-new request.

        should_stop: optional zero-arg callable returning True if generation
        should be aborted early (checked between streamed chunks).
        """
        payload = {
            "model": "local-model",  # ignored by LM Studio when one model is loaded
            "messages": self._build_messages(image_path, tags_text),
            "stream": True,
        }

        if self.debug_mode:
            logger.debug(f"Sending {os.path.basename(image_path)} to {self.endpoint} "
                         f"(with tags: {bool(tags_text)})")

        response = requests.post(
            self.endpoint,
            json=payload,
            stream=True,
            timeout=self.request_timeout,
        )
        with self._lock:
            self._active_response = response

        try:
            response.raise_for_status()
            content_parts = []

            for raw_line in response.iter_lines(decode_unicode=True):
                if should_stop and should_stop():
                    response.close()
                    raise LocalLLMCancelled()

                if not raw_line or not raw_line.startswith("data:"):
                    continue

                data_str = raw_line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue

                delta = choices[0].get("delta") or {}
                # Deliberately only ever read "content" - some models stream
                # their thinking tokens through a separate "reasoning_content"
                # field, which we ignore entirely so it never reaches the file.
                piece = delta.get("content")
                if piece:
                    content_parts.append(piece)

            raw_caption = "".join(content_parts)
            final_caption = self._strip_reasoning(raw_caption).strip()

            if self.debug_mode:
                logger.debug(f"Final caption for {os.path.basename(image_path)}: {final_caption}")

            return final_caption
        finally:
            with self._lock:
                self._active_response = None

    @staticmethod
    def _strip_reasoning(text):
        """Remove inline <think>/<thinking>/<reasoning>/<reflection> blocks
        some local models emit before their final answer."""
        return _REASONING_TAG_PATTERN.sub("", text)


class NaturalLanguageCaptionThread(QThread):
    caption_generated = Signal(str, str)
    process_completed = Signal()
    error_occurred = Signal(str)
    stopped = Signal()

    TAG_CAPTIONS_DIRNAME = "Tag Captions"

    def __init__(self, captioner, folder_path, recursive=False):
        super().__init__()
        self.captioner = captioner
        self.folder_path = folder_path
        self.recursive = recursive
        self._stop_event = threading.Event()

    def request_stop(self):
        """Cooperative stop: flag the loop to exit on its next check, and
        immediately abort any in-flight request to the local model."""
        self._stop_event.set()
        self.captioner.stop_current_request()

    def _should_stop(self):
        return self._stop_event.is_set()

    def run(self):
        try:
            image_files = self._get_image_files(self.folder_path)
            total_files = len(image_files)
            logger.info(f"Starting natural language caption generation for {total_files} images")

            tag_captions_dir = os.path.join(self.folder_path, self.TAG_CAPTIONS_DIRNAME)

            for image_path in image_files:
                if self._should_stop():
                    logger.info("Natural language captioning stopped by user")
                    self.stopped.emit()
                    return

                try:
                    tags_text = self._relocate_and_read_tags(image_path, tag_captions_dir)

                    caption = self.captioner.caption_image(
                        image_path,
                        tags_text=tags_text,
                        should_stop=self._should_stop,
                    )

                    txt_path = os.path.splitext(image_path)[0] + '.txt'
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(caption + '\n')

                    self.caption_generated.emit(image_path, caption)

                except LocalLLMCancelled:
                    logger.info("Natural language captioning stopped by user")
                    self.stopped.emit()
                    return
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {str(e)}")
                    self.error_occurred.emit(f"Error processing {image_path}: {str(e)}")
                    continue

            logger.info("Natural language caption generation completed")
            self.process_completed.emit()

        except Exception as e:
            error_msg = f"Process error: {str(e)}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)

    def _relocate_and_read_tags(self, image_path, tag_captions_dir):
        """Move any existing Danbooru-tag .txt file beside this image out of
        the way into the 'Tag Captions' folder (mirroring subfolder structure
        when recursive), and return its contents so they can be handed to the
        model. Returns None if there are no existing tags for this image.

        Safe to re-run: if the tag file was already relocated by a previous
        pass, it's read from its new location instead of being moved again.
        """
        image_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        original_txt = os.path.join(image_dir, base_name + '.txt')

        relative_dir = os.path.relpath(image_dir, self.folder_path)
        mirrored_dir = os.path.normpath(os.path.join(tag_captions_dir, relative_dir))
        mirrored_txt = os.path.join(mirrored_dir, base_name + '.txt')

        if os.path.exists(original_txt):
            os.makedirs(mirrored_dir, exist_ok=True)
            with open(original_txt, 'r', encoding='utf-8') as f:
                tags_text = f.read().strip()
            if os.path.exists(mirrored_txt):
                os.remove(mirrored_txt)  # avoid clobber errors on a repeated run
            os.replace(original_txt, mirrored_txt)
            return tags_text if tags_text else None

        if os.path.exists(mirrored_txt):
            # Already relocated by an earlier run.
            with open(mirrored_txt, 'r', encoding='utf-8') as f:
                tags_text = f.read().strip()
            return tags_text if tags_text else None

        return None

    def _get_image_files(self, folder_path):
        image_files = []
        if self.recursive:
            for root, dirs, files in os.walk(folder_path):
                # Never descend into our own relocated-tags folder.
                dirs[:] = [d for d in dirs if d != self.TAG_CAPTIONS_DIRNAME]
                for file in files:
                    if file.lower().endswith(IMAGE_EXTENSIONS):
                        image_files.append(os.path.join(root, file))
        else:
            image_files = [
                os.path.join(folder_path, f) for f in os.listdir(folder_path)
                if os.path.isfile(os.path.join(folder_path, f)) and
                f.lower().endswith(IMAGE_EXTENSIONS)
            ]
        return image_files

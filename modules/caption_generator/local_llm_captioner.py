"""
Natural-language captioning via a local, OpenAI-compatible LLM server
(default: http://127.0.0.1:8890, but any server exposing the same
/v1/chat/completions endpoint will work - LM Studio, text-generation-webui,
koboldcpp, Ollama's OpenAI-compat endpoint, etc.).

Design notes (see the conversation this was built from for the full spec):
- A system prompt IS sent with every request, overriding any system prompt
  configured on the server. It is read from config.yaml at the app root,
  from the key matching the training type chosen in the UI
  ("caption_system_prompt_character" or "caption_system_prompt_style",
  edited by the user themselves). If that key is missing or empty, a
  built-in default captioning prompt is used instead so captioning never
  breaks.
- We send the image, plus the existing Danbooru tags as plain text if there
  are any, in the user turn.
- Every image is sent as a brand new request (no chat history is kept
  between images) to avoid burning context on unrelated prior turns.
- Sampling settings (temperature, top_p, etc.) are intentionally NOT sent -
  those are expected to already be configured on the server/model itself.
- Streaming is used so a Stop button can abort mid-generation. The whole
  HTTP exchange (post + body reads) runs on a short-lived daemon thread
  that feeds a queue the worker polls, so Stop works from the moment the
  request is sent and is acted on within ~0.2 s (requests cannot
  interrupt a post() blocked before the response headers arrive, and a
  blocking read can only be observed, not interrupted). Closing the HTTP
  connection while the server is still streaming causes LM Studio (and
  most other llama.cpp-based OpenAI-compatible servers) to stop generating
  on their end too, rather than just abandoning the response on our side.
- An optional prefix entered in the UI is prepended to each caption before
  it is saved, mirroring the "Prefix tags" behavior of the tag captioner
  (a comma is used as separator unless the prefix already ends with one).
- Any inline "thinking" the model emits (<think>...</think> and similar
  wrapper tags some reasoning models use) is stripped before saving - only
  the final answer is written to the .txt file.
"""

import base64
import io
import json
import os
import queue
import re
import socket
import threading
import time

import requests
from PIL import Image

from modules import config as app_config
from modules.logger import setup_logger
from modules.utils import IMAGE_EXTENSIONS

logger = setup_logger()

# The user-editable system prompts live in config.yaml at the app root,
# alongside tab_order, window_size and llm_api_key. Two prompts are
# selectable in the UI: one for character training, one for style
# training. The built-in defaults (app_config.DEFAULT_SYSTEM_PROMPT /
# app_config.DEFAULT_STYLE_SYSTEM_PROMPT) are written into config.yaml
# when the file is first generated, and used as a runtime fallback when a
# key is missing or empty.

# Strips <think>...</think>, <thinking>...</thinking>, <reasoning>...</reasoning>,
# and <reflection>...</reflection> blocks some local reasoning models emit
# inline before their final answer.
_REASONING_TAG_PATTERN = re.compile(
    r"<\s*(think|thinking|reasoning|reflection)\s*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)


def _extract_system_prompt(data):
    """Pull the system-prompt string out of parsed JSON.

    Accepts a bare JSON string ("..."), or an object with the text under one
    of the common keys. Returns None when nothing usable is found so the
    caller can fall back to DEFAULT_SYSTEM_PROMPT.
    """
    if isinstance(data, str):
        return data.strip() or None
    if isinstance(data, dict):
        for key in ("system_prompt", "system", "prompt", "content"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


class LocalLLMCancelled(Exception):
    """Raised internally when a caption request is stopped by the user."""
    pass


class LocalLLMCaptioner:
    """Talks to an OpenAI-compatible /v1/chat/completions endpoint to turn
    an image (plus optional existing tags) into a natural-language caption."""

    def __init__(self, base_url, api_key="", debug_mode=False, request_timeout=300,
                 system_prompt_key=app_config.SYSTEM_PROMPT_CONFIG_KEY_CHARACTER):
        self.base_url = (base_url or "").rstrip('/')
        self.api_key = (api_key or "").strip()
        self.debug_mode = debug_mode
        self.request_timeout = request_timeout
        # Which config.yaml key holds the system prompt for this captioner
        # (character vs style training, chosen in the UI).
        self.system_prompt_key = system_prompt_key
        self._active_response = None
        self._lock = threading.Lock()
        # (config-signature, prompt) so we re-read config.yaml only when it
        # changes, and re-emit the "falling back" warning only once.
        self._system_prompt_cache = None
        # Kept only so UI code that checks `captioner.session` (a WD-tagger
        # concept) doesn't need special-casing everywhere it's read.
        self.session = True

    @property
    def endpoint(self):
        return f"{self.base_url}/v1/chat/completions"

    def release(self):
        """No-op: the model lives on the external LLM server, not in this
        process (the user manages that server's VRAM). Returns False so
        the UI doesn't claim VRAM was freed.
        """
        return False

    # Vision models downscale inputs internally, so sending full-resolution
    # images mostly inflates the base64 payload and slows every request.
    # Capping the longest edge keeps quality while shrinking it a lot.
    MAX_ENCODE_EDGE = 1024

    def _encode_image(self, image_path):
        """Re-encode any supported image as JPEG for broad compatibility
        with vision-capable local models, returned as a base64 data URL."""
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            longest = max(img.size)
            if longest > self.MAX_ENCODE_EDGE:
                scale = self.MAX_ENCODE_EDGE / longest
                img = img.resize(
                    (max(1, int(img.width * scale)), max(1, int(img.height * scale))),
                    Image.Resampling.LANCZOS,
                )
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=95)
            encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"

    def _load_system_prompt(self):
        """Read the user's system prompt from config.yaml at the app root
        (the key chosen at construction: "caption_system_prompt_character"
        or "caption_system_prompt_style"). It is sent with every request and
        overrides any system prompt set on the server. Falls back to the
        built-in default for that training type when the key is missing or
        empty, and re-reads automatically when config.yaml changes, so edits
        take effect without a restart. (Legacy prompts are migrated into
        config.yaml by modules.config on first run.)"""
        try:
            signature = ("mtime", os.path.getmtime(app_config.CONFIG_FILE))
        except OSError:
            signature = ("missing", None)

        cached = self._system_prompt_cache
        if cached is not None and cached[0] == signature:
            return cached[1]

        prompt = None
        if signature[0] != "missing":
            value = app_config.get_value(self.system_prompt_key)
            if isinstance(value, str) and value.strip():
                prompt = value.strip()

        if not prompt:
            prompt = (app_config.DEFAULT_STYLE_SYSTEM_PROMPT
                      if self.system_prompt_key == app_config.SYSTEM_PROMPT_CONFIG_KEY_STYLE
                      else app_config.DEFAULT_SYSTEM_PROMPT)
            logger.warning(
                "No system prompt in %s (key %r) - using the built-in default.",
                app_config.CONFIG_FILE, self.system_prompt_key,
            )

        self._system_prompt_cache = (signature, prompt)
        if self.debug_mode:
            logger.debug("Using system prompt from %s", app_config.CONFIG_FILE)
        return prompt

    def _build_messages(self, image_path, tags_text):
        messages = []
        # The user-editable system prompt overrides whatever is configured
        # on the server.
        system_prompt = self._load_system_prompt()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        content = [
            {"type": "image_url", "image_url": {"url": self._encode_image(image_path)}}
        ]
        # Only ever add the tags as plain text in the user turn; the system
        # prompt is what tells the model what to do with (or without) them.
        if tags_text:
            content.append({"type": "text", "text": tags_text})
        messages.append({"role": "user", "content": content})
        return messages

    @staticmethod
    def _response_reader(response):
        """Best-effort access to the socket-level buffered reader of a
        streaming response (urllib3/http.client private attribute chain).
        Returns None if the chain doesn't match expectations - callers
        must degrade gracefully."""
        try:
            return response.raw._fp.fp
        except (AttributeError, TypeError):
            return None

    @staticmethod
    def _process_sse_line(line, content_parts, should_stop, response):
        """Process one SSE line of the caption stream, appending any
        content delta to content_parts. Returns True when the stream is
        finished ([DONE])."""
        if should_stop and should_stop():
            response.close()
            raise LocalLLMCancelled()

        if not line or not line.startswith("data:"):
            return False

        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            return True

        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            return False

        choices = chunk.get("choices") or []
        if not choices:
            return False

        delta = choices[0].get("delta") or {}
        # Deliberately only ever read "content" - some models stream
        # their thinking tokens through a separate "reasoning_content"
        # field, which we ignore entirely so it never reaches the file.
        piece = delta.get("content")
        if piece:
            content_parts.append(piece)
        return False

    def stop_current_request(self):
        """Abort the in-flight HTTP request, if any. Closing the connection
        while LM Studio is still streaming causes it to stop generating
        server-side too (confirmed behavior: LM Studio logs "Client
        disconnected. Stopping generation..." when this happens).

        The close runs on a short-lived daemon thread: on Windows,
        close() from a second thread can stall against a read() another
        thread is blocked in, and a Stop press must never block its
        caller. The worker notices the stop on its own within ~0.2 s via
        its queue poll, so this is a server-side-abort optimization, not
        the stop mechanism. Requests that are still waiting for their
        first byte have no response to close; the queue poll notices the
        stop flag itself."""
        with self._lock:
            response = self._active_response
        if response is None:
            return

        def _close():
            reader = self._response_reader(response)
            if reader is not None:
                try:
                    reader.raw._sock.shutdown(socket.SHUT_RDWR)
                except (AttributeError, OSError):
                    pass
            try:
                response.close()
            except Exception:
                pass

        threading.Thread(target=_close, daemon=True).start()

    def caption_image(self, image_path, tags_text=None, should_stop=None):
        """Generate a caption for a single image via a brand-new request.

        should_stop: optional zero-arg callable returning True if generation
        should be aborted early (checked every ~0.2 s, in both the header
        and the body phase).

        The whole HTTP exchange (post + streaming body reads) runs on a
        short-lived daemon thread that feeds a queue: requests offers no
        way to interrupt a post() blocked before the response headers
        arrive, and the body is read with read1() (returns as soon as any
        bytes arrive, unlike read(n), which blocks until n bytes
        accumulate). The worker polls the queue, so a stop is acted on
        within ~0.2 s at any point, and the caller is never blocked by
        the server.
        """
        payload = {
            "model": "local-model",  # ignored by LM Studio when one model is loaded
            "messages": self._build_messages(image_path, tags_text),
            "stream": True,
        }

        if self.debug_mode:
            logger.debug(f"Sending {os.path.basename(image_path)} to {self.endpoint} "
                         f"(with tags: {bool(tags_text)})")

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        stream_queue = queue.Queue()

        def _send_and_stream():
            try:
                response = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    stream=True,
                    timeout=(5, self.request_timeout),
                )
            except Exception as e:
                stream_queue.put(("error", e))
                return
            reader = self._response_reader(response)
            stream_queue.put(("response", response))
            try:
                if reader is not None:
                    while True:
                        chunk = reader.read1(65536)
                        if not chunk:
                            break
                        stream_queue.put(("chunk", chunk))
                else:
                    # Fallback (urllib3 internals changed): standard
                    # requests path, one line per item.
                    for line in response.iter_lines(decode_unicode=True):
                        stream_queue.put(("line", line))
                stream_queue.put(("eof", None))
            except Exception as e:
                stream_queue.put(("error", e))

        sender = threading.Thread(target=_send_and_stream, daemon=True)
        sender.start()

        while True:
            try:
                kind, value = stream_queue.get(timeout=0.2)
            except queue.Empty:
                if should_stop and should_stop():
                    raise LocalLLMCancelled()
                continue
            break

        if kind == "error":
            raise value

        response = value
        if should_stop and should_stop():
            response.close()
            raise LocalLLMCancelled()

        with self._lock:
            self._active_response = response

        try:
            response.raise_for_status()
            content_parts = []
            buf = b""
            done = False
            # Deadline anchored to the last real model output (a "data:"
            # SSE line). Keep-alive pings and blank lines do not count, so
            # a server that accepts the request but never generates is
            # aborted after request_timeout seconds of silence instead of
            # blocking the run forever.
            last_data = time.monotonic()
            while not done:
                try:
                    kind, value = stream_queue.get(timeout=0.2)
                except queue.Empty:
                    if should_stop and should_stop():
                        raise LocalLLMCancelled()
                    if time.monotonic() - last_data > self.request_timeout:
                        response.close()
                        raise TimeoutError(
                            f"no model output for {self.request_timeout}s")
                    continue
                if kind == "error":
                    raise value
                if kind == "eof":
                    break
                if kind == "line":
                    if value and value.startswith("data:"):
                        last_data = time.monotonic()
                    if self._process_sse_line(value, content_parts,
                                              should_stop, response):
                        done = True
                    continue
                buf += value
                while b"\n" in buf:
                    raw_line, buf = buf.split(b"\n", 1)
                    line = raw_line.decode("utf-8", "replace").rstrip("\r")
                    if line.startswith("data:"):
                        last_data = time.monotonic()
                    if self._process_sse_line(line, content_parts,
                                              should_stop, response):
                        done = True
                        break
            raw_caption = "".join(content_parts)
            final_caption = self._strip_reasoning(raw_caption).strip()

            if self.debug_mode:
                logger.debug(f"Final caption for {image_path}: {final_caption}")

            return final_caption
        except LocalLLMCancelled:
            raise
        except Exception:
            # A stop from the UI thread closes the socket under us,
            # which surfaces as a ConnectionError here - report it as
            # a stop, not as a per-image error.
            if should_stop and should_stop():
                raise LocalLLMCancelled()
            raise
        finally:
            with self._lock:
                self._active_response = None

    @staticmethod
    def _strip_reasoning(text):
        """Remove inline <think>/<thinking>/<reasoning>/<reflection> blocks
        some local models emit before their final answer."""
        return _REASONING_TAG_PATTERN.sub("", text)



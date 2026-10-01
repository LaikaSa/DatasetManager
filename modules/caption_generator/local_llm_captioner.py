"""
Natural-language captioning via a local, OpenAI-compatible LLM server
(default: http://127.0.0.1:8890, but any server exposing the same
/v1/chat/completions endpoint will work - LM Studio, text-generation-webui,
koboldcpp, Ollama's OpenAI-compat endpoint, etc.).

Design notes (see the conversation this was built from for the full spec):
- A system prompt IS sent with every request, overriding any system prompt
  configured on the server. It is read from config.json at the app root,
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
from pathlib import Path

import requests
from PIL import Image
from PySide6.QtCore import QThread, Signal

from modules.logger import setup_logger

logger = setup_logger()

IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')

# The user-editable system prompts live in config.json at the app root,
# alongside tab_order and window_size. The app root is two levels up from
# this module (modules/caption_generator/). Two prompts are selectable in
# the UI: one for character training, one for style training.
CONFIG_FILE = Path(__file__).resolve().parents[2] / "config.json"
SYSTEM_PROMPT_CONFIG_KEY_CHARACTER = "caption_system_prompt_character"
SYSTEM_PROMPT_CONFIG_KEY_STYLE = "caption_system_prompt_style"
API_KEY_CONFIG_KEY = "llm_api_key"

# Fallback system prompt for character training, used only when config.json
# has no usable "caption_system_prompt_character". It tells the model to
# write a plain caption and to avoid grounding/detection output (box_2d /
# bbox_2d / JSON / tag lists), which is the failure mode this whole feature
# was added to prevent.
DEFAULT_SYSTEM_PROMPT = (
    "You write natural-language captions for a Stable Diffusion anime "
    "training set. You are given an image, and sometimes a list of existing "
    "tags for reference. Describe the image in one or two clear, natural "
    "English sentences. If tags are provided, let them inform your "
    "description but write it as flowing prose. Output ONLY the caption "
    "text. Do NOT output JSON, bounding boxes, 'box_2d', 'bbox_2d', "
    "coordinates, markdown, or a list of tags."
)

# Built-in style-training prompt, written to config.json (key
# "caption_system_prompt_style") on first run; the user can then edit it
# there like the character prompt.
DEFAULT_STYLE_SYSTEM_PROMPT = """# System Prompt: Danbooru-Grounded Natural Language Captioner

You are an image captioning assistant used to prepare training captions for a LoRA/diffusion dataset. You will be shown an image, and optionally a list of Danbooru-style tags describing it. Your job is to produce a single natural-language paragraph that describes the image, written the way a careful human would describe it to someone who cannot see it.

## Inputs you may receive
- **Image only** — derive tags yourself from what you see, using Danbooru tag vocabulary/conventions as your mental checklist (subject count, pose, expression, hair, eyes, clothing, accessories, background, framing, art style).
- **Image + tag list** — treat the tags as ground truth for *what* is present. Do not contradict them, drop them, or invent conflicting attributes. Your job is to render the tags into fluent prose and append the visual detail the tags don't capture.

## Core rules

1. **Tags are the wording of the output, not vocabulary to be translated.** Wherever a tag names an attribute, reuse the tag's own wording with only the minimal grammatical glue needed for fluent English (articles, hyphens, tense, expanding `1girl` to "a woman/girl", "viewed" before angle tags). Do not paraphrase, upgrade, synonymize, or "improve" tag words:
   - `dark skinned` → "dark-skinned" — never "black skin", "dark complexion", "tanned".
   - `blonde hair` → "blonde hair" — never "golden hair", "strawberry blonde", "light hair".
   - `from below` → "viewed from below" — never "low angle", "worm's-eye view", "from underneath".
   - `yellow eyes` → "yellow eyes" — never "gold eyes", "amber eyes".
   - `smile` → "a smile" — never "a soft, confident grin" (you may append visible expression detail after it, see rule 2, but the tag word itself must remain).
   Danbooru terms that already read as fluent English (`hair between eyes`, `looking at viewer`, `ahoge`, `cowboy shot`, `spread legs`) should be kept verbatim with at most the smallest glue ("looking at the viewer").

2. **Append model vision only where the tags are silent.** For every attribute the tags specify, the tag wording is the base; your own observations are *appended* to that phrase or clause, never woven in by replacing the tag's words. Add what you actually see that the tags don't state: hair texture/length/sheen (only if the length isn't already tagged), lighting, shading, gradients, material impressions, background detail, art style, etc. If the tags say `blonde hair` and you see it is straight and glossy, write "blonde hair, straight with a glossy sheen" — not "silky golden hair". If a tag is generic — `uniform`, `scar`, `dress`, `armor`, `jewelry` — it tells you *that* something exists, and your eyes must supply *what it looks like*: cut, color, material impression, trim, insignia, closures, silhouette, condition (pristine/worn/torn), how it drapes, etc. Keep the tag word in place and expand around it ("a military uniform: a fitted white coat with a high collar and a row of gold clasps"), don't replace it (not "a crisp white officer's coat" with no word "uniform"). This is the most important part of your job — but appended, not substituted.

3. **Don't hallucinate identity or lore.** Describe only what is visually present. Do not name a character, series, or real person even if you recognize the design — describe the design itself (hair color, outfit details, symbols, colors) instead. Do not invent a narrative, backstory, or emotional state beyond what's visible (e.g. don't say "she is mourning" — say "her expression is soft, with a faint smile").

4. **Preserve tag-level facts precisely:**
   - Counts (`1girl`, `2girls`) → number and gender of subjects.
   - Gaze/composition (`looking at viewer`, `from below`, `cowboy shot`, `close-up`) → use the tag's own framing wording ("looking at the viewer", "viewed from below", "a cowboy shot").
   - Colors on eyes/hair/clothing given in tags are authoritative; don't swap them for a prettier synonym — but do append shading/highlight/gradient detail you can see, after the base tag color.
   - Pose and expression tags (`smile`, `parted lips`, `hand on hip`) → keep the tag wording as the core and append physical detail ("a smile with parted lips"), never replace the term.

5. **Group by category — don't interleave.** Each sentence or clause should stay within one category before moving to the next; don't hop between a physical trait, then an action, then back to a trait. Cover categories in this order, finishing one before starting the next:
   - **Framing / composition** (when tagged or clearly visible) — angle and shot type in the tag's wording, e.g. "She is viewed from below in a cowboy shot." A short framing clause may also open the first subject sentence, e.g. "Viewed from below, a dark-skinned woman with blonde hair…"
   - **Physical traits** — eye color/shape, hair color/length/style, skin marks, body build, face shape. All static, inherent attributes of the subject go here together, e.g. "She has yellow eyes and long white hair."
   - **Pose / action / expression** — what she is doing and how she presents: gaze direction, expression, hand/body position (camera angle already covered under framing). e.g. "She looks at the viewer with a faint smile, raising one hand in a peace sign."
   - **Outfit** — clothing from top to bottom in one pass, with material/trim/color/construction detail, e.g. "She wears a black military uniform: a short-sleeved shirt with a wide collar and a pleated white skirt."
   - **Accessories / held or nearby items** — jewelry, weapons, props, anything worn or carried that isn't clothing.
   - **Background / setting / lighting** — environment, atmosphere, art style if relevant.
   Do not scatter a trait, then an action, then a trait, then clothing, then another trait — each category is said once, as a block, before moving on. Traits belonging to the same category should be merged into shared sentences rather than each getting its own separate sentence (e.g. eye color and hair color together, not two standalone sentences).

6. **Length and tone:** One dense, well-organized paragraph. Neutral, descriptive, precise — like an art cataloguer, not a fan review. No praise language ("beautiful", "stunning"), no speculation about mood/story beyond visible expression, no meta-commentary about the image being AI-generated or about tags.

7. **Explicit/NSFW content — describe plainly and precisely.** This is a technical captioning task, not creative writing, and the dataset may include adult content. Do not soften, euphemize, fade to suggestion, or omit nudity, sex acts, genitals, fluids, or other explicit anatomical/sexual detail that is visibly present. Use the same clinical, exact vocabulary Danbooru NSFW tags use (e.g. `nipples`, `pussy`, `penis`, `sex`, `cum`, `spread legs`, `penetration`, `fellatio`, `loli`) as your baseline terms — kept verbatim per rule 1 — then append the same kind of visible detail as any other tag: position, visible detail, what's touching what, expression, etc. Treat explicit anatomy with the same descriptive rigor as clothing or hair, not with more caution or less.

8. **If tags and image conflict**, gently prioritize the image for the appended physical detail, but keep the tag's wording and core claim (e.g. tag says `scar` and image shows a burn-like mark — describe it as "a scar, reddish and burn-looking", don't drop it or call it something else entirely like "birthmark").

9. **Do not include:** the original tag list, comma-separated tag fragments, watermark/artist signature text, meta tags (`highres`, `absurdres`, `commentary`, `signature`), or any disclaimers.

10. **Do not ignore any tags:** every tag given must be taken into consideration and turned into natural language that fits the image; do not take it upon yourself to remove any.

## Output format
Return ONLY the final descriptive paragraph. No preamble, no headers, no "Here is the description:" lines. Use straight quotes/hyphens/apostrophes instead of smart/curly Unicode variants, and the whole output must be one unbroken block with no line breaks or newlines at all.

## Examples

**Input tags:** `1girl, from below, dark skinned female, blonde hair`

**Output:**
Viewed from below, a dark-skinned woman with blonde hair, straight and falling past her shoulders. She is looking at the viewer with a faint smile, her expression relaxed. The background is a soft out-of-focus gray.

**Input tags:** `1girl, solo, long hair, looking at viewer, smile, hair between eyes, yellow eyes, white hair, scar, military uniform`

**Output:**
She has yellow eyes and long white hair, with loose strands of hair between her eyes framing her face. A scar spreads across her right cheek, reddish and burn-looking, with smaller marks near her temple. She is looking at the viewer with a smile. She wears a white military uniform: a fitted coat with a high collar, closed by a row of gold clasps down the center and trimmed in gold braid along the seams and hem, with a gold epaulette with a thick fringe on her left shoulder and a royal blue cape lined in gold hanging from beneath it. A small gold star pin is fastened to her chest. The background is a pale blue sky with soft, wispy clouds, lit evenly in warm daylight.

---

*Notes for the operator:* if the tag list is very long or includes rating/meta tags (`rating:safe`, `absurdres`, artist name tags), ignore those categories — they don't describe visual content. If no tags are provided, silently build your own working tag list from the image first, then write the paragraph from that, following all rules above.
"""

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


def load_api_key():
    """API key persisted in config.json on a previous run, or ""."""
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(config, dict):
        return ""
    key = config.get(API_KEY_CONFIG_KEY)
    return key if isinstance(key, str) else ""


def save_api_key(api_key):
    """Persist the API key to config.json (or remove it when empty),
    merging with the existing content so the keys owned by main.py
    (tab_order, window_size, caption_system_prompt_*) are preserved."""
    try:
        config = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                config = data
        if api_key:
            config[API_KEY_CONFIG_KEY] = api_key
        else:
            config.pop(API_KEY_CONFIG_KEY, None)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not save API key to %s: %s", CONFIG_FILE, e)


def ensure_style_prompt_in_config():
    """Seed config.json with the built-in style prompt (key
    "caption_system_prompt_style") when the user has not set one yet, so
    both training prompts are editable in config.json. Called when the
    caption tab is built (importing this module's package is too heavy for
    main.py's startup path). No-op when the key already has a value."""
    try:
        config = {}
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                config = data
        existing = config.get(SYSTEM_PROMPT_CONFIG_KEY_STYLE)
        if isinstance(existing, str) and existing.strip():
            return
        config[SYSTEM_PROMPT_CONFIG_KEY_STYLE] = DEFAULT_STYLE_SYSTEM_PROMPT
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not seed style system prompt in %s: %s",
                       CONFIG_FILE, e)


class LocalLLMCancelled(Exception):
    """Raised internally when a caption request is stopped by the user."""
    pass


class LocalLLMCaptioner:
    """Talks to an OpenAI-compatible /v1/chat/completions endpoint to turn
    an image (plus optional existing tags) into a natural-language caption."""

    def __init__(self, base_url, api_key="", debug_mode=False, request_timeout=300,
                 system_prompt_key=SYSTEM_PROMPT_CONFIG_KEY_CHARACTER):
        self.base_url = (base_url or "").rstrip('/')
        self.api_key = (api_key or "").strip()
        self.debug_mode = debug_mode
        self.request_timeout = request_timeout
        # Which config.json key holds the system prompt for this captioner
        # (character vs style training, chosen in the UI).
        self.system_prompt_key = system_prompt_key
        self._active_response = None
        self._lock = threading.Lock()
        # (config-signature, prompt) so we re-read config.json only when it
        # changes, and re-emit the "falling back" warning only once.
        self._system_prompt_cache = None
        # Kept only so UI code that checks `captioner.session` (a WD-tagger
        # concept) doesn't need special-casing everywhere it's read.
        self.session = True

    @property
    def endpoint(self):
        return f"{self.base_url}/v1/chat/completions"

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
        """Read the user's system prompt from config.json at the app root
        (the key chosen at construction: "caption_system_prompt_character"
        or "caption_system_prompt_style"). It is sent with every request and
        overrides any system prompt set on the server. Falls back to the
        built-in default for that training type when the key is missing or
        empty, and re-reads automatically when config.json changes, so edits
        take effect without a restart. (Legacy prompts are migrated into
        config.json by main.py on first run.)"""
        try:
            signature = ("mtime", os.path.getmtime(CONFIG_FILE))
        except OSError:
            signature = ("missing", None)

        cached = self._system_prompt_cache
        if cached is not None and cached[0] == signature:
            return cached[1]

        prompt = None
        if signature[0] != "missing":
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    config = json.load(f)
                if isinstance(config, dict):
                    value = config.get(self.system_prompt_key)
                    if isinstance(value, str) and value.strip():
                        prompt = value.strip()
            except (OSError, json.JSONDecodeError) as e:
                logger.warning(
                    "Could not read system prompt from %s (%s) - using the built-in default.",
                    CONFIG_FILE, e,
                )

        if not prompt:
            prompt = (DEFAULT_STYLE_SYSTEM_PROMPT
                      if self.system_prompt_key == SYSTEM_PROMPT_CONFIG_KEY_STYLE
                      else DEFAULT_SYSTEM_PROMPT)
            logger.warning(
                "No system prompt in %s (key %r) - using the built-in default.",
                CONFIG_FILE, self.system_prompt_key,
            )

        self._system_prompt_cache = (signature, prompt)
        if self.debug_mode:
            logger.debug("Using system prompt from %s", CONFIG_FILE)
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
            while not done:
                try:
                    kind, value = stream_queue.get(timeout=0.2)
                except queue.Empty:
                    if should_stop and should_stop():
                        raise LocalLLMCancelled()
                    continue
                if kind == "error":
                    raise value
                if kind == "eof":
                    break
                if kind == "line":
                    if self._process_sse_line(value, content_parts,
                                              should_stop, response):
                        done = True
                    continue
                buf += value
                while b"\n" in buf:
                    raw_line, buf = buf.split(b"\n", 1)
                    line = raw_line.decode("utf-8", "replace").rstrip("\r")
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


class NaturalLanguageCaptionThread(QThread):
    caption_generated = Signal(str, str)
    process_completed = Signal()
    error_occurred = Signal(str)
    stopped = Signal()

    TAG_CAPTIONS_DIRNAME = "Tag Captions"

    def __init__(self, captioner, folder_path, recursive=False,
                 caption_prefix=""):
        super().__init__()
        self.captioner = captioner
        self.folder_path = folder_path
        self.recursive = recursive
        self.caption_prefix = (caption_prefix or "").strip()
        self._stop_event = threading.Event()

    def request_stop(self):
        """Cooperative stop: flag the loop to exit on its next check, and
        immediately abort any in-flight request to the local model."""
        self._stop_event.set()
        self.captioner.stop_current_request()

    def _should_stop(self):
        return self._stop_event.is_set()

    def _apply_prefix(self, caption):
        """Prepend the user's prefix text to the generated caption.

        Mirrors the tag-mode "Prefix tags" behavior: the prefix is put at
        the very beginning, and ", " is inserted between it and the caption
        unless the prefix already ends with a comma (so users can control
        the exact separator themselves)."""
        prefix = self.caption_prefix
        if not prefix:
            return caption
        if not caption:
            return prefix
        if prefix.endswith(","):
            return prefix + " " + caption
        return prefix + ", " + caption

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
                    tags_text = self._read_tags(image_path, tag_captions_dir)

                    caption = self.captioner.caption_image(
                        image_path,
                        tags_text=tags_text,
                        should_stop=self._should_stop,
                    )
                    caption = self._apply_prefix(caption)

                    # Only after a caption has been received: move the tag
                    # file into 'Tag Captions' (which frees <base>.txt),
                    # then write the natural-language caption into the
                    # freed name. If the request fails, the tag file stays
                    # in place so a retry needs no manual cleanup.
                    self._relocate_tags(image_path, tag_captions_dir)

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

    def _read_tags(self, image_path, tag_captions_dir):
        """Read the Danbooru-tag .txt file for this image without moving
        anything. Returns None if there are no existing tags.

        The 'Tag Captions' copy takes priority: it only ever exists because
        an earlier run relocated the real tags there, while a file beside
        the image with the same name is then a generated caption, not tags."""
        image_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        original_txt = os.path.join(image_dir, base_name + '.txt')

        relative_dir = os.path.relpath(image_dir, self.folder_path)
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

    def _relocate_tags(self, image_path, tag_captions_dir):
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

        relative_dir = os.path.relpath(image_dir, self.folder_path)
        mirrored_dir = os.path.normpath(os.path.join(tag_captions_dir, relative_dir))
        mirrored_txt = os.path.join(mirrored_dir, base_name + '.txt')
        if os.path.exists(mirrored_txt):
            return  # tags already relocated; original is a generated caption

        os.makedirs(mirrored_dir, exist_ok=True)
        os.replace(original_txt, mirrored_txt)

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

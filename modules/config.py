"""Single owner of config.yaml at the app root.

Every config read/write (tab order, window size, caption system prompts,
LLM API key) goes through this module, so there is exactly one writer
implementation and one place for the one-time migrations.
"""
import json
from pathlib import Path

import yaml

from modules.logger import setup_logger

logger = setup_logger()

CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.yaml"
# The old JSON config file, kept only for a one-time migration to
# config.yaml.
LEGACY_CONFIG_FILE = Path(__file__).resolve().parent.parent / "config.json"
DEFAULT_WINDOW_SIZE = [1500, 900]

# Legacy standalone caption system-prompt file, kept only for a one-time
# migration into config.yaml (key "caption_system_prompt_character").
LEGACY_SYSTEM_PROMPT_FILE = (
    Path(__file__).resolve().parent / "caption_generator" / "systemprompt.json"
)
# The caption system prompts live in config.yaml: one for character
# training, one for style training; the caption UI picks which is sent.
# "caption_system_prompt" is the pre-split key, migrated to the character
# key on first run.
SYSTEM_PROMPT_CONFIG_KEY = "caption_system_prompt"
SYSTEM_PROMPT_CONFIG_KEY_CHARACTER = "caption_system_prompt_character"
SYSTEM_PROMPT_CONFIG_KEY_STYLE = "caption_system_prompt_style"
API_KEY_CONFIG_KEY = "llm_api_key"

# Built-in default system prompts, written into config.yaml when the file
# is first generated (and filled into any older file missing the keys):
# one for character training, one for style training. The caption UI can
# edit both in config.yaml; the captioner falls back to these when a key
# is missing or empty.
DEFAULT_SYSTEM_PROMPT = """You write natural-language training captions for a character LoRA dataset. You are given an image, and optionally a list of Danbooru-style tags describing it. Describe the image in one neutral, precise paragraph, the way a careful human would describe it to someone who cannot see it.

The caption must be character-agnostic: it must stay valid no matter which character appears in the image. Do NOT describe the character's inherent physical appearance - hair (color, length, style), eyes (color, shape), face shape, skin marks (scars, freckles, tattoos), or body build. Refer to subjects only by number, gender, and a basic age category when clearly visible (e.g. "a woman", "a young girl", "a male").

When tags are provided, treat them as ground truth for what is present and turn them into natural language in the caption - every tag that fits the character-agnostic rule must be taken into consideration, none removed on your own. Tags about pose, action, expression, clothing, accessories, held items and background belong in the caption; tags about the character's identity or inherent appearance (hair, eyes, face, skin, body, character or series names) are used only to understand the image, not described.

Do describe everything else in the image: pose, action, expression, gaze direction, clothing from top to bottom with color, cut, material and construction detail, accessories and held items, and the background, setting and lighting. Do not name a character, series, or real person even if you recognize the design. Describe explicit content plainly and precisely, without softening or omitting it.

Output ONLY the caption: one unbroken paragraph with no line breaks, no preamble, no headers, no tag lists, no JSON, no bounding boxes, no disclaimers. Use straight quotes and hyphens, not smart Unicode variants.
"""

DEFAULT_STYLE_SYSTEM_PROMPT = """You write natural-language training captions for a style LoRA dataset. You are given an image, and optionally a list of Danbooru-style tags describing it. Describe the whole image in one neutral, precise paragraph, the way a careful human would describe it to someone who cannot see it - every detail that is visible.

Cover, in order: the subject(s) and their physical appearance (hair color, length and style, eye color, face, skin, body build), pose, action and expression, clothing from top to bottom with color, cut, material and construction detail, accessories and held items, and finally the background, setting, lighting and overall art style. Do not omit a detail because it seems minor. Do not name a character, series, or real person even if you recognize the design. Describe explicit content plainly and precisely, without softening or omitting it.

When tags are provided, treat them as ground truth for what is present: every tag given must be taken into consideration and turned into natural language that fits the image - do not take it upon yourself to remove any. Where a tag and your own observation differ, keep the tag's claim and append the visible detail.

Output ONLY the caption: one unbroken paragraph with no line breaks, no preamble, no headers, no tag lists, no JSON, no bounding boxes, no disclaimers. Use straight quotes and hyphens, not smart Unicode variants.
"""


def _default_config(tab_definitions):
    """Full set of config.yaml headers for a freshly generated file: tab
    order, window size, a blank API key, and both system prompts set to
    the built-in defaults."""
    return {
        "tab_order": [key for key, _ in tab_definitions],
        "window_size": list(DEFAULT_WINDOW_SIZE),
        "llm_api_key": "",
        "caption_system_prompt_character": DEFAULT_SYSTEM_PROMPT,
        "caption_system_prompt_style": DEFAULT_STYLE_SYSTEM_PROMPT,
    }


def save_config(config, path=CONFIG_FILE, merge=True):
    """Write config.yaml. With merge=True (the default), keys owned by other
    parts of the app (e.g. the caption API key) that are not in the
    in-memory dict are preserved. With merge=False the dict is written
    as-is - used by migrations, where the dict is the full image of the
    file and key deletions must be reflected on disk."""
    try:
        merged = {}
        if merge and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    merged = data
            except (OSError, yaml.YAMLError):
                pass
        merged.update(config)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(merged, f, allow_unicode=True, sort_keys=False,
                           width=1000)
    except OSError as e:
        logger.warning("Could not write config %s: %s", path, e)


def get_value(key, default=None, path=CONFIG_FILE):
    """Read one key from config.yaml (default when missing/unreadable)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return default
    if isinstance(data, dict):
        return data.get(key, default)
    return default


def set_value(key, value, path=CONFIG_FILE):
    """Set one key in config.yaml, preserving all other keys.
    Pass value=None to delete the key."""
    try:
        data = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                data = loaded
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False,
                           width=1000)
    except (OSError, yaml.YAMLError) as e:
        logger.warning("Could not save %r to %s: %s", key, path, e)


def load_api_key():
    """API key persisted in config.yaml on a previous run, or ""."""
    key = get_value(API_KEY_CONFIG_KEY)
    return key if isinstance(key, str) else ""


def save_api_key(api_key):
    """Persist the API key to config.yaml (blank string when empty, so the
    header always exists), preserving all other keys."""
    set_value(API_KEY_CONFIG_KEY, (api_key or "").strip())


def _legacy_tab_order():
    """One-time migration: tab order saved in the old QSettings (registry)
    store, if any. Returns a list of keys or None."""
    try:
        from PySide6.QtCore import QSettings
        saved = QSettings("DatasetManager", "ImageProcessingTool").value("tab_order", [])
    except Exception:
        return None
    if isinstance(saved, str):  # QSettings may return a single str for a 1-item list
        saved = [saved]
    return saved if isinstance(saved, list) and saved else None


def _read_legacy_system_prompt(legacy_path):
    """Read the legacy systemprompt.json as prompt text. Accepts plain text
    (the common case, e.g. a Markdown prompt), a bare JSON string, or a JSON
    object with a 'system_prompt' key. Returns None if unreadable or empty."""
    try:
        raw = legacy_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return raw  # plain text (e.g. Markdown)
    if isinstance(data, str):
        return data.strip() or None
    if isinstance(data, dict):
        value = data.get("system_prompt")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _migrate_system_prompts(config, path=CONFIG_FILE):
    """One-time migration of the caption system prompt into the split
    config.yaml keys. The legacy systemprompt.json (or the pre-split
    "caption_system_prompt" key) is moved to
    "caption_system_prompt_character", and the pre-split key is removed.
    The style prompt ("caption_system_prompt_style") is seeded by the
    caption tab itself, because importing its module is too heavy for the
    startup path. No-op when everything is already migrated."""
    changed = False
    if not config.get(SYSTEM_PROMPT_CONFIG_KEY_CHARACTER):
        prompt = _read_legacy_system_prompt(LEGACY_SYSTEM_PROMPT_FILE)
        if prompt:
            config[SYSTEM_PROMPT_CONFIG_KEY_CHARACTER] = prompt
            changed = True
            try:
                LEGACY_SYSTEM_PROMPT_FILE.unlink()
                logger.info("Migrated caption system prompt into %s and removed %s",
                            path, LEGACY_SYSTEM_PROMPT_FILE)
            except OSError as e:
                logger.warning("Migrated caption system prompt into %s but could not "
                               "remove the legacy file %s: %s", path,
                               LEGACY_SYSTEM_PROMPT_FILE, e)
    legacy = config.pop(SYSTEM_PROMPT_CONFIG_KEY, None)
    if legacy and not config.get(SYSTEM_PROMPT_CONFIG_KEY_CHARACTER):
        config[SYSTEM_PROMPT_CONFIG_KEY_CHARACTER] = legacy
    if legacy:
        changed = True  # pre-split key removed (or moved)
    if changed:
        save_config(config, path, merge=False)
    return config


def _migrate_legacy_json_config(path=CONFIG_FILE):
    """One-time migration: move the old config.json to config.yaml. When
    config.yaml already exists the JSON is stale and simply removed."""
    legacy = LEGACY_CONFIG_FILE
    if not legacy.exists():
        return
    try:
        if path.exists():
            legacy.unlink()
            logger.info("Removed stale %s (%s is current)", legacy, path)
            return
        with open(legacy, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
        save_config(data, path)
        legacy.unlink()
        logger.info("Migrated %s to %s", legacy, path)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not migrate %s to %s: %s", legacy, path, e)


def _ensure_required_keys(config, path=CONFIG_FILE):
    """Make sure every config.yaml header exists: keys missing from an
    older config file are filled in with their defaults (idempotent -
    no write once all keys are present)."""
    changed = False
    for key, default in (
        (SYSTEM_PROMPT_CONFIG_KEY_CHARACTER, DEFAULT_SYSTEM_PROMPT),
        (SYSTEM_PROMPT_CONFIG_KEY_STYLE, DEFAULT_STYLE_SYSTEM_PROMPT),
        (API_KEY_CONFIG_KEY, ""),
    ):
        if key not in config:
            config[key] = default
            changed = True
    if changed:
        save_config(config, path)


def load_config(tab_definitions=None, path=CONFIG_FILE):
    """Load config.yaml; auto-generate with defaults if missing or unreadable.

    The old config.json is migrated to config.yaml on first run. When the
    file is missing, a tab order saved in the old QSettings store is carried
    over as a one-time migration. A legacy systemprompt.json is likewise
    migrated into the "caption_system_prompt_character" key.

    Every header (tab_order, window_size, llm_api_key, and both system
    prompts) is guaranteed to exist after this call: freshly generated
    files contain it, older files are filled in with the defaults.
    """
    _migrate_legacy_json_config(path)
    config = None
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                config = data
            else:
                logger.warning("Config %s is not a mapping; regenerating", path)
        except (OSError, yaml.YAMLError) as e:
            logger.warning("Could not read config %s (%s); regenerating", path, e)
    if config is None:
        config = _default_config(tab_definitions)
        if not path.exists():
            legacy = _legacy_tab_order()
            if legacy:
                config["tab_order"] = legacy  # validated against tab_definitions on use
        save_config(config, path)
    _migrate_system_prompts(config, path)
    _ensure_required_keys(config, path)
    return config
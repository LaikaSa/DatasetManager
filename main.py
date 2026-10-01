import importlib
import json
import sys
import threading
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QMenu,
                              QToolButton, QWidget)
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDragMoveEvent, QAction
from PySide6.QtCore import Qt, QSettings, QTimer
from modules.logger import setup_logger
from modules import settings as app_settings
# NOTE: the tab modules are intentionally NOT imported here. They pull in
# heavy dependencies (torch ~2 s, pandas, onnxruntime, cv2, ...) which used to
# delay the GUI appearing at startup. Each tab is built lazily on first visit
# (see _build_tab / _ensure_tab_built), and its module is pre-warmed on a
# background thread after startup (see _start_prewarm) so the first click
# doesn't freeze the UI while Python imports the dependencies.
import os  # Add this for path operations
from pathlib import Path
logger = setup_logger()

# tab key -> module path, shared by _build_tab and the pre-warm worker
TAB_MODULE_PATHS = {
    "duplicate": "modules.duplicate_detector",
    "resizer": "modules.image_resizer",
    "upscaler": "modules.Upscaler.upscaler",
    "caption": "modules.caption_generator",
    "tag_editor": "modules.tag_editor",
    "conversion": "modules.Conversion_Tools",
}

# User preferences (tab order, window size) live in config.json at the
# app root; auto-generated with defaults on first run.
CONFIG_FILE = Path(__file__).resolve().parent / "config.json"
DEFAULT_WINDOW_SIZE = [1500, 900]


def _default_config(tab_definitions):
    return {
        "tab_order": [key for key, _ in tab_definitions],
        "window_size": list(DEFAULT_WINDOW_SIZE),
    }


def _save_config_file(config, path=CONFIG_FILE):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")
    except OSError as e:
        logger.warning("Could not write config %s: %s", path, e)


def _legacy_tab_order():
    """One-time migration: tab order saved in the old QSettings (registry)
    store, if any. Returns a list of keys or None."""
    try:
        saved = QSettings("DatasetManager", "ImageProcessingTool").value("tab_order", [])
    except Exception:
        return None
    if isinstance(saved, str):  # QSettings may return a single str for a 1-item list
        saved = [saved]
    return saved if isinstance(saved, list) and saved else None

# Legacy standalone caption system-prompt file, kept only for a one-time
# migration into config.json (key "caption_system_prompt").
LEGACY_SYSTEM_PROMPT_FILE = (
    Path(__file__).resolve().parent / "modules" / "caption_generator" / "systemprompt.json"
)
SYSTEM_PROMPT_CONFIG_KEY = "caption_system_prompt"


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


def _migrate_system_prompt(config, path=CONFIG_FILE):
    """One-time migration: move the caption system prompt out of the legacy
    systemprompt.json into config.json (key "caption_system_prompt"), then
    remove the legacy file. No-op when the key is already set or there is no
    legacy file to migrate."""
    if config.get(SYSTEM_PROMPT_CONFIG_KEY):
        return config
    if not LEGACY_SYSTEM_PROMPT_FILE.exists():
        return config
    prompt = _read_legacy_system_prompt(LEGACY_SYSTEM_PROMPT_FILE)
    if not prompt:
        return config
    config[SYSTEM_PROMPT_CONFIG_KEY] = prompt
    _save_config_file(config, path)
    try:
        LEGACY_SYSTEM_PROMPT_FILE.unlink()
        logger.info("Migrated caption system prompt into %s and removed %s",
                    path, LEGACY_SYSTEM_PROMPT_FILE)
    except OSError as e:
        logger.warning("Migrated caption system prompt into %s but could not "
                       "remove the legacy file %s: %s", path,
                       LEGACY_SYSTEM_PROMPT_FILE, e)
    return config


def _load_config_file(path=CONFIG_FILE, tab_definitions=None):
    """Load config.json; auto-generate with defaults if missing or unreadable.

    When the file is missing, a tab order saved in the old QSettings store
    is carried over as a one-time migration. A legacy systemprompt.json is
    likewise migrated into the "caption_system_prompt" key on first run.
    """
    config = None
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                config = data
            else:
                logger.warning("Config %s is not a JSON object; regenerating", path)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Could not read config %s (%s); regenerating", path, e)
    if config is None:
        config = _default_config(tab_definitions)
        if not path.exists():
            legacy = _legacy_tab_order()
            if legacy:
                config["tab_order"] = legacy  # validated against tab_definitions on use
        _save_config_file(config, path)
    _migrate_system_prompt(config, path)
    return config

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing main application")
        self.setWindowTitle("Image Processing Tool")
        self.setMinimumSize(1000, 600)
        self.setAcceptDrops(True)  # Enable drop for main window

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setMovable(True)  # Allow click-and-drag tab reordering
        self.setCentralWidget(self.tabs)

        # (stable key, display label) in the default order. The widgets
        # themselves are created lazily when the tab is first shown.
        self.tab_definitions = [
            ("duplicate", "Duplicate Detection"),
            ("resizer", "Resizer"),
            ("upscaler", "Upscaler"),
            ("caption", "Caption Generator"),
            ("tag_editor", "Tags Editor"),
            ("conversion", "Conversion Tools"),
        ]
        # User preferences (tab order, window size) live in config.json at
        # the app root; auto-generated with defaults on first run.
        self.config = _load_config_file(tab_definitions=self.tab_definitions)
        # Debounced persistence of the window size the user drags it to,
        # so a corner drag writes once, not once per pixel.
        self._size_save_timer = QTimer(self)
        self._size_save_timer.setSingleShot(True)
        self._size_save_timer.timeout.connect(self._persist_window_size)
        self._apply_saved_size()

        self._building_tab = False  # re-entrancy guard for _ensure_tab_built
        self._built_keys = set()    # tab keys whose real widget is already built
        self._add_tabs_in_saved_order()

        # Persist the new order whenever the user drags a tab into place
        self.tabs.tabBar().tabMoved.connect(self._save_tab_order)

        # Settings cog (top-right): choose the compute device for models
        self._build_settings_button()

        # Import the heavy tab modules in the background now, so clicking a
        # tab later doesn't freeze the UI on its first visit.
        self._start_prewarm()

    def _build_settings_button(self):
        """Create the settings cog at the right end of the tab bar row."""
        self.menuBar().setNativeMenuBar(False)
        self._settings_button = QToolButton()
        self._settings_button.setIcon(app_settings.create_settings_icon())
        self._settings_button.setToolTip("Settings - compute device (GPU/CPU), parallel loading")
        self._settings_button.setPopupMode(QToolButton.InstantPopup)
        self._device_menu = QMenu(self._settings_button)
        self._settings_button.setMenu(self._device_menu)
        # Sit on the tabs row, right-aligned, instead of the menu-bar corner.
        self.tabs.setCornerWidget(self._settings_button, Qt.Corner.TopRightCorner)
        # Rebuild lazily on first open: list_devices() imports torch, and we
        # don't want that cost (or even the import) at startup.
        self._device_menu.aboutToShow.connect(self._rebuild_device_menu)

    def _rebuild_device_menu(self):
        """(Re)populate the device menu, checking the currently selected one."""
        self._device_menu.clear()
        header = self._device_menu.addAction("Compute device")
        header.setEnabled(False)

        selected = app_settings.get_selected_device_id()
        for dev in app_settings.list_devices():
            action = QAction(dev["label"], self._device_menu)
            action.setCheckable(True)
            action.setChecked(dev["id"] == selected)
            action.triggered.connect(
                lambda checked=False, d=dev: self._select_device(d["id"])
            )
            self._device_menu.addAction(action)

        # App-wide parallel-processing switch (used by the tag editor's
        # folder loading and the conversion tools' batch operations).
        self._device_menu.addSeparator()
        parallel = QAction("Parallel Loading", self._device_menu)
        parallel.setCheckable(True)
        parallel.setChecked(app_settings.is_parallel_enabled())
        parallel.setToolTip(
            "Use multiple CPU cores to speed up loading and batch processing "
            "(may use more memory)"
        )
        parallel.toggled.connect(self._set_parallel_enabled)
        self._device_menu.addAction(parallel)

    def _select_device(self, device_id):
        app_settings.set_selected_device_id(device_id)
        logger.info("Compute device set to: %s", app_settings.device_label(device_id))
        self._rebuild_device_menu()

    def _set_parallel_enabled(self, enabled):
        app_settings.set_parallel_enabled(enabled)
        logger.info("Parallel loading set to: %s", "on" if enabled else "off")

    def _build_tab(self, key):
        """Import and construct the tab widget for a key (deferred imports).

        The module is normally already imported by the pre-warm thread; if the
        user clicks before it gets there, the import lock makes this call wait
        briefly and reuse the in-progress import.
        """
        module = importlib.import_module(TAB_MODULE_PATHS[key])
        if key == "duplicate":
            return module.DuplicateDetectorTab()
        if key == "resizer":
            return module.ImageResizerTab()
        if key == "upscaler":
            return module.UpscalerTab()
        if key == "caption":
            return module.CaptionGeneratorTab()
        if key == "tag_editor":
            return module.TagEditorTab()
        if key == "conversion":
            return module.ConversionTab()
        raise KeyError(f"Unknown tab key: {key}")

    def _start_prewarm(self):
        """Import all tab modules on a daemon thread so first-visit clicks are fast.

        Only the imports run off the main thread (never widget construction,
        which must happen in the GUI thread). If the user opens a tab before
        its import finishes, the import lock makes _build_tab block until the
        background import completes and then reuse it - no double import.
        """
        def worker():
            for key, path in TAB_MODULE_PATHS.items():
                if key in self._built_keys:
                    continue  # already imported by building that tab
                try:
                    importlib.import_module(path)
                except Exception as e:  # keep pre-warming the rest
                    logger.warning("Pre-warm import of %s failed: %s", path, e)
            logger.info("Tab module pre-warm complete")

        t = threading.Thread(target=worker, name="tab-prewarm", daemon=True)
        t.start()

    def _add_tabs_in_saved_order(self):
        """Add tabs using the order saved from a previous session, if any.

        Light placeholder widgets are added first; the real tab (and its
        module, which may take seconds to import) is only built when the
        tab is first selected.
        """
        label_by_key = {key: label for key, label in self.tab_definitions}
        saved_order = self.config.get("tab_order", [])
        if isinstance(saved_order, str):  # QSettings may return a single str for a 1-item list
            saved_order = [saved_order]

        # Keep saved keys that still exist, then append any tabs missing from the saved order
        ordered_keys = [key for key in saved_order if key in label_by_key]
        ordered_keys += [key for key, _ in self.tab_definitions if key not in ordered_keys]

        self.tab_keys = []  # index -> key, kept in sync with the actual visual tab order
        self.tab_widgets = {}  # key -> current widget (placeholder until built)
        for key in ordered_keys:
            placeholder = QWidget()
            self.tabs.addTab(placeholder, label_by_key[key])
            self.tab_widgets[key] = placeholder
            self.tab_keys.append(key)

        self.tabs.currentChanged.connect(self._ensure_tab_built)
        self._ensure_tab_built(self.tabs.currentIndex())

    def _ensure_tab_built(self, index):
        """Replace the placeholder at `index` with the real tab, once."""
        # insertTab() below emits currentChanged synchronously - the guard
        # prevents that signal from re-entering this method before
        # tab_widgets[key] points at the real widget (infinite recursion).
        if self._building_tab or index < 0 or index >= self.tabs.count():
            return
        key = self.tab_keys[index]
        if key in self._built_keys:
            return

        self._building_tab = True
        try:
            real = self._build_tab(key)
            label = self.tabs.tabText(index)
            self.tabs.insertTab(index, real, label)
            self.tabs.removeTab(index + 1)  # drop the placeholder
            self.tab_widgets[key] = real
            self._built_keys.add(key)
            self.tabs.setCurrentIndex(index)
        finally:
            self._building_tab = False

    def _save_tab_order(self, *_args):
        """Recompute tab order from current widget positions and persist it."""
        # Use indexOf() (resolves by C++ pointer) - Python wrapper identity
        # is not stable for reparented widgets, so a dict keyed on wrappers
        # would raise KeyError.
        self.tab_keys = [None] * self.tabs.count()
        for key, widget in self.tab_widgets.items():
            i = self.tabs.indexOf(widget)
            if i >= 0:
                self.tab_keys[i] = key
        self.config["tab_order"] = self.tab_keys
        _save_config_file(self.config)

    def _apply_saved_size(self):
        """Restore the window size the user last set, else the default."""
        size = self.config.get("window_size")
        if (isinstance(size, (list, tuple)) and len(size) == 2
                and all(isinstance(v, int) and v > 0 for v in size)):
            self.resize(size[0], size[1])
        else:
            self.resize(DEFAULT_WINDOW_SIZE[0], DEFAULT_WINDOW_SIZE[1])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Skip while maximized/minimized so we always store the last
        # *normal* size, not the work-area size of a maximized window.
        if self.isMaximized() or self.isMinimized():
            return
        self._size_save_timer.start(300)

    def _persist_window_size(self):
        self.config["window_size"] = [self.width(), self.height()]
        _save_config_file(self.config)

    def closeEvent(self, event):
        # Save the final size now (the debounce timer may not have fired
        # yet for a quick drag-then-close).
        if not self.isMaximized() and not self.isMinimized():
            self._persist_window_size()
        event.accept()

    def _current_tab_accepts_drops(self):
        # Only the Upscaler tab has a dropEvent, so only show the "can drop"
        # cursor over it - otherwise drops on other tabs are silently ignored.
        from modules.Upscaler.upscaler import UpscalerTab
        return isinstance(self.tabs.currentWidget(), UpscalerTab)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and self._current_tab_accepts_drops():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls() and self._current_tab_accepts_drops():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        # Get the current active tab
        current_tab = self.tabs.currentWidget()
        
        # Handle the drop based on the current tab
        # (deferred import: only reached when a drop actually happens)
        from modules.Upscaler.upscaler import UpscalerTab
        if isinstance(current_tab, UpscalerTab):
            current_subtab = current_tab.tabs.currentWidget()
            tab_index = current_tab.tabs.currentIndex()
            
            if tab_index == 0:  # Single Image tab
                # Handle single image drop
                if event.mimeData().hasUrls():
                    url = event.mimeData().urls()[0]
                    path = url.toLocalFile()
                    if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        current_subtab.input_path.setText(path)
                        event.accept()
            elif tab_index == 1:  # Multiple Images tab
                # Handle multiple images drop
                files = []
                for url in event.mimeData().urls():
                    path = url.toLocalFile()
                    if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        files.append(path)
                    elif os.path.isdir(path):
                        current_subtab.dir_input.setText(path)
                        current_subtab.process_directory(path)
                        event.accept()
                        return
                
                if files:
                    current_subtab.selected_paths = files
                    current_subtab.refresh_list()
                    current_subtab.parent.check_input()
                    event.accept()

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    logger.info("Starting application")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    logger.info("Application started successfully")
    sys.exit(app.exec())
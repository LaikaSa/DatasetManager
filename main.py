import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QMenu,
                              QToolButton, QWidget)
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDragMoveEvent, QAction
from PySide6.QtCore import Qt, QSettings
from modules.logger import setup_logger
from modules import settings as app_settings
# NOTE: the tab modules are intentionally NOT imported here. They pull in
# heavy dependencies (torch ~2 s, pandas, onnxruntime, cv2, ...) which used to
# delay the GUI appearing at startup. Each tab is built lazily on first visit
# (see _build_tab / _ensure_tab_built).
import os  # Add this for path operations
logger = setup_logger()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing main application")
        self.setWindowTitle("Image Processing Tool")
        self.setMinimumSize(1000, 600)
        self.setAcceptDrops(True)  # Enable drop for main window

        # Used to remember the user's preferred tab order between sessions
        self.settings = QSettings("DatasetManager", "ImageProcessingTool")

        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.setMovable(True)  # Allow click-and-drag tab reordering
        self.setCentralWidget(self.tabs)

        # (stable key, display label) in the default order. The widgets
        # themselves are created lazily when the tab is first shown.
        self.tab_definitions = [
            ("duplicate", "Duplicate Detection"),
            ("resizer", "Image Resizer"),
            ("upscaler", "Upscaler"),
            ("caption", "Caption Generator"),
            ("tag_editor", "Tags Editor"),
            ("conversion", "Conversion Tools"),
        ]

        self._building_tab = False  # re-entrancy guard for _ensure_tab_built
        self._built_keys = set()    # tab keys whose real widget is already built
        self._add_tabs_in_saved_order()

        # Persist the new order whenever the user drags a tab into place
        self.tabs.tabBar().tabMoved.connect(self._save_tab_order)

        # Settings cog (top-right): choose the compute device for models
        self._build_settings_button()

    def _build_settings_button(self):
        """Create the settings cog in the menu-bar corner with a device picker."""
        self.menuBar().setNativeMenuBar(False)
        self._settings_button = QToolButton()
        self._settings_button.setIcon(app_settings.create_settings_icon())
        self._settings_button.setToolTip("Settings - choose compute device (GPU/CPU)")
        self._settings_button.setPopupMode(QToolButton.InstantPopup)
        self._device_menu = QMenu(self._settings_button)
        self._settings_button.setMenu(self._device_menu)
        self.menuBar().setCornerWidget(self._settings_button, Qt.Corner.TopRightCorner)
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

    def _select_device(self, device_id):
        app_settings.set_selected_device_id(device_id)
        logger.info("Compute device set to: %s", app_settings.device_label(device_id))
        self._rebuild_device_menu()

    def _build_tab(self, key):
        """Import and construct the tab widget for a key (deferred imports)."""
        if key == "duplicate":
            from modules.duplicate_detector import DuplicateDetectorTab
            return DuplicateDetectorTab()
        if key == "resizer":
            from modules.image_resizer import ImageResizerTab
            return ImageResizerTab()
        if key == "upscaler":
            from modules.Upscaler.upscaler import UpscalerTab
            return UpscalerTab()
        if key == "caption":
            from modules.caption_generator import CaptionGeneratorTab
            return CaptionGeneratorTab()
        if key == "tag_editor":
            from modules.tag_editor import TagEditorTab
            return TagEditorTab()
        if key == "conversion":
            from modules.Conversion_Tools import ConversionTab
            return ConversionTab()
        raise KeyError(f"Unknown tab key: {key}")

    def _add_tabs_in_saved_order(self):
        """Add tabs using the order saved from a previous session, if any.

        Light placeholder widgets are added first; the real tab (and its
        module, which may take seconds to import) is only built when the
        tab is first selected.
        """
        label_by_key = {key: label for key, label in self.tab_definitions}
        saved_order = self.settings.value("tab_order", [])
        if isinstance(saved_order, str):  # QSettings may return a single str for a 1-item list
            saved_order = [saved_order]

        # Keep saved keys that still exist, then append any tabs missing from the saved order
        ordered_keys = [key for key in saved_order if key in label_by_key]
        ordered_keys += [key for key, _ in self.tab_definitions if key not in ordered_keys]

        self.tab_keys = []  # index -> key, kept in sync with the actual visual tab order
        self.tab_widgets = {}  # key -> current widget (placeholder until built)
        for key in ordered_keys:
            self.tabs.addTab(QWidget(), label_by_key[key])
            self.tab_widgets[key] = self.tabs.currentWidget()
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
        self.settings.setValue("tab_order", self.tab_keys)

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
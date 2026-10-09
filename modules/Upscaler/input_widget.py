"""Input folder/file selector for the upscaler tab.

Standalone widget: no back-reference to the tab. The tab is told about
changes via the ``input_changed`` signal, and the resolution filter is
pushed in via ``set_resolution_filter``.
"""
import os

from PIL import Image
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QToolButton, QCheckBox, QLineEdit,
                              QListWidget, QListWidgetItem, QFileDialog,
                              QLabel, QMenu)
from PySide6.QtCore import Qt, Signal

from modules.logger import setup_logger
from modules.utils import IMAGE_EXTENSIONS, open_in_folder

logger = setup_logger()


class InputWidget(QWidget):
    input_changed = Signal()

    def __init__(self):
        super().__init__()
        self.input_type = None
        self.selected_paths = []
        self._size_cache = {}  # path -> (w, h); avoid re-opening files on every filter pass
        self._filter_enabled = False
        self._filter_max_res = 0
        self.init_ui()


    def set_editing_enabled(self, enabled):
        """Enable/disable the path row and the (tab-placed) list panel."""
        self.setEnabled(enabled)
        self.list_panel.setEnabled(enabled)

    def _get_size(self, p):
        if p not in self._size_cache:
            with Image.open(p) as img:
                self._size_cache[p] = img.size
        return self._size_cache[p]

    def init_ui(self):
        # This widget is the top path-input row; the tab places it.
        path_layout = QHBoxLayout(self)
        path_layout.setContentsMargins(0, 0, 0, 0)
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter image path or folder path, or drag & drop here...")
        self.path_input.setMinimumWidth(300)
        self.path_input.setMaximumWidth(400)
        self.browse_btn = QToolButton()
        self.browse_btn.setText("Browse")
        self.browse_btn.setPopupMode(QToolButton.InstantPopup)
        browse_menu = QMenu(self.browse_btn)
        self.browse_file_action = browse_menu.addAction("Image file...")
        self.browse_folder_action = browse_menu.addAction("Folder...")
        self.browse_btn.setMenu(browse_menu)
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.browse_btn)
        self.recursive_cb = QCheckBox("Recursive")
        self.recursive_cb.setToolTip("Include images from subfolders")
        path_layout.addWidget(self.recursive_cb)
        path_layout.addStretch()

        self.browse_file_action.triggered.connect(self.browse_file)
        self.browse_folder_action.triggered.connect(self.browse_folder)
        self.path_input.textChanged.connect(self.on_path_changed)

        # File list panel: a child of this widget (so enabling/disabling
        # propagates to it) but placed by the tab, below the controls.
        self.list_panel = QWidget(self)
        panel_layout = QVBoxLayout(self.list_panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)

        # Info label
        self.status_label = QLabel("No input selected")
        self.status_label.setAlignment(Qt.AlignCenter)

        # File list
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(180)
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.show_context_menu)
        self.file_list.setUniformItemSizes(True)

        panel_layout.addWidget(self.status_label)
        panel_layout.addWidget(self.file_list)

    # ── path typing / pasting ──────────────────────────────────────────────
    def on_path_changed(self, text):
        text = text.strip()
        if not text:
            self.input_type = None
            self.selected_paths = []
            self._size_cache.clear()
            self.refresh_list()
            self.input_changed.emit()
            return
        if os.path.isfile(text) and text.lower().endswith(IMAGE_EXTENSIONS):
            self.input_type = 'file'
            self.selected_paths = [text]
            self._size_cache.clear()
            self.refresh_list()
            self.input_changed.emit()
        elif os.path.isdir(text):
            self.load_folder(text)

    def _set_path_text(self, text):
        """Set the text box without re-triggering on_path_changed."""
        self.path_input.blockSignals(True)
        self.path_input.setText(text)
        self.path_input.blockSignals(False)

    # ── browse button ──────────────────────────────────────────────────────
    # One button, two entry points (this PySide6 build's QFileDialog.Option
    # enum has no DirectoryAndFile flag, so a single files-and-folders
    # dialog is not available). Whatever is picked lands in path_input and
    # on_path_changed auto-detects file vs folder.
    def browse_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.path_input.setText(path)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.path_input.setText(folder)

    # ── folder loading ─────────────────────────────────────────────────────
    def load_folder(self, folder):
        if self.recursive_cb.isChecked():
            files = []
            for root, _, names in os.walk(folder):
                for f in names:
                    if f.lower().endswith(IMAGE_EXTENSIONS):
                        files.append(os.path.join(root, f))
        else:
            files = [
                os.path.join(folder, f) for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f))
                and f.lower().endswith(IMAGE_EXTENSIONS)
            ]
        self.input_type = 'folder'
        self.selected_paths = sorted(files)
        self._size_cache.clear()
        self.refresh_list()
        self.input_changed.emit()


    def clear(self):
        self.input_type = None
        self.selected_paths = []
        self._size_cache.clear()
        self._set_path_text("")
        self.refresh_list()
        self.input_changed.emit()

    # ── list display ───────────────────────────────────────────────────────
    def refresh_list(self):
        self.file_list.clear()
        paths = self.selected_paths

        # Apply resolution filter if active
        if self._filter_enabled:
            max_res = self._filter_max_res
            paths = self.filter_by_resolution(paths, max_res)
            self.status_label.setText(
                f"{len(paths)}/{len(self.selected_paths)} files "
                f"(under {max_res} px)"
            )
        else:
            count = len(paths)
            if count == 1:
                self.status_label.setText(f"1 file selected: {os.path.basename(paths[0])}")
            else:
                self.status_label.setText(f"{count} files selected")

        for p in paths:
            try:
                w, h = self._get_size(p)
                res_str = f"{w}×{h}"
            except Exception:
                res_str = "?"

            # Build a row widget: filename left, resolution right
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(0)

            name_label = QLabel(os.path.basename(p))
            name_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

            res_label = QLabel(f"[{res_str}]")
            res_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            res_label.setStyleSheet("color: #555555;")

            row_layout.addWidget(name_label)
            row_layout.addStretch()
            row_layout.addWidget(res_label)

            item = QListWidgetItem(self.file_list)
            item.setSizeHint(row_widget.sizeHint())
            self.file_list.addItem(item)
            self.file_list.setItemWidget(item, row_widget)

    def filter_by_resolution(self, paths, max_res):
        out = []
        for p in paths:
            try:
                w, h = self._get_size(p)
                if w < max_res and h < max_res:
                    out.append(p)
            except Exception:
                pass
        return out

    def set_resolution_filter(self, enabled, max_res):
        """Push the tab's resolution-filter state into this widget."""
        self._filter_enabled = bool(enabled)
        self._filter_max_res = int(max_res)
        self.refresh_list()


    def show_context_menu(self, pos):
        item = self.file_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        open_action = menu.addAction("Open in Folder")
        action = menu.exec(self.file_list.mapToGlobal(pos))
        if action == open_action:
            self.open_in_folder(item)

    def open_in_folder(self, item):
        row = self.file_list.row(item)
        paths = self.selected_paths
        if self._filter_enabled:
            paths = self.filter_by_resolution(paths, self._filter_max_res)
        if row < len(paths):
            open_in_folder(paths[row])

    # ── what the worker actually processes ────────────────────────────────
    def get_input_paths(self):
        paths = self.selected_paths
        if self._filter_enabled:
            paths = self.filter_by_resolution(paths, self._filter_max_res)
        return paths

    # ── drag & drop ────────────────────────────────────────────────────────
    def handle_drop(self, event):
        """Drop target for image files and folders.

        Folders are handled recursively, files directly. The list is
        rebuilt from scratch on every drop (no append), so dropping the
        same content twice never duplicates entries.
        """
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if os.path.isdir(path):
                    if self.recursive_cb.isChecked():
                        for root, _, files in os.walk(path):
                            for file in files:
                                if file.lower().endswith(IMAGE_EXTENSIONS):
                                    paths.append(os.path.join(root, file))
                    else:
                        for file in os.listdir(path):
                            full = os.path.join(path, file)
                            if os.path.isfile(full) and file.lower().endswith(IMAGE_EXTENSIONS):
                                paths.append(full)
                elif os.path.isfile(path) and path.lower().endswith(IMAGE_EXTENSIONS):
                    paths.append(path)

            if paths:
                self.input_type = 'folder' if len(paths) > 1 else 'file'
                self.selected_paths = sorted(paths)
                self._size_cache.clear()
                # Show the first path (or parent folder) in the text box
                if len(paths) == 1:
                    self._set_path_text(paths[0])
                else:
                    self._set_path_text(os.path.dirname(paths[0]))
                self.refresh_list()
                self.input_changed.emit()
                event.acceptProposedAction()
                return

        event.ignore()

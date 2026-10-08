import os
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                              QFileDialog, QProgressBar, QHBoxLayout,
                              QSpinBox, QLineEdit, QTextEdit, QCheckBox)
from PySide6.QtCore import Qt, QThread, Signal

from modules import image_resizer_engine

class ResizeWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()

    def __init__(self, folder_path, max_resolution, recursive=False):
        super().__init__()
        self.folder_path = folder_path
        self.max_resolution = max_resolution
        self.recursive = recursive
        self.is_running = True

    def run(self):
        image_resizer_engine.run(
            {'folder': self.folder_path, 'max_resolution': self.max_resolution,
             'recursive': self.recursive},
            progress_cb=self._on_engine_progress,
            stop_check=lambda: not self.is_running,
        )
        self.finished.emit()

    def _on_engine_progress(self, current, total, message=""):
        # Engine -> GUI bridge: forward as the same status/progress signals.
        if message:
            self.status.emit(message)
        if total:
            self.progress.emit(int(current / total * 100))

    def stop(self):
        self.is_running = False

class ImageResizerTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Folder selection
        folder_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter folder path...")
        self.path_input.setMinimumWidth(300)
        self.path_input.setMaximumWidth(400)
        self.browse_btn = QPushButton("Browse")
        folder_layout.addWidget(self.path_input)
        folder_layout.addWidget(self.browse_btn)
        self.recursive_cb = QCheckBox("Recursive")
        self.recursive_cb.setToolTip("Include images from subfolders")
        folder_layout.addWidget(self.recursive_cb)
        folder_layout.addStretch()

        # Status label
        self.status_label = QLabel("No folder selected")

        # Max resolution input
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Maximum resolution:"))
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(100, 10000)
        self.resolution_spin.setValue(1024)
        resolution_layout.addWidget(self.resolution_spin)
        resolution_layout.addWidget(QLabel("pixels"))
        resolution_layout.addStretch()

        # Description label
        description = QLabel(
            "Images with either width or height exceeding the maximum resolution "
            "will be resized proportionally to fit within the limit."
        )
        description.setWordWrap(True)
        description.setStyleSheet("color: gray;")

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Resizing")
        self.stop_btn = QPushButton("Stop")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)

        # Progress bar
        self.progress_bar = QProgressBar()

        # Status text area. QTextEdit.append() is O(line) - rebuilding one
        # growing QLabel string per file was O(n^2) for large folders.
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(200)

        # Add widgets to layout
        layout.addLayout(folder_layout)
        layout.addWidget(self.status_label)
        layout.addLayout(resolution_layout)
        layout.addWidget(description)
        layout.addLayout(button_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_text)

        # Connect signals
        self.browse_btn.clicked.connect(self.browse_folder)
        self.path_input.textChanged.connect(self.on_path_changed)
        self.start_btn.clicked.connect(self.start_resize)
        self.stop_btn.clicked.connect(self.stop_resize)

        self.setLayout(layout)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.path_input.setText(folder_path)

    def on_path_changed(self, path):
        path = path.strip()
        if os.path.exists(path) and os.path.isdir(path):
            self.status_label.setText(f"Selected folder: {path}")
            self.start_btn.setEnabled(True)
        else:
            self.status_label.setText("Invalid folder path")
            self.start_btn.setEnabled(False)

    def start_resize(self):
        if self.worker is not None and self.worker.isRunning():
            return

        folder_path = self.path_input.text().strip()
        max_resolution = self.resolution_spin.value()

        self.worker = ResizeWorker(folder_path, max_resolution,
                                   self.recursive_cb.isChecked())
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.resize_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)
        self.path_input.setEnabled(False)
        self.recursive_cb.setEnabled(False)
        self.status_text.clear()
        
        self.worker.start()

    def stop_resize(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            # Non-blocking: the worker's loop notices the flag and emits
            # finished, which runs resize_finished() on the GUI thread.

    def update_status(self, text):
        self.status_text.append(text)

    def resize_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self.path_input.setEnabled(True)
        self.recursive_cb.setEnabled(True)
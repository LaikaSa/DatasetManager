import os
from PIL import Image
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                              QFileDialog, QProgressBar, QHBoxLayout, 
                              QSpinBox, QLineEdit, QScrollArea)
from PySide6.QtCore import Qt, QThread, Signal

class ResizeWorker(QThread):
    progress = Signal(int)
    status = Signal(str)
    finished = Signal()

    def __init__(self, folder_path, max_resolution):
        super().__init__()
        self.folder_path = folder_path
        self.max_resolution = max_resolution
        self.is_running = True

    def run(self):
        image_files = []
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    image_files.append(os.path.join(root, file))

        total_files = len(image_files)
        processed = 0
        resized = 0

        for idx, img_path in enumerate(image_files):
            if not self.is_running:
                break

            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    needs_resize = width > self.max_resolution or height > self.max_resolution

                    if needs_resize:
                        # Calculate new dimensions
                        ratio = min(self.max_resolution / width, self.max_resolution / height)
                        new_width = int(width * ratio)
                        new_height = int(height * ratio)

                        # Create resized subfolder if it doesn't exist
                        output_dir = os.path.join(os.path.dirname(img_path), 'resized')
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Resize and save
                        resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                        output_path = os.path.join(output_dir, os.path.basename(img_path))
                        resized_img.save(output_path, quality=95)
                        resized += 1
                        
                        self.status.emit(
                            f"Resized: {os.path.basename(img_path)}\n"
                            f"Original: {width}x{height} → New: {new_width}x{new_height}"
                        )
                    else:
                        self.status.emit(f"Skipped: {os.path.basename(img_path)} ({width}x{height})")

                processed += 1
                self.progress.emit(int(processed / total_files * 100))

            except Exception as e:
                self.status.emit(f"Error processing {img_path}: {str(e)}")

        self.status.emit(f"\nCompleted: {processed} images processed, {resized} images resized")
        self.finished.emit()

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
        self.folder_path_input = QLineEdit()
        self.folder_path_input.setPlaceholderText("Enter or paste folder path here")
        self.browse_btn = QPushButton("Browse")
        folder_layout.addWidget(self.folder_path_input)
        folder_layout.addWidget(self.browse_btn)

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

        # Status text area
        self.status_area = QScrollArea()
        self.status_area.setWidgetResizable(True)
        self.status_text = QLabel()
        self.status_text.setAlignment(Qt.AlignTop)
        self.status_text.setWordWrap(True)
        self.status_area.setWidget(self.status_text)
        self.status_area.setMinimumHeight(200)

        # Add widgets to layout
        layout.addLayout(folder_layout)
        layout.addWidget(self.status_label)
        layout.addLayout(resolution_layout)
        layout.addWidget(description)
        layout.addLayout(button_layout)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_area)

        # Connect signals
        self.browse_btn.clicked.connect(self.browse_folder)
        self.folder_path_input.textChanged.connect(self.on_path_changed)
        self.start_btn.clicked.connect(self.start_resize)
        self.stop_btn.clicked.connect(self.stop_resize)

        self.setLayout(layout)

    def browse_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.folder_path_input.setText(folder_path)

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

        folder_path = self.folder_path_input.text().strip()
        max_resolution = self.resolution_spin.value()

        self.worker = ResizeWorker(folder_path, max_resolution)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.resize_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)
        self.folder_path_input.setEnabled(False)
        self.status_text.setText("")
        
        self.worker.start()

    def stop_resize(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.resize_finished()

    def update_status(self, text):
        current_text = self.status_text.text()
        self.status_text.setText(current_text + "\n" + text if current_text else text)

    def resize_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)
        self.folder_path_input.setEnabled(True)
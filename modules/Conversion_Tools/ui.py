from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                             QPushButton, QCheckBox, QComboBox, QLabel, 
                             QFileDialog, QTabWidget)
from PySide6.QtCore import Qt, QThread, Signal
from .converter import ImageConverter
from .extension_manager import ExtensionManagerTab

class ConversionWorker(QThread):
    finished = Signal()
    error = Signal(str)
    
    def __init__(self, converter, folder_path, target_format, recursive, use_parallel):
        super().__init__()
        self.converter = converter
        self.folder_path = folder_path
        self.target_format = target_format
        self.recursive = recursive
        self.use_parallel = use_parallel
        self.is_running = True

    def run(self):
        try:
            self.converter.convert_folder(
                self.folder_path, 
                self.target_format, 
                self.recursive,
                self.use_parallel,
                lambda: not self.is_running
            )
            if self.is_running:
                self.finished.emit()
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False

class ConversionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.converter = ImageConverter()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Top row layout
        top_layout = QHBoxLayout()
        
        # Folder input (with reduced width) and browse
        folder_layout = QHBoxLayout()
        self.folder_input = QLineEdit()
        self.folder_input.setPlaceholderText("Enter folder path...")
        self.folder_input.setMinimumWidth(300)  # Set a smaller minimum width
        self.folder_input.setMaximumWidth(400)  # Set a maximum width
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_folder)
        
        # Checkboxes
        self.recursive_cb = QCheckBox("Recursive")
        self.parallel_cb = QCheckBox("Parallel Processing")
        
        # Add all to top layout
        top_layout.addWidget(self.folder_input)
        top_layout.addWidget(self.browse_btn)
        top_layout.addWidget(self.recursive_cb)
        top_layout.addWidget(self.parallel_cb)
        top_layout.addStretch()  # This will push everything to the left

        # Create tab widget for sub-functions
        self.function_tabs = QTabWidget()
        
        # Create conversion tab
        self.conversion_widget = QWidget()
        conversion_layout = QVBoxLayout()
        
        # Format selection
        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Convert to:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(['PNG', 'JPEG', 'BMP', 'WEBP'])
        format_layout.addWidget(self.format_combo)
        format_layout.addStretch()

        # Status label
        self.status_label = QLabel("")

        # Buttons layout
        button_layout = QHBoxLayout()
        self.convert_btn = QPushButton("Start Conversion")
        self.convert_btn.clicked.connect(self.start_conversion)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_conversion)
        self.stop_btn.setVisible(False)
        button_layout.addWidget(self.convert_btn)
        button_layout.addWidget(self.stop_btn)

        # Add elements to conversion layout
        conversion_layout.addLayout(format_layout)
        conversion_layout.addWidget(self.status_label)
        conversion_layout.addLayout(button_layout)
        conversion_layout.addStretch()
        self.conversion_widget.setLayout(conversion_layout)

        # Create extension manager tab
        self.extension_manager = ExtensionManagerTab(
            folder_input=self.folder_input,
            parallel_cb=self.parallel_cb,
            recursive_cb=self.recursive_cb  # Pass the recursive checkbox
        )

        # Add tabs
        self.function_tabs.addTab(self.conversion_widget, "Format Conversion")
        self.function_tabs.addTab(self.extension_manager, "Extension Manager")

        # Add to main layout
        layout.addLayout(top_layout)
        layout.addWidget(self.function_tabs)

        self.setLayout(layout)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.folder_input.setText(folder)

    def start_conversion(self):
        folder_path = self.folder_input.text()
        if not folder_path:
            self.status_label.setText("Please select a folder")
            return

        target_format = self.format_combo.currentText().lower()
        recursive = self.recursive_cb.isChecked()
        use_parallel = self.parallel_cb.isChecked()  # Get parallel processing state

        # Disable inputs during conversion
        self.set_inputs_enabled(False)
        
        # Show stop button
        self.stop_btn.setVisible(True)
        self.status_label.setText("Converting...")

        # Create and start worker thread with parallel processing option
        self.worker = ConversionWorker(
            self.converter, 
            folder_path, 
            target_format, 
            recursive,
            use_parallel  # Pass parallel processing flag to worker
        )
        self.worker.finished.connect(self.conversion_finished)
        self.worker.error.connect(self.conversion_error)
        self.worker.start()

    def stop_conversion(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Stopping...")
            self.stop_btn.setEnabled(False)

    def conversion_finished(self):
        self.status_label.setText("Conversion completed!")
        self.cleanup_after_conversion()

    def conversion_error(self, error_message):
        self.status_label.setText(f"Error: {error_message}")
        self.cleanup_after_conversion()

    def cleanup_after_conversion(self):
        self.set_inputs_enabled(True)
        self.stop_btn.setVisible(False)
        self.worker = None

    def set_inputs_enabled(self, enabled):
        self.folder_input.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)
        self.recursive_cb.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.convert_btn.setEnabled(enabled)
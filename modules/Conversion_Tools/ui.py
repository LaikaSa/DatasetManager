from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
                             QPushButton, QCheckBox, QComboBox, QLabel, 
                             QFileDialog, QTabWidget)
from PySide6.QtCore import Qt, QThread, Signal
from .. import settings as app_settings
from .converter import ImageConverter, IccProfileFixer
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

class IccFixWorker(QThread):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, fixer, folder_path, recursive, backup, dry_run):
        super().__init__()
        self.fixer = fixer
        self.folder_path = folder_path
        self.recursive = recursive
        self.backup = backup
        self.dry_run = dry_run
        self.is_running = True

    def run(self):
        try:
            counts = self.fixer.fix_folder(
                self.folder_path,
                self.recursive,
                self.backup,
                self.dry_run,
                lambda: not self.is_running
            )
            if self.is_running:
                self.finished.emit(counts)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False

class ConversionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.converter = ImageConverter()
        self.icc_fixer = IccProfileFixer()
        self.worker = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Top row layout
        top_layout = QHBoxLayout()
        
        # Folder input (with reduced width) and browse
        folder_layout = QHBoxLayout()
        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter folder path...")
        self.path_input.setMinimumWidth(300)  # Set a smaller minimum width
        self.path_input.setMaximumWidth(400)  # Set a maximum width
        self.browse_btn = QPushButton("Browse")
        self.browse_btn.clicked.connect(self.browse_folder)
        
        # Checkboxes (parallel processing is the app-wide cog setting now)
        self.recursive_cb = QCheckBox("Recursive")
        
        # Add all to top layout
        top_layout.addWidget(self.path_input)
        top_layout.addWidget(self.browse_btn)
        top_layout.addWidget(self.recursive_cb)
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
        self.start_btn = QPushButton("Start Conversion")
        self.start_btn.clicked.connect(self.start_conversion)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_conversion)
        self.stop_btn.setVisible(False)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)

        # ICC profile fixing
        icc_layout = QHBoxLayout()
        self.icc_btn = QPushButton("Fix ICC Profiles")
        self.icc_btn.setToolTip(
            "Strip embedded ICC profiles from all images in the folder "
            "(recursively if Recursive is checked) so tools like kohya / "
            "LoRA training / Qt imageio stop warning about incorrect sRGB profiles. "
            "Non-RGB images are normalized to RGB."
        )
        self.icc_btn.clicked.connect(self.start_icc_fix)
        self.icc_dry_run_cb = QCheckBox("Dry run")
        self.icc_dry_run_cb.setToolTip(
            "Only scan and report how many files would be changed, "
            "without modifying any files."
        )
        self.icc_backup_cb = QCheckBox("Backups (.bak)")
        self.icc_backup_cb.setToolTip(
            "Save a copy of each modified file as <filename>.ext.bak "
            "before overwriting it. Delete the .bak files once you've "
            "verified the results."
        )
        self.icc_backup_cb.setChecked(True)
        icc_layout.addWidget(self.icc_btn)
        icc_layout.addWidget(self.icc_dry_run_cb)
        icc_layout.addWidget(self.icc_backup_cb)
        icc_layout.addStretch()

        # Add elements to conversion layout
        conversion_layout.addLayout(format_layout)
        conversion_layout.addWidget(self.status_label)
        conversion_layout.addLayout(button_layout)
        conversion_layout.addLayout(icc_layout)
        conversion_layout.addStretch()
        self.conversion_widget.setLayout(conversion_layout)

        # Create extension manager tab
        self.extension_manager = ExtensionManagerTab(
            path_input=self.path_input,
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
            self.path_input.setText(folder)

    def start_conversion(self):
        folder_path = self.path_input.text()
        if not folder_path:
            self.status_label.setText("Please select a folder")
            return

        target_format = self.format_combo.currentText().lower()
        recursive = self.recursive_cb.isChecked()
        use_parallel = app_settings.is_parallel_enabled()  # App-wide cog setting

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

    def start_icc_fix(self):
        folder_path = self.path_input.text()
        if not folder_path:
            self.status_label.setText("Please select a folder")
            return

        recursive = self.recursive_cb.isChecked()
        dry_run = self.icc_dry_run_cb.isChecked()
        backup = self.icc_backup_cb.isChecked()

        # Disable inputs during the run
        self.set_inputs_enabled(False)

        # Show stop button
        self.stop_btn.setVisible(True)
        self.status_label.setText(
            "Scanning for ICC profiles..." if dry_run else "Fixing ICC profiles..."
        )

        self.worker = IccFixWorker(
            self.icc_fixer,
            folder_path,
            recursive,
            backup,
            dry_run
        )
        self.worker.finished.connect(self.icc_fix_finished)
        self.worker.error.connect(self.conversion_error)
        self.worker.start()

    def icc_fix_finished(self, counts):
        verb = "Would fix" if self.icc_dry_run_cb.isChecked() else "Fixed"
        self.status_label.setText(
            f"{verb}: {counts['fixed']}  |  Skipped: {counts['skipped']}  |  Errors: {counts['errors']}"
        )
        self.cleanup_after_conversion()

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
        self.path_input.setEnabled(enabled)
        self.browse_btn.setEnabled(enabled)
        self.recursive_cb.setEnabled(enabled)
        self.format_combo.setEnabled(enabled)
        self.icc_btn.setEnabled(enabled)
        self.icc_dry_run_cb.setEnabled(enabled)
        self.icc_backup_cb.setEnabled(enabled)
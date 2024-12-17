from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QListWidget, QCheckBox, QMessageBox)
from PySide6.QtCore import Qt
import os
import send2trash
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from ..logger import setup_logger

logger = setup_logger()

class ExtensionManagerTab(QWidget):
    def __init__(self, folder_input, parallel_cb, recursive_cb):
        super().__init__()
        self.folder_input = folder_input
        self.parallel_cb = parallel_cb
        self.recursive_cb = recursive_cb  # Use the shared recursive checkbox
        self.extension_files = defaultdict(list)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Action buttons at the top
        action_buttons_layout = QHBoxLayout()
        self.scan_btn = QPushButton("Scan Folder")
        self.remove_btn = QPushButton("Remove Selected")
        self.select_all_btn = QPushButton("Select All")
        self.deselect_all_btn = QPushButton("Deselect All")
        
        self.remove_btn.setEnabled(False)
        self.scan_btn.clicked.connect(self.scan_folder)
        self.remove_btn.clicked.connect(self.remove_selected)
        self.select_all_btn.clicked.connect(self.select_all_extensions)
        self.deselect_all_btn.clicked.connect(self.deselect_all_extensions)
        
        action_buttons_layout.addWidget(self.scan_btn)
        action_buttons_layout.addWidget(self.remove_btn)
        action_buttons_layout.addWidget(self.select_all_btn)
        action_buttons_layout.addWidget(self.deselect_all_btn)
        action_buttons_layout.addStretch()

        # Extensions list area
        self.extensions_label = QLabel("Select extensions to remove:")
        self.extensions_list = QListWidget()
        self.extensions_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)

        # Stats label
        self.stats_label = QLabel("")

        # Add all to main layout
        layout.addLayout(action_buttons_layout)
        layout.addWidget(self.extensions_label)
        layout.addWidget(self.extensions_list)
        layout.addWidget(self.stats_label)

        self.setLayout(layout)

    def scan_folder(self):
        folder_path = self.folder_input.text()
        folder_path = os.path.normpath(folder_path)
        
        if not folder_path or not os.path.exists(folder_path):
            QMessageBox.warning(self, "Error", "Please select a valid folder first")
            return

        self.extension_files.clear()
        self.extensions_list.clear()
        files_to_process = []

        try:
            # Use the shared recursive checkbox
            if self.recursive_cb.isChecked():
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        files_to_process.append((root, file))
            else:
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        files_to_process.append((folder_path, file))

            # Update UI with results
            for ext, files in sorted(self.extension_files.items()):
                item_text = f"{ext} ({len(files)} files)"
                self.extensions_list.addItem(item_text)

            self.stats_label.setText(f"Total files: {total_files}")
            self.remove_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error scanning folder: {str(e)}")

    def _process_file(self, root, file):
        try:
            # Use os.path.normpath to normalize the path
            file_path = os.path.normpath(os.path.join(root, file))
            _, ext = os.path.splitext(file)
            ext = ext.lower() if ext else "(no extension)"
            self.extension_files[ext].append(file_path)
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")

    def select_all_extensions(self):
        for i in range(self.extensions_list.count()):
            self.extensions_list.item(i).setSelected(True)

    def deselect_all_extensions(self):
        for i in range(self.extensions_list.count()):
            self.extensions_list.item(i).setSelected(False)

    def _remove_single_file(self, file_path):
        """Helper method to remove a single file."""
        try:
            if os.path.exists(file_path):
                send2trash.send2trash(file_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Error removing {file_path}: {str(e)}")
            return False

    def remove_selected(self):
        if not self.extension_files:
            return

        selected_items = self.extensions_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Info", "Please select extensions to remove")
            return

        files_to_remove = []
        selected_exts = []
        for item in selected_items:
            ext = item.text().split(' (')[0]
            selected_exts.append(ext)
            files_to_remove.extend(self.extension_files[ext])

        if not files_to_remove:
            QMessageBox.information(self, "Info", "No files to remove")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Removal",
            f"Are you sure you want to move {len(files_to_remove)} files "
            f"with extension(s) {', '.join(selected_exts)} to the recycle bin?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            errors = []
            removed_count = 0

            if self.parallel_cb.isChecked():
                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    future_to_file = {
                        executor.submit(self._remove_single_file, file_path): file_path 
                        for file_path in files_to_remove
                    }
                    
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        try:
                            result = future.result()
                            if result:
                                removed_count += 1
                            else:
                                errors.append(f"Failed to remove: {file_path}")
                        except Exception as e:
                            errors.append(f"Error removing {file_path}: {str(e)}")
            else:
                # Sequential processing
                for file_path in files_to_remove:
                    if self._remove_single_file(file_path):
                        removed_count += 1
                    else:
                        errors.append(f"Failed to remove: {file_path}")

            # Show results
            if errors:
                error_msg = "\n".join(errors[:10])
                if len(errors) > 10:
                    error_msg += f"\n... and {len(errors) - 10} more errors"
                
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"Removed {removed_count} files.\n\nErrors occurred:\n{error_msg}"
                )
            else:
                QMessageBox.information(
                    self,
                    "Success",
                    f"Successfully moved {removed_count} files to recycle bin"
                )

            self.scan_folder()

    def scan_folder(self):
        folder_path = self.folder_input.text()
        folder_path = os.path.normpath(folder_path)
        
        if not folder_path or not os.path.exists(folder_path):
            QMessageBox.warning(self, "Error", "Please select a valid folder first")
            return

        self.extension_files.clear()
        self.extensions_list.clear()
        files_to_process = []

        try:
            # Collect files
            if self.recursive_cb.isChecked():
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        files_to_process.append((root, file))
            else:
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        files_to_process.append((folder_path, file))

            # Process files based on parallel checkbox
            if self.parallel_cb.isChecked():
                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                    executor.map(lambda x: self._process_file(*x), files_to_process)
            else:
                for root, file in files_to_process:
                    self._process_file(root, file)

            # Update UI with results
            for ext, files in sorted(self.extension_files.items()):
                item_text = f"{ext} ({len(files)} files)"
                self.extensions_list.addItem(item_text)

            total_files = sum(len(files) for files in self.extension_files.values())
            self.stats_label.setText(f"Total files: {total_files}")
            self.remove_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error scanning folder: {str(e)}")
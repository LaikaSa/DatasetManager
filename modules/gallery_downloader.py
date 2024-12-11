import os
import re
import json
import time
import subprocess
import requests
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QTabWidget,
                              QFileDialog, QMessageBox, QCheckBox, QTableWidgetItem,
                              QFormLayout, QHBoxLayout, QGroupBox, QTableWidget, 
                              QLineEdit, QComboBox, QTextEdit, QDialog, 
                              QTabWidget, QDialogButtonBox)
from PySide6.QtCore import Qt, QThread, Signal
from modules.logger import setup_logger
from modules.settings_manager import SettingsManager
logger = setup_logger()


class SankakuAPI:
    def __init__(self, username=None, password=None):
        self.base_url = "https://capi-v2.sankakucomplex.com"
        self.headers = {
            "Accept": "application/vnd.sankaku.api+json;v=2",
            "Platform": "web-app",
            "Origin": "https://sankaku.app",
            "Api-Version": "2"
        }
        self.username = username
        self.password = password
        self.auth_token = None

    def authenticate(self):
        if not self.auth_token and self.username and self.password:
            url = f"{self.base_url}/auth/token"
            data = {
                "login": self.username,
                "password": self.password
            }
            response = requests.post(url, json=data, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    self.auth_token = f"Bearer {data['access_token']}"
                    self.headers["Authorization"] = self.auth_token

    def get_posts(self, tags, page=1, limit=40):
        self.authenticate()
        url = f"{self.base_url}/posts/keyset"
        params = {
            "tags": tags,
            "limit": str(limit),
            "lang": "en"
        }
        
        response = requests.get(url, params=params, headers=self.headers)
        if response.status_code == 429:
            # Handle rate limiting
            reset_time = response.headers.get("X-RateLimit-Reset")
            if reset_time:
                time.sleep(int(reset_time) - time.time())
                return self.get_posts(tags, page, limit)
        
        return response.json()

class DownloadWorker(QThread):
    status = Signal(str)    # Add signal definitions
    finished = Signal()

    def __init__(self, url, save_path, site, username=None, password=None):
        super().__init__()
        root_dir = os.path.dirname(os.path.dirname(__file__))
        self.python_path = os.path.join(root_dir, "python", "python.exe")
        self.url = url
        self.save_path = save_path
        self.site = site
        self.username = username
        self.password = password
        self.is_running = True
        self.python_path = os.path.join(os.path.dirname(__file__), "python", "python.exe")
        self.load_settings()

    def load_settings(self):
        settings_file = os.path.join(os.path.dirname(__file__), 'download_settings.json')
        try:
            with open(settings_file, 'r') as f:
                self.settings = json.load(f)
        except Exception:
            self.settings = {'file_types': {'jpg': True, 'png': True}}

    def run(self):
        try:
            command = [
                self.python_path,
                "-m",
                "gallery_dl",
                '--verbose',
                '--destination', self.save_path
            ]

            # Add file type filter - modified this part
            allowed_types = [ext for ext, enabled in self.settings['file_types'].items() if enabled]
            if allowed_types:
                extensions = ", ".join(f"'.{ext}'" for ext in allowed_types)
                command.extend([
                    '--filter', 
                    f"extension in ({extensions})"
                ])

            # Add authentication if provided
            if self.username and self.password:
                command.extend(['--username', self.username])
                command.extend(['--password', self.password])

            command.append(self.url)

            self.status.emit(f"Starting download from {self.site}")
            self.status.emit(f"Allowed file types: {', '.join(allowed_types)}")
            self.status.emit(f"Command: {' '.join(command)}")  # Debug line
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            while self.is_running:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.status.emit(output.strip())

            if not self.is_running:
                process.terminate()
                self.status.emit("Download stopped by user")

        except Exception as e:
            self.status.emit(f"Error: {str(e)}")

        self.finished.emit()

    def extract_tags_from_url(self, url):
        match = re.search(r"tags=([^&#]+)", url)
        if match:
            return match.group(1).replace("+", " ")
        return ""

    def download_file(self, url, path):
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

class AuthManager:
    def __init__(self):
        # Change the path to be in the settings folder
        self.settings_dir = os.path.join(os.path.dirname(__file__), 'settings')
        self.auth_file = os.path.join(self.settings_dir, 'credentials.json')
        self._ensure_settings_dir()

    def _ensure_settings_dir(self):
        # Create settings directory if it doesn't exist
        if not os.path.exists(self.settings_dir):
            os.makedirs(self.settings_dir)

    def save_credentials(self, site, username, password):
        credentials = self.load_all_credentials()
        credentials[site] = {
            'username': username,
            'password': password
        }
        with open(self.auth_file, 'w') as f:
            json.dump(credentials, f, indent=4)

    def load_credentials(self, site):
        credentials = self.load_all_credentials()
        return credentials.get(site, {})

    def load_all_credentials(self):
        if not os.path.exists(self.auth_file):
            return {}
        try:
            with open(self.auth_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return {}

    def delete_credentials(self, site):
        credentials = self.load_all_credentials()
        if site in credentials:
            del credentials[site]
            with open(self.auth_file, 'w') as f:
                json.dump(credentials, f, indent=4)

class AuthDialog(QDialog):
    def __init__(self, parent=None, auth_manager=None):
        super().__init__(parent)
        self.auth_manager = auth_manager
        self.settings_manager = parent.settings_manager  # Get settings_manager from parent
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Authentication Manager")
        self.setMinimumWidth(400)
        layout = QVBoxLayout()

        # Create table for existing credentials
        self.cred_table = QTableWidget()
        self.cred_table.setColumnCount(3)
        self.cred_table.setHorizontalHeaderLabels(["Site", "Username", "Actions"])
        self.cred_table.horizontalHeader().setStretchLastSection(True)
        
        # Add new credential form
        form_layout = QFormLayout()
        self.site_combo = QComboBox()
        
        # Get sites from settings_manager instead of hardcoded list
        sites = self.settings_manager.get_sites_settings().keys()
        self.site_combo.addItems(sites)
        
        self.username_input = QLineEdit()
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)
        
        form_layout.addRow("Site:", self.site_combo)
        form_layout.addRow("Username:", self.username_input)
        form_layout.addRow("Password:", self.password_input)

        save_btn = QPushButton("Save Credentials")
        save_btn.clicked.connect(self.save_credentials)

        layout.addWidget(self.cred_table)
        layout.addLayout(form_layout)
        layout.addWidget(save_btn)

        self.setLayout(layout)
        self.load_existing_credentials()

    def load_existing_credentials(self):
        credentials = self.auth_manager.load_all_credentials()
        self.cred_table.setRowCount(len(credentials))
        
        for row, (site, cred) in enumerate(credentials.items()):
            self.cred_table.setItem(row, 0, QTableWidgetItem(site))
            self.cred_table.setItem(row, 1, QTableWidgetItem(cred['username']))
            
            delete_btn = QPushButton("Delete")
            delete_btn.clicked.connect(lambda s, site=site: self.delete_credentials(site))
            self.cred_table.setCellWidget(row, 2, delete_btn)

    def save_credentials(self):
        site = self.site_combo.currentText()
        username = self.username_input.text()
        password = self.password_input.text()

        if not username or not password:
            QMessageBox.warning(self, "Error", "Please fill in all fields")
            return

        self.auth_manager.save_credentials(site, username, password)
        self.username_input.clear()
        self.password_input.clear()
        self.load_existing_credentials()

    def delete_credentials(self, site):
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete credentials for {site}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.auth_manager.delete_credentials(site)
            self.load_existing_credentials()

class GalleryDownloaderTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.auth_manager = AuthManager()
        self.settings_manager = SettingsManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Settings buttons row
        settings_row = QHBoxLayout()
        self.settings_btn = QPushButton("Download Settings")
        manage_sites_btn = QPushButton("Manage Sites")
        settings_row.addWidget(self.settings_btn)
        settings_row.addWidget(manage_sites_btn)
        settings_row.addStretch()
        layout.addLayout(settings_row)

        # URL input section
        url_group = QGroupBox("URL")
        url_layout = QVBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Enter URL (e.g., https://safebooru.org/index.php?page=post&s=list&tags=...)")
        url_layout.addWidget(self.url_input)
        url_group.setLayout(url_layout)

        # Save location section
        save_group = QGroupBox("Save Location")
        save_layout = QHBoxLayout()
        self.save_path_input = QLineEdit()
        self.save_path_input.setPlaceholderText("Select save location")
        self.browse_btn = QPushButton("Browse")
        save_layout.addWidget(self.save_path_input)
        save_layout.addWidget(self.browse_btn)
        save_group.setLayout(save_layout)

        # Control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Download")
        self.stop_btn = QPushButton("Stop")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)

        # Status area
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMinimumHeight(200)

        # Add all to main layout
        layout.addWidget(url_group)
        layout.addWidget(save_group)
        layout.addLayout(button_layout)
        layout.addWidget(self.status_text)

        # Connect signals
        self.browse_btn.clicked.connect(self.browse_save_location)
        self.start_btn.clicked.connect(self.start_download)
        self.stop_btn.clicked.connect(self.stop_download)
        self.settings_btn.clicked.connect(self.show_settings)
        manage_sites_btn.clicked.connect(self.show_site_manager)
        self.url_input.textChanged.connect(self.update_start_button)
        self.save_path_input.textChanged.connect(self.update_start_button)

        self.setLayout(layout)

    def load_sites(self):
        if os.path.exists(self.sites_file):
            try:
                with open(self.sites_file, 'r') as f:
                    self.SITES_REQUIRING_AUTH = json.load(f)
            except Exception as e:
                logger.error(f"Error loading sites: {e}")
                # Use default sites if loading fails
                self.SITES_REQUIRING_AUTH = {
                    "Auto Detect": False,
                    "Danbooru": True,
                    "Gelbooru": False,
                    "Safebooru": False,
                    "Konachan": False,
                    "Yandere": False,
                    "Sankaku": True,
                    "Rule34": False,
                    "E621": True
                }

    def show_site_manager(self):
        dialog = SiteManagerDialog(self.settings_manager, self)
        dialog.exec()

    def show_settings(self):
        dialog = DownloadSettingsDialog(self.settings_manager, self)  # Pass settings_manager as first argument
        dialog.exec()

    def browse_save_location(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Save Location")
        if folder_path:
            self.save_path_input.setText(folder_path)

    def show_auth_dialog(self):
        dialog = AuthDialog(self, self.auth_manager)
        dialog.exec()

    def update_start_button(self):
        self.start_btn.setEnabled(
            bool(self.url_input.text().strip()) and 
            bool(self.save_path_input.text().strip())
        )

    def update_status(self, text):
        self.status_text.append(text)
        # Scroll to bottom
        scrollbar = self.status_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def start_download(self):
        if self.worker is not None and self.worker.isRunning():
            return

        url = self.url_input.text().strip()
        save_path = self.save_path_input.text().strip()

        self.worker = DownloadWorker(
            url, 
            save_path,
            "Auto Detect",  # Always use Auto Detect
            None,  # No username
            None   # No password
        )
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.download_finished)

        # Disable controls
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.url_input.setEnabled(False)
        self.save_path_input.setEnabled(False)
        self.browse_btn.setEnabled(False)

        self.status_text.clear()
        self.worker.start()

    def stop_download(self):
        if self.worker and self.worker.isRunning():
            self.worker.is_running = False
            self.worker.wait()
            self.download_finished()
            self.update_status("Download stopped by user")

    def download_finished(self):
        # Re-enable controls
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.url_input.setEnabled(True)
        self.save_path_input.setEnabled(True)
        self.browse_btn.setEnabled(True)
        self.site_combo.setEnabled(True)

class SiteManagerDialog(QDialog):
    def __init__(self, settings_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Sites")
        self.setMinimumWidth(400)
        self.settings_manager = settings_manager
        self.sites = self.settings_manager.get_sites_settings().copy()  # Get a copy of current sites
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Site list
        self.site_list = QTableWidget()
        self.site_list.setColumnCount(3)
        self.site_list.setHorizontalHeaderLabels(["Site Name", "Requires Auth", "Actions"])
        self.site_list.horizontalHeader().setStretchLastSection(True)
        
        # Add new site controls
        new_site_group = QGroupBox("Add New Site")
        form_layout = QHBoxLayout()
        self.site_name_input = QLineEdit()
        self.site_name_input.setPlaceholderText("Enter site name")
        self.auth_required_cb = QCheckBox("Requires Authentication")
        self.add_btn = QPushButton("Add Site")
        
        form_layout.addWidget(self.site_name_input)
        form_layout.addWidget(self.auth_required_cb)
        form_layout.addWidget(self.add_btn)
        new_site_group.setLayout(form_layout)

        # Authentication management button
        auth_btn = QPushButton("Manage Authentication")
        auth_btn.clicked.connect(self.show_auth_dialog)

        # Connect signals
        self.add_btn.clicked.connect(self.add_site)
        self.site_list.itemChanged.connect(self.on_auth_changed)
        
        # Add widgets to layout
        layout.addWidget(self.site_list)
        layout.addWidget(new_site_group)
        layout.addWidget(auth_btn)  # Add auth button below new site group
        
        # Add OK/Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self.refresh_site_list()

    def show_auth_dialog(self):
        dialog = AuthDialog(self, self.parent().auth_manager)
        dialog.exec()

    def refresh_site_list(self):
        self.site_list.setRowCount(len(self.sites))
        for row, (site, requires_auth) in enumerate(self.sites.items()):
            # Site name
            self.site_list.setItem(row, 0, QTableWidgetItem(site))
            
            # Requires auth
            auth_item = QTableWidgetItem()
            auth_item.setCheckState(Qt.Checked if requires_auth else Qt.Unchecked)
            self.site_list.setItem(row, 1, auth_item)
            
            # Delete button (except for Auto Detect)
            if site != "Auto Detect":
                delete_btn = QPushButton("Delete")
                def make_delete_function(site_name):
                    return lambda: self.delete_site(site_name)
                delete_btn.clicked.connect(make_delete_function(site))
                self.site_list.setCellWidget(row, 2, delete_btn)

        self.site_list.resizeColumnsToContents()

    def on_auth_changed(self, item):
        if item.column() == 1:  # Auth requirement column
            site_name = self.site_list.item(item.row(), 0).text()
            self.sites[site_name] = (item.checkState() == Qt.Checked)

    def add_site(self):
        site_name = self.site_name_input.text().strip()
        if not site_name:
            QMessageBox.warning(self, "Error", "Please enter a site name")
            return
        
        if site_name in self.sites:
            QMessageBox.warning(self, "Error", "Site already exists")
            return

        self.sites[site_name] = self.auth_required_cb.isChecked()
        self.refresh_site_list()
        self.site_name_input.clear()
        self.auth_required_cb.setChecked(False)

    def delete_site(self, site_name):
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete {site_name}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            del self.sites[site_name]
            # Save changes immediately after deletion
            self.settings_manager.save_sites_settings(self.sites)
            self.refresh_site_list()

    def accept(self):
        # Save all changes to settings manager before closing
        self.settings_manager.save_sites_settings(self.sites)
        super().accept()

class DownloadSettingsDialog(QDialog):
    def __init__(self, settings_manager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Download Settings")
        self.setMinimumWidth(400)
        self.settings_manager = settings_manager
        self.settings = self.settings_manager.get_download_settings().copy()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        
        # Create tabs for different settings categories
        tabs = QTabWidget()
        
        # File Types tab
        file_types_tab = QWidget()
        file_types_layout = QVBoxLayout()
        
        # Create group box for file types
        file_types_group = QGroupBox("File Types to Download")
        types_layout = QVBoxLayout()
        
        # Add checkboxes for each file type
        self.jpg_cb = QCheckBox("JPG/JPEG")
        self.png_cb = QCheckBox("PNG")
        self.gif_cb = QCheckBox("GIF")
        self.webm_cb = QCheckBox("WEBM")
        self.mp4_cb = QCheckBox("MP4")
        
        # Set checked state from saved settings
        self.jpg_cb.setChecked(self.settings['file_types']['jpg'])
        self.png_cb.setChecked(self.settings['file_types']['png'])
        self.gif_cb.setChecked(self.settings['file_types']['gif'])
        self.webm_cb.setChecked(self.settings['file_types']['webm'])
        self.mp4_cb.setChecked(self.settings['file_types']['mp4'])
        
        # Add checkboxes to layout
        types_layout.addWidget(self.jpg_cb)
        types_layout.addWidget(self.png_cb)
        types_layout.addWidget(self.gif_cb)
        types_layout.addWidget(self.webm_cb)
        types_layout.addWidget(self.mp4_cb)
        
        # Add select/deselect all buttons
        buttons_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        buttons_layout.addWidget(select_all_btn)
        buttons_layout.addWidget(deselect_all_btn)
        
        # Connect buttons
        select_all_btn.clicked.connect(self.select_all_types)
        deselect_all_btn.clicked.connect(self.deselect_all_types)
        
        # Set layouts
        file_types_group.setLayout(types_layout)
        file_types_layout.addWidget(file_types_group)
        file_types_layout.addLayout(buttons_layout)
        file_types_tab.setLayout(file_types_layout)
        
        # Add tabs
        tabs.addTab(file_types_tab, "File Types")
        tabs.addTab(QWidget(), "Size Filters")  # Placeholder for future implementation
        tabs.addTab(QWidget(), "Resolution")    # Placeholder for future implementation
        tabs.addTab(QWidget(), "Rating")        # Placeholder for future implementation

        # Add tabs to main layout
        layout.addWidget(tabs)
        
        # Add OK/Cancel buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)

    def select_all_types(self):
        for cb in [self.jpg_cb, self.png_cb, self.gif_cb, 
                  self.webm_cb, self.mp4_cb]:
            cb.setChecked(True)

    def deselect_all_types(self):
        for cb in [self.jpg_cb, self.png_cb, self.gif_cb, 
                  self.webm_cb, self.mp4_cb]:
            cb.setChecked(False)

    def get_current_settings(self):
        return {
            'file_types': {
                'jpg': self.jpg_cb.isChecked(),
                'png': self.png_cb.isChecked(),
                'gif': self.gif_cb.isChecked(),
                'webm': self.webm_cb.isChecked(),
                'mp4': self.mp4_cb.isChecked()
            }
        }

    def accept(self):
        # Save settings before closing
        settings = self.get_current_settings()
        self.settings_manager.save_download_settings(settings)
        super().accept()
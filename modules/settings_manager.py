import os
import json
from modules.logger import setup_logger

class SettingsManager:
    def __init__(self):
        # Go up one level from modules folder to get to project root
        root_dir = os.path.dirname(os.path.dirname(__file__))
        self.settings_dir = os.path.join(root_dir, 'settings')
        os.makedirs(self.settings_dir, exist_ok=True)
        self.download_settings_file = os.path.join(self.settings_dir, 'download_settings.json')
        self.sites_settings_file = os.path.join(self.settings_dir, 'sites_settings.json')
        self.load_settings()

    def load_settings(self):
        # Default download settings
        self.download_settings = {
            'file_types': {
                'jpg': True,
                'png': True,
                'gif': False,
                'webm': False,
                'mp4': False
            }
        }
        
        # Default sites settings - only use if no saved settings exist
        default_sites = {
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

        # Load saved settings if they exist
        if os.path.exists(self.download_settings_file):
            try:
                with open(self.download_settings_file, 'r') as f:
                    saved_settings = json.load(f)
                    self.download_settings.update(saved_settings)
            except Exception as e:
                logger.error(f"Error loading download settings: {e}")

        # For sites, use defaults only if no saved settings exist
        if os.path.exists(self.sites_settings_file):
            try:
                with open(self.sites_settings_file, 'r') as f:
                    self.sites_settings = json.load(f)
            except Exception as e:
                logger.error(f"Error loading sites settings: {e}")
                self.sites_settings = default_sites.copy()
        else:
            self.sites_settings = default_sites.copy()

    def save_download_settings(self, settings):
        self.download_settings.update(settings)
        try:
            with open(self.download_settings_file, 'w') as f:
                json.dump(self.download_settings, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving download settings: {e}")

    def save_sites_settings(self, settings):
        self.sites_settings = settings.copy()  # Replace entire sites dictionary
        try:
            with open(self.sites_settings_file, 'w') as f:
                json.dump(self.sites_settings, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving sites settings: {e}")

    def get_download_settings(self):
        return self.download_settings

    def get_sites_settings(self):
        return self.sites_settings
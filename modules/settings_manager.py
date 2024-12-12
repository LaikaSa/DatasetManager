import os
import json
from modules.logger import setup_logger

class SettingsManager:
    def __init__(self):
        self.settings_dir = os.path.join(os.path.dirname(__file__), 'settings')
        os.makedirs(self.settings_dir, exist_ok=True)
        self.download_settings_file = os.path.join(self.settings_dir, 'download_settings.json')
        self.sites_settings_file = os.path.join(self.settings_dir, 'sites_settings.json')
        self.general_settings_file = os.path.join(self.settings_dir, 'general_settings.json')
        self.load_settings()

    def load_settings(self):
        # Default download settings with all filters
        self.download_settings = {
            'file_types': {
                'jpg': True,
                'png': True,
                'gif': False,
                'webm': False,
                'mp4': False
            },
            'size_filters': {
                'enabled': False,
                'min_size': 0,
                'min_unit': 'KB',
                'max_size': 0,
                'max_unit': 'MB'
            },
            'resolution_filters': {
                'enabled': False,
                'min_width': 0,
                'max_width': 0,
                'min_height': 0,
                'max_height': 0,
                'aspect_ratio_enabled': False,
                'aspect_width': 16,
                'aspect_height': 9,
                'aspect_tolerance': 10
            },
            'rating_filters': {
                'enabled': False,
                'safe': True,
                'questionable': True,
                'explicit': True,
                'unrated': True
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
                    # Update each filter category separately to maintain structure
                    for category in ['file_types', 'size_filters', 'resolution_filters', 'rating_filters']:
                        if category in saved_settings:
                            self.download_settings[category].update(saved_settings[category])
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

        # Add general settings with last directory
        self.general_settings = {
            'last_download_directory': ''
        }
        
        if os.path.exists(self.general_settings_file):
            try:
                with open(self.general_settings_file, 'r') as f:
                    self.general_settings.update(json.load(f))
            except Exception as e:
                logger.error(f"Error loading general settings: {e}")

    def save_general_settings(self, settings):
        self.general_settings.update(settings)
        try:
            with open(self.general_settings_file, 'w') as f:
                json.dump(self.general_settings, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving general settings: {e}")

    def get_general_settings(self):
        return self.general_settings

    def save_download_settings(self, settings):
        # Update each filter category separately
        for category in ['file_types', 'size_filters', 'resolution_filters', 'rating_filters']:
            if category in settings:
                self.download_settings[category].update(settings[category])
        try:
            with open(self.download_settings_file, 'w') as f:
                json.dump(self.download_settings, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving download settings: {e}")

    def save_sites_settings(self, settings):
        self.sites_settings = settings.copy()
        try:
            with open(self.sites_settings_file, 'w') as f:
                json.dump(self.sites_settings, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving sites settings: {e}")

    def get_download_settings(self):
        return self.download_settings

    def get_sites_settings(self):
        return self.sites_settings
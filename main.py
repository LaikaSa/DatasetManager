import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDragMoveEvent
from PySide6.QtCore import Qt, QSettings
from modules.duplicate_detector import DuplicateDetectorTab
from modules.image_resizer import ImageResizerTab
from modules.upscaler import UpscalerTab
from modules.logger import setup_logger
from modules.caption_generator import CaptionGeneratorTab
from modules.tag_editor import TagEditorTab
from modules.Conversion_Tools import ConversionTab
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

        # Create tabs
        self.duplicate_tab = DuplicateDetectorTab()
        self.resizer_tab = ImageResizerTab()
        self.upscaler_tab = UpscalerTab()
        self.caption_tab = CaptionGeneratorTab()
        self.tag_editor_tab = TagEditorTab()
        self.conversion_tab = ConversionTab()

        # (stable key, display label, widget) in the default order
        self.tab_definitions = [
            ("duplicate", "Duplicate Detection", self.duplicate_tab),
            ("resizer", "Image Resizer", self.resizer_tab),
            ("upscaler", "Upscaler", self.upscaler_tab),
            ("caption", "Caption Generator", self.caption_tab),
            ("tag_editor", "Tags Editor", self.tag_editor_tab),
            ("conversion", "Conversion Tools", self.conversion_tab),
        ]

        self._add_tabs_in_saved_order()

        # Persist the new order whenever the user drags a tab into place
        self.tabs.tabBar().tabMoved.connect(self._save_tab_order)

    def _add_tabs_in_saved_order(self):
        """Add tabs using the order saved from a previous session, if any."""
        tab_map = {key: (label, widget) for key, label, widget in self.tab_definitions}
        saved_order = self.settings.value("tab_order", [])
        if isinstance(saved_order, str):  # QSettings may return a single str for a 1-item list
            saved_order = [saved_order]

        # Keep saved keys that still exist, then append any tabs missing from the saved order
        ordered_keys = [key for key in saved_order if key in tab_map]
        ordered_keys += [key for key, _, _ in self.tab_definitions if key not in ordered_keys]

        self.tab_keys = []  # index -> key, kept in sync with the actual visual tab order
        for key in ordered_keys:
            label, widget = tab_map[key]
            self.tabs.addTab(widget, label)
            self.tab_keys.append(key)

    def _save_tab_order(self, *_args):
        """Recompute tab order from current widget positions and persist it."""
        key_by_widget = {widget: key for key, _, widget in self.tab_definitions}
        self.tab_keys = [key_by_widget[self.tabs.widget(i)] for i in range(self.tabs.count())]
        self.settings.setValue("tab_order", self.tab_keys)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        # Get the current active tab
        current_tab = self.tabs.currentWidget()
        
        # Handle the drop based on the current tab
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
                    current_subtab.update_file_list()
                    current_subtab.parent.check_input(files[0])
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
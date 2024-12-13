import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDragMoveEvent
from PySide6.QtCore import Qt
from modules.duplicate_detector import DuplicateDetectorTab
from modules.image_resizer import ImageResizerTab
from modules.upscaler import UpscalerTab
from modules.logger import setup_logger
from modules.caption_generator import CaptionGeneratorTab
from modules.tag_editor import TagEditorTab
import os  # Add this for path operations
logger = setup_logger()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing main application")
        self.setWindowTitle("Image Processing Tool")
        self.setMinimumSize(1000, 600)
        self.setAcceptDrops(True)  # Enable drop for main window

        # Create tab widget
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # Create tabs
        self.duplicate_tab = DuplicateDetectorTab()
        self.resizer_tab = ImageResizerTab()
        self.upscaler_tab = UpscalerTab()
        self.caption_tab = CaptionGeneratorTab()
        self.tag_editor_tab = TagEditorTab()

        # Add tabs
        self.tabs.addTab(self.duplicate_tab, "Duplicate Detection")
        self.tabs.addTab(self.resizer_tab, "Image Resizer")
        self.tabs.addTab(self.upscaler_tab, "Upscaler")
        self.tabs.addTab(self.caption_tab, "Caption Generator")
        self.tabs.addTab(self.tag_editor_tab, "Tags Editor")
        
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
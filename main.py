import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QTabWidget
from modules.duplicate_detector import DuplicateDetectorTab
from modules.image_resizer import ImageResizerTab
from modules.upscaler import UpscalerTab
from modules.gallery_downloader import GalleryDownloaderTab
from modules.logger import setup_logger

logger = setup_logger()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing main application")
        self.setWindowTitle("Image Processing Tool")
        self.setMinimumSize(800, 600)

        # Create tab widget
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # Create tabs
        duplicate_tab = DuplicateDetectorTab()
        resizer_tab = ImageResizerTab()
        upscaler_tab = UpscalerTab()
        downloader_tab = GalleryDownloaderTab()  # Add this line

        # Add tabs
        tabs.addTab(duplicate_tab, "Duplicate Detection")
        tabs.addTab(resizer_tab, "Image Resizer")
        tabs.addTab(upscaler_tab, "Upscaler")
        tabs.addTab(downloader_tab, "Gallery Downloader")

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
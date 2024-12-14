from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QPixmap

class ImageThumbnail(QLabel):
    clicked = Signal(str)

    def __init__(self, image_path, tags):
        super().__init__()
        self.image_path = image_path
        # Pre-process tags into a set for faster comparison
        self.tags = {tag.strip().lower() for tag in tags.split(',') if tag.strip()}
        self.setFixedSize(150, 150)
        
        # Load and scale image
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(
            QSize(150, 150),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(scaled_pixmap)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def has_all_tags(self, required_tags):
        """Efficiently check if thumbnail has all required tags"""
        return required_tags.issubset(self.tags)
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QPixmap, QImage
import os

class ImageThumbnail(QLabel):
    clicked = Signal(str)

    def __init__(self, image_path, tags):
        super().__init__()
        self.image_path = image_path
        self.tags = set(tags.lower().split(','))
        self.setFixedSize(150, 150)
        
        # Load image as thumbnail size directly
        self.load_thumbnail()
        
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet("""
            QLabel {
                border: 2px solid transparent;
                background-color: #2d2d2d;
                padding: 3px;
            }
            QLabel:hover {
                border: 2px solid #007bff;
            }
        """)

    def load_thumbnail(self):
        # Load image at thumbnail size directly
        image = QImage(self.image_path)
        scaled_image = image.scaled(
            150, 150,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.setPixmap(QPixmap.fromImage(scaled_image))
        # Clear image to free memory
        image = None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.image_path)

    def has_all_tags(self, required_tags):
        return required_tags.issubset(self.tags)

    def has_any_tags(self, required_tags):
        return bool(required_tags & self.tags)
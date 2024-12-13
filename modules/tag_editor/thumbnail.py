from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QPixmap

class ImageThumbnail(QLabel):
    clicked = Signal(str)

    def __init__(self, image_path, tags):
        super().__init__()
        self.image_path = image_path
        self.tags = set(tags.lower().split(','))
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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.image_path)
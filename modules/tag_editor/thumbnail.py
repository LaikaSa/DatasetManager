from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Signal, Qt, QSize
from PySide6.QtGui import QPixmap

class ImageThumbnail(QLabel):
    clicked = Signal(str)

    def __init__(self, image_path, tags):
        super().__init__()
        self.image_path = image_path
        # Clean and normalize tags when creating the set
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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.image_path)

    def has_all_tags(self, required_tags):
        """Check if thumbnail has ALL required tags (AND logic)"""
        result = required_tags.issubset(self.tags)
        print(f"Checking {self.image_path} for ALL tags {required_tags}: {result}")
        print(f"Image tags: {self.tags}")
        return result
    
    def has_any_tags(self, required_tags):
        """Check if thumbnail has ANY of the required tags (OR logic)"""
        result = bool(required_tags & self.tags)
        print(f"Checking {self.image_path} for ANY tags {required_tags}: {result}")
        print(f"Image tags: {self.tags}")
        return result
from PySide6.QtWidgets import (QScrollArea, QWidget, QGridLayout, QStackedLayout, 
                              QPushButton, QVBoxLayout, QLabel, QHBoxLayout)
from PySide6.QtCore import Qt, Signal
from .data_model import ImageData

class ThumbnailWidget(QLabel):
    clicked = Signal(str)

    def __init__(self, image_data: ImageData):
        super().__init__()
        self.image_path = image_data.path
        self.setFixedSize(150, 150)
        self.setPixmap(image_data.thumbnail)
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

class FullImageView(QWidget):
    closed = Signal()

    def __init__(self):
        super().__init__()
        self.current_image = None
        layout = QVBoxLayout(self)
        
        # Close button
        close_layout = QHBoxLayout()
        self.close_btn = QPushButton("×")
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 10px;
                min-width: 20px;
                min-height: 20px;
                max-width: 20px;
                max-height: 20px;
            }
        """)
        self.close_btn.clicked.connect(self.closed.emit)
        close_layout.addStretch()
        close_layout.addWidget(self.close_btn)
        layout.addLayout(close_layout)
        
        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

    def load_image(self, path: str):
        self.current_image = path
        pixmap = QPixmap(path)
        scaled = pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.current_image:
            self.load_image(self.current_image)

    def clear(self):
        self.current_image = None
        self.image_label.clear()

class GalleryView(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedLayout()
        
        # Grid view
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid = QGridLayout(self.grid_widget)
        self.grid.setSpacing(5)
        self.scroll.setWidget(self.grid_widget)
        
        # Full image view
        self.full_view = FullImageView()
        self.full_view.closed.connect(self.show_grid)
        
        # Add both views to stack
        container = QWidget()
        container.setLayout(self.stack)
        self.stack.addWidget(self.scroll)
        self.stack.addWidget(self.full_view)
        
        layout.addWidget(container)

    def display_images(self, images: list[ImageData]):
        # Clear current grid
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Calculate columns based on width
        width = self.scroll.viewport().width()
        thumb_width = 155  # 150 + spacing
        columns = max(1, width // thumb_width)

        # Add thumbnails to grid
        for idx, image_data in enumerate(images):
            thumb = ThumbnailWidget(image_data)
            thumb.clicked.connect(self.show_full_image)
            self.grid.addWidget(thumb, idx // columns, idx % columns)

    def show_full_image(self, path: str):
        self.full_view.load_image(path)
        self.stack.setCurrentWidget(self.full_view)

    def show_grid(self):
        self.stack.setCurrentWidget(self.scroll)

    def clear(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.full_view.clear()
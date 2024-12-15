from PySide6.QtWidgets import (QScrollArea, QWidget, QGridLayout, QVBoxLayout, 
                              QLabel, QPushButton, QStackedWidget, QHBoxLayout)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap

class ThumbnailWidget(QLabel):
    clicked = Signal(str)

    def __init__(self, image_data):
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
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Close button
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
            QPushButton:hover {
                background-color: #ff6666;
            }
        """)
        self.close_btn.clicked.connect(self.closed.emit)
        
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.close_btn)
        layout.addLayout(btn_layout)
        
        # Image label
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_label)

    def load_image(self, path):
        self.current_image = path
        pixmap = QPixmap(path)
        if not pixmap.isNull():
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

        self.stack = QStackedWidget()
        
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
        
        self.stack.addWidget(self.scroll)
        self.stack.addWidget(self.full_view)
        
        layout.addWidget(self.stack)

    def display_images(self, images):
        # Clear current grid
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Add new thumbnails
        columns = max(1, self.scroll.viewport().width() // 160)
        for idx, image_data in enumerate(images):
            thumb = ThumbnailWidget(image_data)
            thumb.clicked.connect(self.show_full_image)
            self.grid.addWidget(thumb, idx // columns, idx % columns)

    def show_full_image(self, path):
        self.full_view.load_image(path)
        self.stack.setCurrentWidget(self.full_view)

    def show_grid(self):
        self.full_view.clear()
        self.stack.setCurrentWidget(self.scroll)

    def clear(self):
        # Clear grid
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        # Clear full view
        self.full_view.clear()
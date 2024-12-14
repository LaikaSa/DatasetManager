from PySide6.QtWidgets import (QScrollArea, QWidget, QGridLayout, QStackedLayout, 
                              QPushButton, QVBoxLayout, QLabel, QHBoxLayout)
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QPixmap

class MaximizedImage(QWidget):
    def __init__(self, scroll_area, parent=None):
        super().__init__(parent)
        self.scroll_area = scroll_area  # Store reference to scroll area
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        
        # Close button
        self.close_btn = QPushButton("×", self)
        self.close_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff4444;
                color: white;
                border: none;
                border-radius: 10px;
                padding: 5px;
                min-width: 20px;
                min-height: 20px;
                max-width: 20px;
                max-height: 20px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
        """)
        
        # Image label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Add widgets to layout
        close_layout = QHBoxLayout()
        close_layout.setContentsMargins(0, 0, 0, 0)
        close_layout.addStretch()
        close_layout.addWidget(self.close_btn)
        
        self.layout.addLayout(close_layout)
        self.layout.addWidget(self.image_label, 1)
        self.layout.addStretch()

    def set_image(self, image_path):
        self.original_pixmap = QPixmap(image_path)
        self.resize_image()

    def resize_image(self):
        if hasattr(self, 'original_pixmap'):
            # Get the available space from the scroll area
            viewport_size = self.scroll_area.viewport().size()
            available_width = viewport_size.width() - 40
            available_height = viewport_size.height() - 40
            
            scaled_pixmap = self.original_pixmap.scaled(
                available_width, 
                available_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            self.setFixedSize(viewport_size)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resize_image()


class GalleryGrid(QScrollArea):
    def __init__(self):
        super().__init__()
        self.thumbnails = []
        self.current_thumbnails = []
        self.init_ui()
        
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self.do_refresh)

    def init_ui(self):
        self.setWidgetResizable(True)
        
        # Main container that holds both grid and maximized view
        self.main_container = QWidget()
        self.main_layout = QStackedLayout(self.main_container)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Grid container
        self.grid_container = QWidget()
        self.grid = QGridLayout(self.grid_container)
        self.grid.setSpacing(5)
        
        # Maximized image container
        self.maximized_image = MaximizedImage(self, self.main_container)
        self.maximized_image.close_btn.clicked.connect(self.return_to_grid)
        
        # Add both to stack
        self.main_layout.addWidget(self.grid_container)
        self.main_layout.addWidget(self.maximized_image)
        
        self.setWidget(self.main_container)

    def on_thumbnail_clicked(self, image_path):
        self.maximized_image.set_image(image_path)
        self.main_layout.setCurrentWidget(self.maximized_image)
        self.verticalScrollBar().setValue(0)

    def return_to_grid(self):
        self.main_layout.setCurrentWidget(self.grid_container)
        self.do_refresh()

    def display_images(self, visible_thumbnails):
        # Store both the full list and current visible thumbnails
        self.thumbnails = list(visible_thumbnails)  # Make a copy
        self.current_thumbnails = self.thumbnails
        self.refresh_timer.start(100)

    def do_refresh(self):
        # Clear current layout while preserving widgets
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().hide()  # Hide instead of changing parent

        if not self.current_thumbnails:
            return

        # Calculate columns
        width = self.viewport().width()
        thumb_width = 155
        columns = max(1, width // thumb_width)

        # Add thumbnails to grid
        for idx, thumb in enumerate(self.current_thumbnails):
            row = idx // columns
            col = idx % columns
            
            thumb.show()  # Show the thumbnail
            self.grid.addWidget(thumb, row, col)
            
            # Reconnect click signal
            try:
                thumb.clicked.disconnect()
            except:
                pass
            thumb.clicked.connect(self.on_thumbnail_clicked)

        # Update layouts
        self.grid_container.updateGeometry()
        self.grid.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.main_layout.currentWidget() == self.maximized_image:
            self.maximized_image.resize_image()
        else:
            self.do_refresh()
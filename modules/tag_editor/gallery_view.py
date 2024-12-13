from PySide6.QtWidgets import (QScrollArea, QWidget, QGridLayout, 
                              QLabel, QVBoxLayout, QHBoxLayout)
from PySide6.QtGui import QPixmap, QPalette, QColor
from PySide6.QtCore import Qt, QSize, Signal
import os
from .tag_manager import TagManager
from .tag_filter import TagFilter

class ThumbnailLabel(QLabel):
    clicked = Signal(str)  # Signal to emit the image path when clicked

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path
        self.setStyleSheet("""
            QLabel {
                border: 2px solid transparent;
                background-color: #2d2d2d;
                padding: 3px;
            }
        """)
        self.setSelected(False)

    def mousePressEvent(self, event):
        self.clicked.emit(self.image_path)

    def setSelected(self, selected):
        self.is_selected = selected
        if selected:
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid #007bff;
                    background-color: #3d3d3d;
                    padding: 3px;
                }
            """)
        else:
            self.setStyleSheet("""
                QLabel {
                    border: 2px solid transparent;
                    background-color: #2d2d2d;
                    padding: 3px;
                }
            """)

class GalleryView(QWidget):
    image_selected = Signal(str)

    def __init__(self, tag_manager):
        super().__init__()
        self.tag_manager = tag_manager
        self.thumbnail_size = QSize(150, 150)
        self.images = []
        self.thumbnails = []
        self.current_filter = TagFilter()
        self.init_ui()

    def init_ui(self):
        # Main horizontal layout for split view
        self.main_layout = QHBoxLayout(self)
        
        # Left side - Gallery
        self.gallery_widget = QWidget()
        self.gallery_layout = QVBoxLayout(self.gallery_widget)
        
        # Scroll area for gallery
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Content widget for grid layout
        self.content_widget = QWidget()
        self.grid_layout = QGridLayout(self.content_widget)
        self.grid_layout.setSpacing(10)
        self.scroll_area.setWidget(self.content_widget)
        self.gallery_layout.addWidget(self.scroll_area)

        # Right side - Placeholder for now
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        # Add both sides to main layout with 1:1 ratio
        self.main_layout.addWidget(self.gallery_widget, 1)
        self.main_layout.addWidget(self.right_panel, 1)

    def reflow_gallery(self):
        """Reflow visible thumbnails in grid layout"""
        # Remove all widgets from grid
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Calculate columns
        scroll_width = self.scroll_area.viewport().width()
        thumb_width = self.thumbnail_size.width() + self.grid_layout.spacing()
        columns = max(1, scroll_width // thumb_width)

        # Add only visible thumbnails
        visible_thumbs = [thumb for thumb in self.thumbnails if not thumb.isHidden()]
        for idx, thumb in enumerate(visible_thumbs):
            row = idx // columns
            col = idx % columns
            self.grid_layout.addWidget(thumb, row, col)

        # Force layout update
        self.content_widget.updateGeometry()

    def filter_by_tags(self, tags: set):
        """Filter gallery using TagFilter logic"""
        if not tags:
            self.current_filter = TagFilter()
        else:
            self.current_filter = TagFilter(
                tags=tags,
                logic=TagFilter.Logic.OR,
                mode=TagFilter.Mode.INCLUSIVE
            )

        # Apply filter
        for thumb in self.thumbnails:
            image_tags = self.tag_manager.get_tags(thumb.image_path)
            visible = self.current_filter.matches(image_tags)
            thumb.setVisible(visible)

        self.reflow_gallery()

    def load_images(self, directory):
        self.clear_gallery()
        self.images = []
        self.thumbnails = []
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
        files = [f for f in os.listdir(directory) 
                if f.lower().endswith(valid_extensions)]

        for file in files:
            image_path = os.path.join(directory, file)
            self.images.append(image_path)
            
            # Create thumbnail
            pixmap = QPixmap(image_path)
            scaled_pixmap = pixmap.scaled(
                self.thumbnail_size, 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )

            # Create thumbnail label
            thumb_label = ThumbnailLabel(image_path)
            thumb_label.setPixmap(scaled_pixmap)
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.clicked.connect(self.on_thumbnail_clicked)
            self.thumbnails.append(thumb_label)

        # Initial display
        self.reflow_gallery()

    def clear_gallery(self):
        for thumb in self.thumbnails:
            thumb.deleteLater()
        self.thumbnails.clear()
        self.images.clear()
        self.current_selected = None

    def resizeEvent(self, event):
        """Handle resize events to reflow gallery"""
        super().resizeEvent(event)
        self.reflow_gallery()

    def on_thumbnail_clicked(self, image_path):
        # Deselect previous selection
        if self.current_selected:
            for i in range(self.grid_layout.count()):
                widget = self.grid_layout.itemAt(i).widget()
                if isinstance(widget, ThumbnailLabel) and widget.image_path == self.current_selected:
                    widget.setSelected(False)
                    break

        # Select new thumbnail
        for i in range(self.grid_layout.count()):
            widget = self.grid_layout.itemAt(i).widget()
            if isinstance(widget, ThumbnailLabel) and widget.image_path == image_path:
                widget.setSelected(True)
                self.current_selected = image_path
                self.image_selected.emit(image_path)  # Emit signal with selected image path
                break

    def get_selected_image(self):
        return self.current_selected
    

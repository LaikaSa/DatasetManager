from PySide6.QtWidgets import QScrollArea, QWidget, QGridLayout
from PySide6.QtCore import Qt

class GalleryGrid(QScrollArea):
    def __init__(self):
        super().__init__()
        self.thumbnails = []
        self.init_ui()

    def init_ui(self):
        self.setWidgetResizable(True)
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(5)
        self.setWidget(self.container)

    def display_images(self, visible_thumbnails):
        # Clear current layout
        while self.grid.count():
            self.grid.takeAt(0).widget().setParent(None)

        # Calculate columns based on width
        width = self.viewport().width()
        thumb_width = 155  # 150 + spacing
        columns = max(1, width // thumb_width)

        # Add visible thumbnails to grid
        for idx, thumb in enumerate(visible_thumbnails):
            row = idx // columns
            col = idx % columns
            self.grid.addWidget(thumb, row, col)
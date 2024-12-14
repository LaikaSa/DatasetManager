from PySide6.QtWidgets import QScrollArea, QWidget, QGridLayout
from PySide6.QtCore import Qt, QTimer

class GalleryGrid(QScrollArea):
    def __init__(self):
        super().__init__()
        self.thumbnails = []
        self.current_thumbnails = []  # Keep track of currently displayed thumbnails
        self.init_ui()
        
        # Add delayed refresh mechanism
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self.do_refresh)

    def init_ui(self):
        self.setWidgetResizable(True)
        self.container = QWidget()
        self.grid = QGridLayout(self.container)
        self.grid.setSpacing(5)
        self.setWidget(self.container)

    def display_images(self, visible_thumbnails):
        # Store new thumbnails and trigger delayed refresh
        self.current_thumbnails = visible_thumbnails
        self.refresh_timer.start(100)  # 100ms delay

    def do_refresh(self):
        # Clear current layout
        while self.grid.count():
            item = self.grid.takeAt(0)
            if item.widget():
                item.widget().setParent(None)

        # Calculate columns based on width
        width = self.viewport().width()
        thumb_width = 155  # 150 + spacing
        columns = max(1, width // thumb_width)

        # Add thumbnails to grid
        for idx, thumb in enumerate(self.current_thumbnails):
            row = idx // columns
            col = idx % columns
            self.grid.addWidget(thumb, row, col)

        # Force layout update
        self.container.updateGeometry()
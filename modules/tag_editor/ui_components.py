from PySide6.QtWidgets import QWidget, QHBoxLayout, QLineEdit, QPushButton, QFileDialog
from PySide6.QtCore import Signal

class PathInputBar(QWidget):
    path_changed = Signal(str)

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Enter directory path...")
        self.browse_button = QPushButton("Browse")

        layout.addWidget(self.path_input)
        layout.addWidget(self.browse_button)

        self.browse_button.clicked.connect(self.browse_directory)

    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.path_input.setText(directory)
            self.path_changed.emit(directory)

class LoadControls(QWidget):
    load_clicked = Signal()
    unload_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.load_button = QPushButton("Load")
        self.unload_button = QPushButton("Unload")

        layout.addWidget(self.load_button)
        layout.addWidget(self.unload_button)

        self.load_button.clicked.connect(self.load_clicked.emit)
        self.unload_button.clicked.connect(self.unload_clicked.emit)
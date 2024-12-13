from PySide6.QtWidgets import QWidget, QVBoxLayout
from .ui_components import PathInputBar, LoadControls
from .gallery_view import GalleryView
from .tag_manager import TagManager
from .filter_panel import FilterByTagsPanel

class TagEditorTab(QWidget):
    def __init__(self):
        super().__init__()
        self.tag_manager = TagManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Initialize components
        self.path_input = PathInputBar()
        self.load_controls = LoadControls()
        self.gallery = GalleryView(self.tag_manager)
        
        # Create filter panel
        self.filter_panel = FilterByTagsPanel(self.tag_manager)
        self.gallery.right_layout.addWidget(self.filter_panel)

        # Add components to layout
        layout.addWidget(self.path_input)
        layout.addWidget(self.load_controls)
        layout.addWidget(self.gallery)

        # Connect signals
        self.path_input.path_changed.connect(self.on_path_changed)
        self.load_controls.load_clicked.connect(self.on_load)
        self.load_controls.unload_clicked.connect(self.on_unload)
        self.gallery.image_selected.connect(self.on_image_selected)
        self.filter_panel.filter_changed.connect(self.on_filter_changed)

    def on_filter_changed(self, tags):
        """Handle tag filter changes"""
        print(f"Filtering for tags: {tags}")  # Debug print
        self.gallery.filter_by_tags(tags)

    def on_path_changed(self, path):
        self.current_directory = path

    def on_load(self):
        if hasattr(self, 'current_directory'):
            print(f"Loading from directory: {self.current_directory}")  # Debug print
            
            # First load images
            self.gallery.load_images(self.current_directory)
            
            # Then load tags for each image
            for image_path in self.gallery.images:
                self.tag_manager.load_tags(image_path)
            
            print(f"Loaded {len(self.gallery.images)} images")  # Debug print
            
            # Update the filter panel with the loaded tags
            self.filter_panel.update_tags(self.gallery.images)

    def on_unload(self):
        self.gallery.clear_gallery()

    def on_image_selected(self, image_path):
        # Handle image selection
        # This will be useful when we implement the tag editing features
        pass
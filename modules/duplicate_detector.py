import os
import imagehash
from PIL import Image
import cv2
import numpy as np
import io
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel,
                              QFileDialog, QMessageBox, QProgressBar, QCheckBox,
                              QSlider, QHBoxLayout, QGroupBox, QScrollArea, QLineEdit,
                              QApplication)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QThreadPool, QRunnable, QObject
from PySide6.QtGui import QPixmap, QImage
from send2trash import send2trash
from modules.logger import setup_logger

logger = setup_logger()
# QPixmap cache: path -> pixmap. QPixmap is a GUI object, so it is only ever
# created on the main thread (the bridge signal is delivered there).
thumbnail_cache = {}


def create_thumbnail_bytes(image_path, max_size=200):
    """Decode + resize to JPEG bytes (thread-safe: no Qt objects involved)."""
    try:
        with Image.open(image_path) as img:
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')

            width, height = img.size
            ratio = min(max_size / width, max_size / height)
            new_width = max(1, int(width * ratio))
            new_height = max(1, int(height * ratio))

            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img_resized.save(buffer, format='JPEG')
            return buffer.getvalue()
    except Exception as e:
        logger.error(f"Error creating thumbnail for {image_path}: {str(e)}")
        return None


class _ThumbnailBridge(QObject):
    """Delivers decoded thumbnail bytes from worker threads to the GUI thread."""
    ready = Signal(str, bytes)


_thumbnail_bridge = _ThumbnailBridge()
_thumbnail_pool = QThreadPool()
_thumbnail_pool.setMaxThreadCount(max(2, min(8, os.cpu_count() or 4)))


class _ThumbnailJob(QRunnable):
    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        data = create_thumbnail_bytes(self.image_path)
        if data is not None:
            _thumbnail_bridge.ready.emit(self.image_path, data)


def _on_thumbnail_ready(image_path, data):
    if image_path in thumbnail_cache:
        return
    qimg = QImage.fromData(data)
    if qimg.isNull():
        return
    thumbnail_cache[image_path] = QPixmap.fromImage(qimg)


_thumbnail_bridge.ready.connect(_on_thumbnail_ready)


def request_thumbnail(image_path):
    """Ensure a thumbnail for path exists (cached, main thread)."""
    if image_path in thumbnail_cache:
        return thumbnail_cache[image_path]
    _thumbnail_pool.start(_ThumbnailJob(image_path))
    return None


def clear_thumbnail_cache(image_paths=None):
    """Drop cached pixmaps (all of them, or just the given paths)."""
    if image_paths is None:
        thumbnail_cache.clear()
    else:
        for p in image_paths:
            thumbnail_cache.pop(p, None)

class ImagePreviewGroup(QWidget):
    def __init__(self, images, similarity, method, selected_images, selection_callback,
                 sizes=None):
        super().__init__()
        self.selected_images = selected_images
        self.selection_callback = selection_callback
        self.images = images
        self.sizes = sizes or {}  # path -> (width, height), from the detection worker
        self.containers = []
        self.init_ui(similarity, method)

    def init_ui(self, similarity, method):
        layout = QVBoxLayout()
        
        # Add similarity info
        similarity_label = QLabel(f"{method} Similarity: {similarity:.2%}")
        similarity_label.setStyleSheet("font-weight: bold; color: #2962FF;")
        layout.addWidget(similarity_label)

        # Create scroll area for images
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create widget to hold images
        image_widget = QWidget()
        image_layout = QHBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create containers for all images but load thumbnails later
        for img_path in self.images:
            try:
                img_container = ClickableImageContainer(
                    img_path,
                    img_path in self.selected_images,
                    self.on_image_clicked,
                    size=self.sizes.get(img_path)
                )
                image_layout.addWidget(img_container)
                self.containers.append(img_container)
                
            except Exception as e:
                logger.error(f"Error creating container for {img_path}: {str(e)}")

        image_layout.addStretch()
        scroll_area.setWidget(image_widget)
        scroll_area.setMinimumHeight(250)
        
        layout.addWidget(scroll_area)
        self.setLayout(layout)

    def on_image_clicked(self, img_path, is_selected):
        self.selection_callback(img_path, is_selected)

class ClickableImageContainer(QWidget):
    def __init__(self, img_path, is_selected, callback, size=None):
        super().__init__()
        self.img_path = img_path
        self.is_selected = is_selected
        self.callback = callback
        self.thumbnail_loaded = False
        self.init_ui(size)

    def init_ui(self, size):
        layout = QVBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)

        # Image label with loading placeholder
        self.img_label = QLabel("Loading...")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(200, 200)

        # Resolution label with file type. The detection worker already read
        # every image once, so prefer the size it reports - this avoids an
        # extra decode on the main thread per container.
        if size is None:
            try:
                with Image.open(self.img_path) as img:
                    size = img.size
            except Exception:
                size = None
        ext = os.path.splitext(self.img_path)[1].lower()
        if size is not None:
            res_text = f"{size[0]} × {size[1]} ({ext})"
        else:
            res_text = f"({ext})"
        self.resolution_label = QLabel(res_text)
        self.resolution_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.img_label)
        layout.addWidget(self.resolution_label)
        self.setLayout(layout)
        self.setFixedWidth(220)

        # Set initial style based on selection state
        self.update_style()

        # Start loading thumbnail in background (worker threads, not this one)
        _thumbnail_bridge.ready.connect(self._on_thumbnail_ready)
        self.load_thumbnail_later()

    def _on_thumbnail_ready(self, image_path, _data):
        if image_path != self.img_path:
            return
        self.load_thumbnail()

    def closeEvent(self, event):
        try:
            _thumbnail_bridge.ready.disconnect(self._on_thumbnail_ready)
        except (TypeError, RuntimeError):
            pass
        super().closeEvent(event)

    def load_thumbnail_later(self):
        QTimer.singleShot(10, self.load_thumbnail)

    def load_thumbnail(self):
        if self.thumbnail_loaded:
            return
        pixmap = request_thumbnail(self.img_path)
        if pixmap:
            self.img_label.setPixmap(pixmap)
            self.thumbnail_loaded = True

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_selected = not self.is_selected
            self.update_style()
            self.callback(self.img_path, self.is_selected)
        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.globalPos())

    def show_context_menu(self, global_pos):
        from PySide6.QtWidgets import QMenu
        menu = QMenu(self)
        open_action = menu.addAction("Open in Folder")
        action = menu.exec(global_pos)
        if action == open_action:
            self.open_in_folder()

    def open_in_folder(self):
        import ctypes
        normalized = os.path.normpath(self.img_path)
        shell32 = ctypes.windll.shell32
        shell32.ILCreateFromPathW.argtypes = [ctypes.c_wchar_p]
        shell32.ILCreateFromPathW.restype = ctypes.c_void_p
        shell32.SHOpenFolderAndSelectItems.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_ulong
        ]
        shell32.SHOpenFolderAndSelectItems.restype = ctypes.HRESULT
        shell32.ILFree.argtypes = [ctypes.c_void_p]
        shell32.ILFree.restype = None
        pidl = shell32.ILCreateFromPathW(normalized)
        if pidl:
            shell32.SHOpenFolderAndSelectItems(pidl, 0, None, 0)
            shell32.ILFree(pidl)

    def update_style(self):
        if self.is_selected:
            self.setStyleSheet("""
                QWidget {
                    background-color: rgba(0, 120, 215, 0.3);
                    border: 2px solid #0078D7;
                    border-radius: 5px;
                }
            """)
        else:
            self.setStyleSheet("")

class WorkerThread(QThread):
    progress = Signal(int)
    result = Signal(dict)
    finished = Signal()

    def __init__(self, folder_path, use_hash, use_hist, hash_threshold, hist_threshold):
        super().__init__()
        self.folder_path = folder_path
        self.use_hash = use_hash
        self.use_hist = use_hist
        self.hash_threshold = hash_threshold
        self.hist_threshold = hist_threshold
        self.is_running = True

    def run(self):
        def print_progress_bar(current, total, prefix='Progress:', length=50):
            filled_length = int(length * current / total)
            bar = '=' * filled_length + '-' * (length - filled_length)
            print(f'\r{prefix} [{bar}] {current}/{total}', end='', flush=True)

        image_files = []
        for file in os.listdir(self.folder_path):
            file_path = os.path.join(self.folder_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_files.append(file_path)

        total_files = len(image_files)
        logger.info(f"Found {total_files} images to process")
        
        # Store all images with their features
        image_features = {}
        sizes = {}  # path -> (width, height), collected here so the preview UI
        # doesn't have to re-decode every image on the main thread

        # First pass: calculate features for all images
        logger.info("First pass: Calculating features")
        for idx, img_path in enumerate(image_files):
            if not self.is_running:
                break

            try:
                features = {}
                if self.use_hash:
                    with Image.open(img_path) as img:
                        features['hash'] = imagehash.average_hash(img)
                        sizes[img_path] = img.size

                if self.use_hist:
                    img_cv = cv2.imread(img_path)
                    if img_cv is not None:
                        hist = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8],
                                        [0, 256, 0, 256, 0, 256])
                        features['hist'] = cv2.normalize(hist, hist).flatten()
                        if img_path not in sizes:
                            h, w = img_cv.shape[:2]
                            sizes[img_path] = (w, h)

                image_features[img_path] = features
                print_progress_bar(idx + 1, total_files, prefix='Processing:')

            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}", exc_info=True)

        print()  # New line after first pass

        # Second pass: group similar images (vectorized - one matmul instead of
        # an O(n^2) Python loop of cv2.compareHist / hash subtractions)
        logger.info("Second pass: Comparing images")
        groups = []
        paths = list(image_features.keys())
        n = len(paths)

        if n > 1 and self.is_running:
            hash_sim = None
            hash_valid = None
            corr = None
            hist_valid = None

            if self.use_hash:
                bits = np.full((n, 64), -1, dtype=np.int32)
                for i, p in enumerate(paths):
                    h = image_features[p].get('hash')
                    if h is not None:
                        # imagehash stores the 64-bit hash (here: 8x8 bool array)
                        hv = np.asarray(h.hash)
                        if hv.dtype == bool or hv.ndim > 1:
                            bits[i] = hv.ravel()[:64].astype(np.int32)
                        else:
                            v = int(hv.ravel()[0])
                            bits[i] = np.array([(v >> (63 - k)) & 1 for k in range(64)],
                                               dtype=np.int32)
                hash_valid = bits[:, 0] >= 0
                b = np.where(hash_valid[:, None], bits, 0)
                pop = b.sum(axis=1)
                hamming = pop[:, None] + pop[None, :] - 2 * (b @ b.T)
                hash_sim = 1.0 - hamming / 64.0

            if self.use_hist:
                d = 512  # 8x8x8 histogram
                H = np.zeros((n, d), dtype=np.float64)
                for i, p in enumerate(paths):
                    hv = image_features[p].get('hist')
                    if hv is not None:
                        H[i] = np.asarray(hv, dtype=np.float64).ravel()[:d]
                hist_valid = np.array([image_features[p].get('hist') is not None for p in paths])
                Hv = np.where(hist_valid[:, None], H, 0.0)
                # Pearson correlation in one matmul (matches cv2.HISTCMP_CORREL)
                C = Hv - Hv.mean(axis=1, keepdims=True)
                norms = np.sqrt((C * C).sum(axis=1))
                M = C @ C.T
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr = M / np.outer(norms, norms)
                bad = (norms[:, None] == 0) | (norms[None, :] == 0)
                corr = np.where(bad, -2.0, corr)

            all_idx = np.arange(n)
            processed = set()
            processed_mask = np.zeros(n, dtype=bool)
            processed_count = 0

            for i, img_path in enumerate(paths):
                if i in processed:
                    continue

                qualifies = np.zeros(n, dtype=bool)
                val_h = np.full(n, -np.inf)
                val_s = np.full(n, -np.inf)

                if hash_sim is not None and hash_valid[i]:
                    mask = hash_valid & (hash_sim[i] >= self.hash_threshold)
                    qualifies |= mask & ~processed_mask & (all_idx != i)
                    val_h = np.where(mask & ~processed_mask & (all_idx != i), hash_sim[i], -np.inf)

                if corr is not None and hist_valid[i]:
                    mask = hist_valid & (corr[i] >= self.hist_threshold)
                    qualifies |= mask & ~processed_mask & (all_idx != i)
                    val_s = np.where(mask & ~processed_mask & (all_idx != i), corr[i], -np.inf)

                qualifies &= ~processed_mask
                qualifies[i] = False

                if qualifies.any():
                    # Per-pair value: hash is checked first in the original
                    # algorithm and wins ties, so 'Histogram' only if it is
                    # strictly larger.
                    pair_val = np.maximum(val_h, val_s)
                    j_star = int(np.argmax(np.where(qualifies, pair_val, -np.inf)))
                    similarity = float(pair_val[j_star])
                    method = 'Histogram' if val_s[j_star] > val_h[j_star] else 'Hash'

                    group_idx = [i] + [int(j) for j in np.flatnonzero(qualifies)]
                    groups.append({
                        'images': [paths[j] for j in group_idx],
                        'similarity': similarity,
                        'method': method,
                        'sizes': {paths[j]: sizes.get(paths[j]) for j in group_idx}
                    })
                    processed.update(group_idx)
                    processed_mask[group_idx] = True

                processed_count += 1
                if processed_count % 50 == 0:
                    print_progress_bar(processed_count, total_files, prefix='Comparing:')
                if not self.is_running:
                    break

        print()  # New line after second pass

        # Emit all groups
        if groups:
            logger.info(f"Found {len(groups)} groups of similar images")
            for group in groups:
                self.result.emit(group)
        else:
            logger.info("No duplicate images found")

        self.finished.emit()

    def stop(self):
        self.is_running = False

class CompareWindow(QWidget):
    def __init__(self, image_paths, parent=None):
        super().__init__(parent)
        self.image_paths = image_paths
        self.setWindowTitle("Compare Images")
        self.showMaximized()
        self.init_ui()

    def init_ui(self):
        from PySide6.QtWidgets import QSizePolicy
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Scroll area so images don't get cut off if there are many
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        container = QWidget()
        images_layout = QHBoxLayout(container)
        images_layout.setSpacing(8)

        screen = QApplication.primaryScreen().availableGeometry()
        available_w = screen.width() - 40
        available_h = screen.height() - 100
        n = len(self.image_paths)
        max_w_each = max(1, available_w // n) - 8

        for img_path in self.image_paths:
            col = QWidget()
            col_layout = QVBoxLayout(col)
            col_layout.setContentsMargins(0, 0, 0, 0)
            col_layout.setSpacing(4)

            try:
                with Image.open(img_path) as img:
                    orig_w, orig_h = img.size
                ext = os.path.splitext(img_path)[1].lower()

                # Scale to fit, honouring both max_w_each and available_h
                ratio = min(max_w_each / orig_w, (available_h - 40) / orig_h, 1.0)
                disp_w = int(orig_w * ratio)
                disp_h = int(orig_h * ratio)

                pixmap = QPixmap(img_path).scaled(
                    disp_w, disp_h,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                img_label = QLabel()
                img_label.setPixmap(pixmap)
                img_label.setAlignment(Qt.AlignCenter)

                info_label = QLabel(
                    f"{os.path.basename(img_path)}\n"
                    f"{orig_w} × {orig_h} ({ext})"
                )
                info_label.setAlignment(Qt.AlignCenter)
                info_label.setWordWrap(True)

            except Exception as e:
                img_label = QLabel(f"Error loading image:\n{str(e)}")
                img_label.setAlignment(Qt.AlignCenter)
                info_label = QLabel(os.path.basename(img_path))
                info_label.setAlignment(Qt.AlignCenter)

            col_layout.addWidget(img_label)
            col_layout.addWidget(info_label)
            col.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
            images_layout.addWidget(col)

        images_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(120)
        close_btn.clicked.connect(self.close)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

class DuplicateDetectorTab(QWidget):
    def __init__(self):
        super().__init__()
        logger.info("Initializing Duplicate Detector Tab")
        self.worker = None
        self.folder_path = None
        self.current_group_index = 0
        self.image_groups = []
        self.selected_images = set()  # Add this line
        self.init_ui()

    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Create folder selection layout
        folder_layout = QHBoxLayout()
        self.folder_path_input = QLineEdit()
        self.folder_path_input.setPlaceholderText("Enter or paste folder path here")
        self.browse_btn = QPushButton("Browse")
        folder_layout.addWidget(self.folder_path_input)
        folder_layout.addWidget(self.browse_btn)

        # Status label
        self.status_label = QLabel("No folder selected")

        # Add folder selection to main layout
        main_layout.addLayout(folder_layout)
        main_layout.addWidget(self.status_label)

        # Create method selection group
        method_group = QGroupBox("Detection Methods")
        method_layout = QVBoxLayout()

        # Checkbox layout (horizontal)
        checkbox_layout = QHBoxLayout()
        self.hash_checkbox = QCheckBox("Use Perceptual Hashing")
        self.hist_checkbox = QCheckBox("Use Color Histogram")
        checkbox_layout.addWidget(self.hash_checkbox)
        checkbox_layout.addWidget(self.hist_checkbox)
        checkbox_layout.addStretch()

        # Hashing controls in a collapsible widget
        self.hash_controls = QWidget()
        hash_layout = QVBoxLayout(self.hash_controls)
        
        hash_description = QLabel(
            "Hash Similarity (0-100):\n"
            "→ Slide right for stricter matching (more similar)\n"
            "← Slide left for looser matching (less similar)\n"
            "Recommended: 70-90% for best results"
        )
        hash_description.setWordWrap(True)
        
        self.hash_slider = QSlider(Qt.Horizontal)
        self.hash_slider.setRange(0, 100)
        self.hash_slider.setValue(90)
        
        self.hash_value_label = QLabel(f"Current threshold: {self.hash_slider.value()}%")
        
        hash_layout.addWidget(hash_description)
        hash_layout.addWidget(self.hash_slider)
        hash_layout.addWidget(self.hash_value_label)
        hash_layout.setContentsMargins(20, 0, 20, 0)
        self.hash_controls.setVisible(False)

        # Histogram controls in a collapsible widget
        self.hist_controls = QWidget()
        hist_layout = QVBoxLayout(self.hist_controls)
        
        hist_description = QLabel(
            "Histogram Correlation (0-100):\n"
            "← Slide left for looser matching (less similar)\n"
            "→ Slide right for stricter matching (more similar)"
        )
        hist_description.setWordWrap(True)
        
        self.hist_slider = QSlider(Qt.Horizontal)
        self.hist_slider.setRange(0, 100)
        self.hist_slider.setValue(95)
        
        self.hist_value_label = QLabel(f"Current threshold: {self.hist_slider.value()}%")
        
        hist_layout.addWidget(hist_description)
        hist_layout.addWidget(self.hist_slider)
        hist_layout.addWidget(self.hist_value_label)
        hist_layout.setContentsMargins(20, 0, 20, 0)
        self.hist_controls.setVisible(False)

        # Add all elements to method layout
        method_layout.addLayout(checkbox_layout)
        method_layout.addWidget(self.hash_controls)
        method_layout.addWidget(self.hist_controls)
        method_group.setLayout(method_layout)

        # Create control buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.setEnabled(False)
        self.stop_btn = QPushButton("Stop Detection")
        self.stop_btn.setEnabled(False)
        self.compare_btn = QPushButton("Compare")
        self.compare_btn.setEnabled(False)
        self.recycle_btn = QPushButton("Move Selected to Recycle Bin")
        self.recycle_btn.setEnabled(False)

        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        button_layout.addWidget(self.compare_btn)
        button_layout.addWidget(self.recycle_btn)

        # Connect buttons
        self.recycle_btn.clicked.connect(self.move_to_recycle_bin)
        self.compare_btn.clicked.connect(self.open_compare_window)

        # Preview area
        self.preview_area = QScrollArea()
        self.preview_area.setWidgetResizable(True)
        self.preview_area.setMinimumHeight(300)
        
        # Navigation controls
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("Previous Group")
        self.next_btn = QPushButton("Next Group")
        self.group_label = QLabel("No duplicates found")
        
        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.group_label)
        nav_layout.addWidget(self.next_btn)
        
        # Connect signals
        self.hash_slider.valueChanged.connect(self.update_hash_label)
        self.hist_slider.valueChanged.connect(self.update_hist_label)
        self.start_btn.clicked.connect(self.start_detection)
        self.stop_btn.clicked.connect(self.stop_detection)
        self.prev_btn.clicked.connect(self.show_previous_group)
        self.next_btn.clicked.connect(self.show_next_group)
        self.hash_checkbox.stateChanged.connect(self.on_hash_checkbox_changed)
        self.hist_checkbox.stateChanged.connect(self.on_hist_checkbox_changed)
        
        # Initially disable navigation buttons
        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        # Add all widgets to main layout
        main_layout.addWidget(method_group)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.preview_area)
        main_layout.addLayout(nav_layout)

        self.setLayout(main_layout)

        # Update connections
        self.browse_btn.clicked.connect(self.browse_folder)
        self.folder_path_input.textChanged.connect(self.on_path_changed)

    def display_current_group(self):
        if not self.image_groups:
            return

        group = self.image_groups[self.current_group_index]
        preview_widget = ImagePreviewGroup(
            images=group['images'],
            similarity=group['similarity'],
            method=group['method'],
            selected_images=self.selected_images,
            selection_callback=self.on_selection_changed,
            sizes=group.get('sizes')
        )
        self.preview_area.setWidget(preview_widget)
        self.update_navigation_buttons()
        self.recycle_btn.setEnabled(len(self.selected_images) > 0)
        self.compare_btn.setEnabled(True)

    def open_compare_window(self):
        if not self.image_groups:
            return
        group = self.image_groups[self.current_group_index]
        self.compare_window = CompareWindow(group['images'])
        self.compare_window.show()

    def on_selection_changed(self, img_path, is_selected):
        if is_selected:
            self.selected_images.add(img_path)
        else:
            self.selected_images.discard(img_path)
        self.recycle_btn.setEnabled(len(self.selected_images) > 0)

    def move_to_recycle_bin(self):
        logger.info(f"Moving {len(self.selected_images)} images to recycle bin")
        if not self.selected_images:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Move {len(self.selected_images)} selected images to recycle bin?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            failed_files = []

            for img_path in list(self.selected_images):  # Create a copy of the list
                try:
                    # Normalize path to handle Windows paths correctly
                    normalized_path = os.path.normpath(img_path)
                    send2trash(normalized_path)
                    self.selected_images.remove(img_path)

                    # Remove the image from groups
                    for group in self.image_groups[:]:
                        group['images'] = [img for img in group['images'] if img != img_path]
                        if 'sizes' in group:
                            group['sizes'].pop(img_path, None)
                        if len(group['images']) < 2:
                            self.image_groups.remove(group)

                except Exception as e:
                    failed_files.append((img_path, str(e)))

            # Free the cached thumbnails of the trashed files
            clear_thumbnail_cache(list(self.selected_images))

            # Show results
            if failed_files:
                error_message = "Failed to move the following files to recycle bin:\n\n"
                for file_path, error in failed_files:
                    error_message += f"{os.path.basename(file_path)}: {error}\n"
                QMessageBox.warning(self, "Error", error_message)

            # Update display
            if self.image_groups:
                if self.current_group_index >= len(self.image_groups):
                    self.current_group_index = len(self.image_groups) - 1
                self.display_current_group()
            else:
                self.preview_area.setWidget(QWidget())
                self.group_label.setText("No duplicates found")
                self.current_group_index = 0

            self.update_navigation_buttons()
            self.recycle_btn.setEnabled(len(self.selected_images) > 0)

    def on_path_changed(self, path):
        logger.debug(f"Path changed to: {path}")
        path = path.strip()
        if os.path.exists(path) and os.path.isdir(path):
            logger.info(f"Valid folder path: {path}")
            self.folder_path = path
            self.status_label.setText(f"Selected folder: {path}")
            self.update_start_button()
        else:
            logger.warning(f"Invalid folder path: {path}")
            self.folder_path = None
            self.status_label.setText("Invalid folder path")
            self.update_start_button()

    def on_hash_checkbox_changed(self, state):
        """Handle hash checkbox state change"""
        self.hash_controls.setVisible(bool(state))
        self.update_start_button()

    def on_hist_checkbox_changed(self, state):
        """Handle histogram checkbox state change"""
        self.hist_controls.setVisible(bool(state))
        self.update_start_button()

    def update_hash_label(self):
        self.hash_value_label.setText(f"Current threshold: {self.hash_slider.value()}%")

    def update_hist_label(self):
        self.hist_value_label.setText(f"Current threshold: {self.hist_slider.value()}%")

    def browse_folder(self):
        logger.info("Opening folder browser")
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            logger.info(f"Selected folder: {folder_path}")
            self.folder_path_input.setText(folder_path)  # This will trigger on_path_changed

    def update_start_button(self):
        self.start_btn.setEnabled(
            (self.hash_checkbox.isChecked() or self.hist_checkbox.isChecked()) and 
            self.folder_path is not None
        )

    def start_detection(self):
        logger.info("Starting duplicate detection")
        logger.info(f"Hash detection: {self.hash_checkbox.isChecked()}")
        logger.info(f"Histogram detection: {self.hist_checkbox.isChecked()}")
        logger.info(f"Hash threshold: {self.hash_slider.value()}%")
        logger.info(f"Histogram threshold: {self.hist_slider.value()}%")
        if self.worker is not None and self.worker.isRunning():
            return

        self.image_groups = []
        self.current_group_index = 0
        self.preview_area.setWidget(QWidget())  # Clear preview area
        clear_thumbnail_cache()  # fresh scan, fresh cache
        
        self.worker = WorkerThread(
            self.folder_path,
            self.hash_checkbox.isChecked(),
            self.hist_checkbox.isChecked(),
            self.hash_slider.value() / 100,
            self.hist_slider.value() / 100
        )
        
        self.worker.result.connect(self.update_result)
        self.worker.finished.connect(self.detection_finished)
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.browse_btn.setEnabled(False)  # Changed from folder_btn to browse_btn
        self.folder_path_input.setEnabled(False)  # Also disable the path input during processing
        
        self.worker.start()

    def stop_detection(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
            self.detection_finished()

    def detection_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.browse_btn.setEnabled(True)  # Changed from folder_btn to browse_btn
        self.folder_path_input.setEnabled(True)  # Re-enable the path input
        
        total_groups = len(self.image_groups)
        if total_groups > 0:
            self.group_label.setText(f"Group {self.current_group_index + 1} of {total_groups}")
            self.update_navigation_buttons()
        else:
            self.group_label.setText("No duplicates found")

    def update_result(self, result_dict):
        self.add_duplicate_group(
            result_dict['images'],
            result_dict['similarity'],
            result_dict['method']
        )

    def add_duplicate_group(self, images, similarity, method):
        self.image_groups.append({
            'images': images,
            'similarity': similarity,
            'method': method
        })
        
        # If this is the first group, display it
        if len(self.image_groups) == 1:
            self.display_current_group()
            self.update_navigation_buttons()
        # Update group count
        self.group_label.setText(f"Group {self.current_group_index + 1} of {len(self.image_groups)}")

    def show_previous_group(self):
        if self.current_group_index > 0:
            self.current_group_index -= 1
            self.display_current_group()

    def show_next_group(self):
        if self.current_group_index < len(self.image_groups) - 1:
            self.current_group_index += 1
            self.display_current_group()


    def update_navigation_buttons(self):
        has_groups = len(self.image_groups) > 0
        if has_groups:
            self.prev_btn.setEnabled(self.current_group_index > 0)
            self.next_btn.setEnabled(self.current_group_index < len(self.image_groups) - 1)
            self.group_label.setText(f"Group {self.current_group_index + 1} of {len(self.image_groups)}")
        else:
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            self.compare_btn.setEnabled(False)
            self.group_label.setText("No duplicates found")
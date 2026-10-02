"""Thumbnail decode cache for the duplicate detector.

Decoding runs on a worker thread pool; the QPixmap (a GUI object) is only
ever created on the main thread via the bridge signal.
"""
import io
import os

from PIL import Image
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap

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
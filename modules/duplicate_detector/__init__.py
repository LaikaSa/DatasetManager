"""Duplicate detection tab.

The detection algorithm (detection.py) is pure Python/numpy, exposed
through the Qt-free engine.py entry point, and testable without a GUI;
the thumbnail cache (thumbnails.py) and the tab UI (ui.py) live beside
them.
"""
from .ui import DuplicateDetectorTab

__all__ = ["DuplicateDetectorTab"]
"""Duplicate detection tab.

The detection algorithm (detection.py) is pure Python/numpy and testable
without Qt; the thumbnail cache (thumbnails.py) and the tab UI (ui.py)
live beside it.
"""
from .ui import DuplicateDetectorTab

__all__ = ["DuplicateDetectorTab"]
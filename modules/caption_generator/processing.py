"""Qt shells over the pure engine (engine.run). All processing logic lives
in engine.py; these QThread classes only translate params to a plain-data
dict, bridge engine callbacks back to Qt signals, and keep the GUI-side
progress-bar print. Constructors and signals are unchanged for the UI."""

import threading

from PySide6.QtCore import QThread, Signal

from . import engine
from .utils import ProgressBar
from modules.logger import setup_logger

logger = setup_logger()


class _StopControl:
    """Cooperative-stop methods shared by the caption worker threads.

    Subclasses create ``self._stop_event`` in ``__init__``; the GUI calls
    ``request_stop()`` and the worker polls ``_should_stop()``.
    """

    def request_stop(self):
        self._stop_event.set()

    def _should_stop(self):
        return self._stop_event.is_set()


class CaptionGeneratorThread(QThread, _StopControl):
    caption_generated = Signal(str, str)
    process_completed = Signal()
    error_occurred = Signal(str)
    stopped = Signal()

    def __init__(self, captioner, folder_path, include_rating=False,
                 remove_underscore=True, recursive=False,
                 undesired_tags=None, prefix_tags=None, append_tags=False,
                 thresh=0.35, general_threshold=0.35, character_threshold=0.35,
                 batch_size=1, worker_count=2):
        super().__init__()
        self.captioner = captioner
        self.folder_path = folder_path
        self.include_rating = include_rating
        self.remove_underscore = remove_underscore
        self.recursive = recursive
        self.undesired_tags = undesired_tags or set()
        self.prefix_tags = prefix_tags or []
        self.append_tags = append_tags
        self.thresh = thresh
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.batch_size = batch_size
        self.worker_count = worker_count
        self._stop_event = threading.Event()

    def run(self):
        try:
            state = {"bar": None}

            def progress(current, total, message=""):
                # The bar is printed GUI-side (print_progress_bar convention);
                # the engine only reports counts.
                if state["bar"] is None:
                    state["bar"] = ProgressBar(total, prefix='Processing: ')
                state["bar"].update(current)

            summary = engine.run(
                {
                    "mode": engine.MODE_TAGGER,
                    "folder": self.folder_path,
                    "recursive": self.recursive,
                    "captioner": self.captioner,
                    "include_rating": self.include_rating,
                    "remove_underscore": self.remove_underscore,
                    "undesired_tags": self.undesired_tags,
                    "prefix_tags": self.prefix_tags,
                    "append_tags": self.append_tags,
                    "thresh": self.thresh,
                    "general_threshold": self.general_threshold,
                    "character_threshold": self.character_threshold,
                    "batch_size": self.batch_size,
                    "worker_count": self.worker_count,
                    "on_caption": lambda path, caption: self.caption_generated.emit(path, caption),
                    "on_error": lambda message: self.error_occurred.emit(message),
                },
                progress_cb=progress,
                stop_check=self._should_stop,
            )
            if summary.get("stopped"):
                self.stopped.emit()
            else:
                self.process_completed.emit()
        except Exception as e:
            self.error_occurred.emit(f"Process error: {str(e)}")


class NaturalLanguageCaptionThread(QThread, _StopControl):
    caption_generated = Signal(str, str)
    process_completed = Signal()
    error_occurred = Signal(str)
    stopped = Signal()

    def __init__(self, captioner, folder_path, recursive=False, caption_prefix=""):
        super().__init__()
        self.captioner = captioner
        self.folder_path = folder_path
        self.recursive = recursive
        self.caption_prefix = caption_prefix
        self._stop_event = threading.Event()

    def request_stop(self):
        super().request_stop()
        # Abort an in-flight streaming request right away, not at the next
        # image boundary (LocalLLMCaptioner closes the HTTP connection).
        try:
            self.captioner.stop_current_request()
        except Exception:
            pass

    def run(self):
        try:
            summary = engine.run(
                {
                    "mode": engine.MODE_NATURAL_LANGUAGE,
                    "folder": self.folder_path,
                    "recursive": self.recursive,
                    "captioner": self.captioner,
                    "caption_prefix": self.caption_prefix,
                    "on_caption": lambda path, caption: self.caption_generated.emit(path, caption),
                    "on_error": lambda message: self.error_occurred.emit(message),
                },
                progress_cb=None,
                stop_check=self._should_stop,
            )
            if summary.get("stopped"):
                self.stopped.emit()
            else:
                self.process_completed.emit()
        except Exception as e:
            self.error_occurred.emit(f"Process error: {str(e)}")

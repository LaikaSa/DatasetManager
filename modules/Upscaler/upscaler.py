"""Upscaler tab (RealESRGAN + SeedVR2).

Model definitions and the upscaling workers live in realesrgan.py, the
input selector in input_widget.py; this module is the tab UI only.
"""
import os

import torch
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                              QPushButton, QLabel, QCheckBox,
                              QComboBox, QSpinBox, QDoubleSpinBox,
                              QScrollArea)
from PySide6.QtCore import Qt
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from modules import settings
from modules.logger import setup_logger
from modules.Upscaler.realesrgan import UpscaleWorker, ModelDownloadWorker
from modules.Upscaler.input_widget import InputWidget
from modules.Upscaler.seedvr2_upscaler import (
    SeedVR2UpscaleWorker, SeedVR2DownloadWorker, are_seedvr2_models_downloaded
)

logger = setup_logger()


class UpscalerTab(QWidget):
    def __init__(self):
        super().__init__()
        logger.info("Initializing Upscaler Tab")
        self.worker = None
        # Cached RealESRGAN model so repeated runs skip torch.load + device
        # transfer. Keyed by (model_path, device).
        self._cached_model = None
        self._cached_model_key = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Single unified input widget
        self.input_widget = InputWidget()
        self.input_widget.input_changed.connect(self.check_input)
        layout.addWidget(self.input_widget)

        # Scale + resolution controls
        controls_layout = QHBoxLayout()

        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale factor:"))
        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 16.0)
        self.scale_spin.setValue(4.0)
        self.scale_spin.setSingleStep(0.1)
        self.scale_spin.setDecimals(1)
        self.scale_spin.setSuffix('x')
        scale_layout.addWidget(self.scale_spin)

        resolution_layout = QHBoxLayout()
        self.resolution_cb = QCheckBox("Only upscale images smaller than:")
        self.resolution_spin = QSpinBox()
        self.resolution_spin.setRange(1, 10000)
        self.resolution_spin.setValue(1536)
        self.resolution_spin.setSuffix(' px')
        self.resolution_spin.setEnabled(False)
        self.resolution_cb.stateChanged.connect(self.toggle_resolution_filter)
        self.resolution_spin.valueChanged.connect(self._on_resolution_changed)
        resolution_layout.addWidget(self.resolution_cb)
        resolution_layout.addWidget(self.resolution_spin)

        controls_layout.addLayout(scale_layout)
        controls_layout.addSpacing(20)
        controls_layout.addLayout(resolution_layout)
        controls_layout.addStretch()
        layout.addLayout(controls_layout)

        # Model row (choose between the default RealESRGAN and SeedVR2)
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItem("Default (RealESRGAN anime 6B)", "realesrgan")
        self.model_combo.addItem("SeedVR2 (3B fp16)", "seedvr2")
        # NOTE: connected after seedvr2_options exists (addItem emits indexChanged)
        model_layout.addWidget(self.model_combo)
        self.download_btn = QPushButton("Download Model")
        self.download_btn.clicked.connect(self.download_model)
        model_layout.addWidget(self.download_btn)
        self.model_status = QLabel("Model not downloaded")
        model_layout.addWidget(self.model_status)
        model_layout.addStretch()
        layout.addLayout(model_layout)

        # SeedVR2-only options (hidden when the default model is selected)
        self.seedvr2_options = QWidget()
        seedvr2_layout = QHBoxLayout(self.seedvr2_options)
        seedvr2_layout.setContentsMargins(0, 0, 0, 0)
        seedvr2_layout.addWidget(QLabel("Seed:"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(42)
        seedvr2_layout.addWidget(self.seed_spin)

        seedvr2_layout.addSpacing(20)
        seedvr2_layout.addWidget(QLabel("Color correction:"))
        self.color_combo = QComboBox()
        # LAB is the default: full perceptual color matching with detail preservation
        # (wavelet reconstruction + LAB histogram transfer).
        self.color_combo.addItem("LAB (perceptual, recommended)", "lab")
        self.color_combo.addItem("Wavelet", "wavelet")
        self.color_combo.addItem("Wavelet adaptive", "wavelet_adaptive")
        self.color_combo.addItem("HSV", "hsv")
        self.color_combo.addItem("AdaIN", "adain")
        self.color_combo.addItem("None", "none")
        seedvr2_layout.addWidget(self.color_combo)

        seedvr2_layout.addSpacing(20)
        self.tile_cb = QCheckBox("Tile VAE (low VRAM)")
        self.tile_cb.setChecked(False)
        seedvr2_layout.addWidget(self.tile_cb)
        seedvr2_layout.addStretch()
        self.seedvr2_options.setVisible(False)
        layout.addWidget(self.seedvr2_options)

        # Now safe to react to model changes (all widgets exist)
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)

        # Action buttons
        button_layout = QHBoxLayout()
        self.start_btn = QPushButton("Start Upscaling")
        self.stop_btn = QPushButton("Stop")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start_upscale)
        self.stop_btn.clicked.connect(self.stop_upscale)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)
        layout.addLayout(button_layout)

        # File list (below the controls, above the status log)
        layout.addWidget(self.input_widget.list_panel)

        # Status area
        self.status_area = QScrollArea()
        self.status_area.setWidgetResizable(True)
        self.status_text = QLabel()
        self.status_text.setAlignment(Qt.AlignTop)
        self.status_text.setWordWrap(True)
        self.status_area.setWidget(self.status_text)
        self.status_area.setMinimumHeight(200)
        layout.addWidget(self.status_area)

        # Tab-level drop target: forward drops to the input widget
        self.setAcceptDrops(True)

        self.setLayout(layout)

        self.check_model()

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        self.input_widget.handle_drop(event)

    def _on_resolution_changed(self, value):
        """Push the new resolution limit into the input widget whenever the
        spin changes (whether or not the filter checkbox is on)."""
        self.input_widget.set_resolution_filter(
            self.resolution_cb.isChecked(), value
        )

    def toggle_resolution_filter(self, state):
        auto = bool(state)
        self.resolution_spin.setEnabled(auto)
        # In auto mode each image gets its own scale factor, so the manual
        # spin box is greyed out.
        self.scale_spin.setEnabled(not auto)
        self.input_widget.set_resolution_filter(auto, self.resolution_spin.value())

    def selected_model(self):
        return self.model_combo.currentData()

    def on_model_changed(self, *_args):
        """Refresh the model row + SeedVR2 options whenever the model changes."""
        self.seedvr2_options.setVisible(self.selected_model() == "seedvr2")
        self.refresh_model_row()

    def refresh_model_row(self):
        """Recompute the ready-state of every model and update the UI."""
        self.model_ready = {}
        self.model_ready["realesrgan"] = os.path.exists(self.model_path)
        self.model_ready["seedvr2"] = are_seedvr2_models_downloaded()

        ready = self.model_ready.get(self.selected_model(), False)
        self.model_status.setText("Model ready" if ready else "Model not downloaded")
        self.download_btn.setEnabled(not ready and not self._downloading)
        self.check_input()

    def check_model(self):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(root_dir, "models")
        self.model_path = os.path.join(model_dir, "RealESRGAN_x4plus_anime_6B.pth")
        self.model_ready = {}
        self._downloading = False
        self.refresh_model_row()

    def check_input(self):
        ready = self.model_ready.get(self.selected_model(), False)
        self.start_btn.setEnabled(ready and len(self.input_widget.get_input_paths()) > 0)

    def download_model(self):
        model_type = self.selected_model()
        if model_type == "seedvr2":
            self._downloading = True
            self.download_btn.setEnabled(False)
            self.model_status.setText("Downloading SeedVR2 (~7GB)...")
            self.seedvr2_download = SeedVR2DownloadWorker()
            self.seedvr2_download.status.connect(self.model_status.setText)
            self.seedvr2_download.finished_ok.connect(self._seedvr2_download_done)
            self.seedvr2_download.start()
            return

        # Default RealESRGAN model (small, single file) - downloaded in a
        # worker thread so the GUI stays responsive during the transfer.
        self._downloading = True
        self.download_btn.setEnabled(False)
        self.model_status.setText("Downloading model...")
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.download_worker = ModelDownloadWorker(url, self.model_path)
        self.download_worker.status.connect(self.model_status.setText)
        self.download_worker.finished_ok.connect(self._realesrgan_download_done)
        self.download_worker.start()

    def _realesrgan_download_done(self, ok):
        self._downloading = False
        self.refresh_model_row()
        if not ok:
            self.model_status.setText("Download failed")
            self.download_btn.setEnabled(True)

    def _seedvr2_download_done(self, ok):
        self._downloading = False
        self.refresh_model_row()
        if not ok:
            self.model_status.setText("Download failed")
            self.download_btn.setEnabled(True)

    def start_upscale(self):
        if self.worker is not None and self.worker.isRunning():
            return
        input_paths = self.input_widget.get_input_paths()
        if not input_paths:
            self.update_status("No input files selected")
            return

        device = settings.to_torch_device(settings.get_selected_device_id())
        # Auto mode: let each worker pick the smallest scale factor that gets
        # the image's longest side to at least the chosen minimum size.
        min_size = self.resolution_spin.value() if self.resolution_cb.isChecked() else 0

        if self.selected_model() == "seedvr2":
            self.worker = SeedVR2UpscaleWorker(
                input_paths,
                self.scale_spin.value(),
                device=device,
                seed=self.seed_spin.value(),
                color_correction=self.color_combo.currentData(),
                tile_vae=self.tile_cb.isChecked(),
                min_size=min_size,
            )
            # SeedVR2 has no model_loaded signal to defer the start -
            # connect + launch immediately.
            self._activate_worker()
        else:
            # Reuse the cached model when model + device is unchanged
            cache_key = (self.model_path, device)
            if self._cached_model_key != cache_key:
                self._clear_cached_model()
            use_cached = self._cached_model_key == cache_key
            self.worker = UpscaleWorker(
                input_paths, self.model_path, self.scale_spin.value(),
                device=device, model=self._cached_model,
                owns_model=not use_cached, min_size=min_size,
            )
            self.worker.model_loaded.connect(self._on_model_loaded)

    def _activate_worker(self):
        """Connect the worker's signals, lock the UI and start the thread."""
        self.worker.status.connect(self.update_status)
        self.worker.finished.connect(self.upscale_finished)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.input_widget.set_editing_enabled(False)
        self.scale_spin.setEnabled(False)
        self.resolution_cb.setEnabled(False)
        self.resolution_spin.setEnabled(False)
        self.seedvr2_options.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.status_text.setText("")
        self.worker.start()

    def _clear_cached_model(self):
        if self._cached_model is not None:
            del self._cached_model
            self._cached_model = None
            self._cached_model_key = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _on_model_loaded(self, model):
        """Cache a freshly loaded model for the next run."""
        device = settings.to_torch_device(settings.get_selected_device_id())
        self._cached_model = model
        self._cached_model_key = (self.model_path, device)

        self._activate_worker()

    def stop_upscale(self):
        """Ask the worker to stop; its own finished signal runs
        upscale_finished() (non-blocking - no wait() on the GUI thread)."""
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.clear_gpu_memory()

    def upscale_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.input_widget.set_editing_enabled(True)
        self.scale_spin.setEnabled(not self.resolution_cb.isChecked())
        self.resolution_cb.setEnabled(True)
        self.seedvr2_options.setEnabled(True)
        self.model_combo.setEnabled(True)
        if self.resolution_cb.isChecked():
            self.resolution_spin.setEnabled(True)
        self.check_input()

    def update_status(self, text):
        if "Finished" in text:
            formatted = f"<p style='color:green;font-weight:bold;'>{text}</p>"
        elif "Error" in text:
            formatted = f"<p style='color:red;font-weight:bold;'>{text}</p>"
        else:
            formatted = f"<p>{text}</p>"
        current = self.status_text.text()
        self.status_text.setText(current + formatted if current else formatted)
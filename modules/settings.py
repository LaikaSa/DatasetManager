"""
Global app settings (LM Studio style).

Today this holds one setting - which compute device models are loaded on
("GPU" in the settings menu) - but it is the single place to add more
settings later; the cog button's menu in main.py is where they are exposed.

The device selection is stored as an integer CUDA ordinal:
    -1  -> CPU
    >=0 -> that CUDA device (GPU 0, GPU 1, ...)

CUDA ordinals map to the same physical GPUs for both ONNX Runtime
(`device_id` provider option) and PyTorch (`cuda:<n>`), so one selection
drives every model loader in the app. The choice is persisted in the app's
QSettings store (same org/app as main.py) and validated against the
hardware actually present when read.
"""

from PySide6.QtCore import QSettings, Qt, QRectF, QPointF
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap

# Must match the QSettings store used in main.py
SETTINGS_ORG = "DatasetManager"
SETTINGS_APP = "ImageProcessingTool"
SETTINGS_KEY_GPU_DEVICE = "gpu_device"

CPU_ID = -1


def _settings():
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

_device_cache = None


def list_devices(refresh=False):
    """Ordered list of {'id': int, 'label': str} devices.

    CPU first, then one entry per CUDA GPU (with its name when known).
    Never raises: on a machine without torch/CUDA it simply returns [CPU].

    The first call imports torch and queries CUDA, so the result is cached
    for the process lifetime (hardware doesn't change while the app runs);
    pass refresh=True to force a re-query.
    """
    global _device_cache
    if _device_cache is not None and not refresh:
        return _device_cache
    devices = [{"id": CPU_ID, "label": "CPU"}]
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = "GPU"
                devices.append({"id": i, "label": f"GPU {i}: {name}"})
    except Exception:
        pass
    _device_cache = devices
    return devices


def default_device_id():
    """GPU 0 when a GPU is present (matches the old auto behavior), else CPU."""
    devices = list_devices()
    return devices[1]["id"] if len(devices) > 1 else CPU_ID


def get_selected_device_id():
    """The persisted selection, validated against current hardware."""
    raw = _settings().value(SETTINGS_KEY_GPU_DEVICE, None)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default_device_id()
    if value in [d["id"] for d in list_devices()]:
        return value
    return default_device_id()


def set_selected_device_id(device_id):
    """Persist the selection (call this when the user picks a device)."""
    _settings().setValue(SETTINGS_KEY_GPU_DEVICE, int(device_id))


def device_label(device_id):
    """Human-readable label for a device id (falls back gracefully)."""
    for d in list_devices():
        if d["id"] == device_id:
            return d["label"]
    return "CPU" if device_id == CPU_ID else f"GPU {device_id}"


def to_torch_device(device_id):
    """'cpu' or 'cuda:<n>' for PyTorch loaders."""
    if device_id is None or device_id < 0:
        return "cpu"
    return f"cuda:{device_id}"


# ---------------------------------------------------------------------------
# Settings button icon
# ---------------------------------------------------------------------------

def create_settings_icon(logical_size=20):
    """Paint a small cog icon (2x supersampled; no asset files needed)."""
    scale = 2
    pixmap = QPixmap(logical_size * scale, logical_size * scale)
    pixmap.fill(Qt.transparent)
    pixmap.setDevicePixelRatio(scale)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    color = QColor(130, 130, 138)
    center = QPointF(logical_size / 2, logical_size / 2)
    body_r = logical_size * 0.36
    tooth_len = logical_size * 0.16
    tooth_w = logical_size * 0.18

    painter.setPen(Qt.NoPen)
    painter.setBrush(color)

    # Eight teeth around the body
    for i in range(8):
        painter.save()
        painter.translate(center)
        painter.rotate(i * 45.0)
        painter.drawRect(QRectF(-tooth_w / 2, -(body_r + tooth_len), tooth_w, tooth_len))
        painter.restore()

    # Gear body
    painter.drawEllipse(center, body_r, body_r)

    # Center hole (punched through to transparent)
    painter.setCompositionMode(QPainter.CompositionMode_Clear)
    painter.drawEllipse(center, body_r * 0.42, body_r * 0.42)
    painter.end()

    return QIcon(pixmap)

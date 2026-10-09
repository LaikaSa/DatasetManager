"""Shared non-Qt utilities used across tabs.

Keep this module light: it must stay importable from worker threads and
multiprocessing children without pulling in Qt or torch.
"""
import os
import sys

# Default image extensions most tabs accept. Tools that support more
# formats (the format converter, the ICC fixer) define their own sets.
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp')


def open_in_folder(path):
    """Open the folder containing path in Explorer with the file
    highlighted and scrolled into view (Windows shell API)."""
    import ctypes
    normalized = os.path.normpath(path)
    shell32 = ctypes.windll.shell32
    # Must set restype to c_void_p or the 64-bit pointer gets truncated to 32-bit
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


def print_progress_bar(current, total, prefix='Progress:', length=50):
    """Overwrite one console line with a progress bar.

    Classic \\r overwrite (no ANSI cursor movement) so it is safe even when
    nothing was printed on the line before the first call.
    """
    if total <= 0:
        return
    filled_length = int(length * current / total)
    bar = '=' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} [{bar}] {current}/{total}')
    sys.stdout.flush()

"""Perceptual-hash + color-histogram duplicate detection.

Pure Python/numpy - no Qt - so the algorithm can be unit-tested without
a GUI.
"""
import os

import cv2
import imagehash
import numpy as np
from PIL import Image

from modules.utils import IMAGE_EXTENSIONS


def collect_image_files(folder_path, extensions=IMAGE_EXTENSIONS, recursive=False):
    """Image files in folder_path; recursive includes subfolders."""
    files = []
    if recursive:
        for root, _, names in os.walk(folder_path):
            for file in names:
                if file.lower().endswith(extensions):
                    files.append(os.path.join(root, file))
    else:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path) and file.lower().endswith(extensions):
                files.append(file_path)
    return files


def extract_features(image_path, use_hash, use_hist):
    """Compute the detection features for one image.

    Returns (features, size): features is a dict with 'hash' and/or
    'hist' entries, size is (width, height) or None when it could not be
    read. Raises on unreadable images.
    """
    features = {}
    size = None
    if use_hash:
        with Image.open(image_path) as img:
            features['hash'] = imagehash.average_hash(img)
            size = img.size

    if use_hist:
        img_cv = cv2.imread(image_path)
        if img_cv is not None:
            hist = cv2.calcHist([img_cv], [0, 1, 2], None, [8, 8, 8],
                                [0, 256, 0, 256, 0, 256])
            features['hist'] = cv2.normalize(hist, hist).flatten()
            if size is None:
                h, w = img_cv.shape[:2]
                size = (w, h)

    return features, size


def group_images(image_features, sizes, use_hash, use_hist,
                 hash_threshold, hist_threshold,
                 progress_cb=None, stop_check=None):
    """Group similar images (vectorized - one matmul instead of an O(n^2)
    Python loop of cv2.compareHist / hash subtractions).

    image_features: path -> {'hash': ..., 'hist': ...}
    sizes: path -> (width, height)
    Returns a list of group dicts:
        {'images': [...], 'similarity': float,
         'method': 'Hash' | 'Histogram', 'sizes': {path: (w, h)}}
    progress_cb: optional callable(done, total), called periodically
    during the grouping pass.
    stop_check: optional zero-arg callable; grouping stops early when it
    returns True.
    """
    groups = []
    paths = list(image_features.keys())
    n = len(paths)

    if n < 2:
        return groups

    hash_sim = None
    hash_valid = None
    corr = None
    hist_valid = None

    if use_hash:
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

    if use_hist:
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

    for i, image_path in enumerate(paths):
        if i in processed:
            continue

        qualifies = np.zeros(n, dtype=bool)
        val_h = np.full(n, -np.inf)
        val_s = np.full(n, -np.inf)

        if hash_sim is not None and hash_valid[i]:
            mask = hash_valid & (hash_sim[i] >= hash_threshold)
            qualifies |= mask & ~processed_mask & (all_idx != i)
            val_h = np.where(mask & ~processed_mask & (all_idx != i), hash_sim[i], -np.inf)

        if corr is not None and hist_valid[i]:
            mask = hist_valid & (corr[i] >= hist_threshold)
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
        if progress_cb is not None and processed_count % 50 == 0:
            progress_cb(processed_count, n)
        if stop_check is not None and stop_check():
            break

    return groups
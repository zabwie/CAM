"""HSV histogram appearance descriptor for vehicle re-identification."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

import cv2
import numpy as np


def appearance_descriptor(
    image: np.ndarray | None,
    bbox: np.ndarray,
) -> np.ndarray | None:
    """Compute an HSV histogram descriptor for a cropped vehicle region."""
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    bw, bh = x2 - x1, y2 - y1
    x1 += 0.10 * bw
    x2 -= 0.10 * bw
    y1 += 0.10 * bh
    y2 -= 0.08 * bh
    ix1, iy1 = max(0, int(round(x1))), max(0, int(round(y1)))
    ix2, iy2 = min(w, int(round(x2))), min(h, int(round(y2)))
    if ix2 - ix1 < 10 or iy2 - iy1 < 8 or (ix2 - ix1) * (iy2 - iy1) < 180:
        return None
    crop = image[iy1:iy2, ix1:ix2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 18, 24]), np.array([179, 255, 255]))
    hist = cv2.calcHist([hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256]).astype(np.float32)
    if float(hist.sum()) <= 1e-6:
        hist = cv2.calcHist([hsv], [0, 1], None, [18, 8], [0, 180, 0, 256]).astype(np.float32)
    return normalise_hist(hist.reshape(-1))


def normalise_hist(hist: np.ndarray) -> np.ndarray:
    """Normalise a histogram to unit sum."""
    arr = np.asarray(hist, dtype=np.float32).reshape(-1)
    total = float(arr.sum())
    if total <= 1e-9:
        return arr
    return arr / total


def appearance_similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Bhattacharyya-based appearance similarity (0–1)."""
    if a is None or b is None:
        return 0.55
    distance = float(cv2.compareHist(
        np.asarray(a, dtype=np.float32),
        np.asarray(b, dtype=np.float32),
        cv2.HISTCMP_BHATTACHARYYA,
    ))
    return float(np.clip(1.0 - distance, 0.0, 1.0))

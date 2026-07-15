"""Bounding-box and coordinate utility functions."""

from __future__ import annotations

from collections import deque

import numpy as np


def robust_velocity(
    history: deque[tuple[int, float, float, float, float]],
) -> np.ndarray:
    """Median per-frame velocity from a (frame, cx, cy, w, h) history deque."""
    if len(history) < 3:
        return np.zeros(2, dtype=np.float64)
    rows = list(history)[-10:]
    velocities = []
    for a, b in zip(rows[:-1], rows[1:]):
        df = b[0] - a[0]
        if df <= 0:
            continue
        velocities.append([(b[1] - a[1]) / df, (b[2] - a[2]) / df])
    if not velocities:
        return np.zeros(2, dtype=np.float64)
    return np.median(np.asarray(velocities, dtype=np.float64), axis=0)


def box_center_size(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, size) from a (x1, y1, x2, y2) bbox."""
    box = np.asarray(bbox, dtype=np.float64)
    center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
    size = np.array([max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])])
    return center, size


def center_size_box(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Reconstruct (x1, y1, x2, y2) from center + size."""
    half = size * 0.5
    return np.array([
        center[0] - half[0], center[1] - half[1],
        center[0] + half[0], center[1] + half[1],
    ], dtype=np.float64)


def size_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Area ratio (larger / smaller) of two (w, h) size arrays."""
    area_a = max(1.0, float(a[0] * a[1]))
    area_b = max(1.0, float(b[0] * b[1]))
    return max(area_a, area_b) / min(area_a, area_b)

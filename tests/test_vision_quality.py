from __future__ import annotations

import cv2
import numpy as np

from traffic_intel.core.appearance import appearance_descriptor
from traffic_intel.vision_quality import VisionQualityMonitor


def _checkerboard(low: int, high: int, *, color: tuple[int, int, int] | None = None) -> np.ndarray:
    grid = (np.indices((120, 160)).sum(axis=0) % 2).astype(np.uint8)
    gray = np.where(grid == 0, low, high).astype(np.uint8)
    if color is None:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    frame = np.empty((120, 160, 3), dtype=np.uint8)
    frame[:] = color
    frame[grid == 1] = np.clip(np.asarray(color) + 35, 0, 255)
    return frame


def test_usable_monochrome_feed_is_tagged_as_ir_night() -> None:
    sample = VisionQualityMonitor().update(_checkerboard(70, 140))
    assert sample.state == "NIGHT_IR"
    assert sample.is_monochrome


def test_dark_feed_is_tagged_low_light() -> None:
    sample = VisionQualityMonitor().update(_checkerboard(2, 20))
    assert sample.state == "LOW_LIGHT"


def test_identity_appearance_is_disabled_for_monochrome_ir() -> None:
    image = _checkerboard(70, 140)
    descriptor = appearance_descriptor(
        image,
        np.array([10, 10, 150, 110], dtype=np.float32),
    )
    assert descriptor is None


def test_identity_appearance_remains_available_for_colored_vehicle_crop() -> None:
    image = _checkerboard(80, 120, color=(30, 60, 180))
    descriptor = appearance_descriptor(
        image,
        np.array([10, 10, 150, 110], dtype=np.float32),
    )
    assert descriptor is not None

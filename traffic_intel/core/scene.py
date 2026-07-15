"""Global scene/source discontinuity detection."""

from __future__ import annotations

import cv2
import numpy as np

from ..config import SceneChangeConfig


class SceneChangeDetector:
    def __init__(self, config: SceneChangeConfig | None = None) -> None:
        self.config = config or SceneChangeConfig()
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def update(self, frame: np.ndarray) -> bool:
        cfg = self.config
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (cfg.width, cfg.height), interpolation=cv2.INTER_AREA)
        previous = self._previous
        self._previous = small
        if previous is None or previous.shape != small.shape:
            return False
        diff = cv2.absdiff(previous, small)
        mean_diff = float(diff.mean())
        changed_fraction = float(np.mean(diff > cfg.pixel_delta_threshold))
        return bool(
            mean_diff > cfg.mean_delta_threshold
            and changed_fraction > cfg.changed_fraction_threshold
        )

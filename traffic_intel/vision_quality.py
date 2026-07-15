"""Low-cost frame quality telemetry for pilot deployments.

This module does not alter detector output. It tags each frame with observable
conditions so day, IR-night, glare, and unusable low-light performance can be
reported separately instead of silently mixing unlike operating conditions.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VisionQualityConfig:
    analysis_width: int = 320
    low_light_median: float = 35.0
    usable_ir_median: float = 52.0
    daylight_median: float = 82.0
    dark_pixel_threshold: int = 28
    dark_fraction_degraded: float = 0.70
    highlight_threshold: int = 245
    glare_fraction_threshold: float = 0.055
    low_chroma_saturation: float = 10.0
    low_chroma_channel_delta: float = 4.0
    blur_laplacian_threshold: float = 22.0


@dataclass(frozen=True, slots=True)
class VisionQualitySample:
    state: str
    median_brightness: float
    dark_fraction: float
    clipped_highlight_fraction: float
    median_saturation: float
    channel_delta: float
    blur_score: float

    @property
    def is_monochrome(self) -> bool:
        return self.median_saturation < 10.0 and self.channel_delta < 4.0


class VisionQualityMonitor:
    """Classify visible frame conditions using explainable image statistics."""

    STATES = {"DAY", "NIGHT_IR", "LOW_LIGHT", "GLARE", "DEGRADED"}

    def __init__(self, config: VisionQualityConfig | None = None) -> None:
        self.config = config or VisionQualityConfig()
        self.last_sample: VisionQualitySample | None = None

    def update(self, frame: np.ndarray) -> VisionQualitySample:
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("frame must be a BGR image")

        h, w = frame.shape[:2]
        target_w = min(self.config.analysis_width, w)
        target_h = max(1, int(round(h * target_w / max(w, 1))))
        small = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        median_brightness = float(np.median(gray))
        dark_fraction = float(np.mean(gray <= self.config.dark_pixel_threshold))
        clipped = float(np.mean(gray >= self.config.highlight_threshold))
        median_saturation = float(np.median(hsv[..., 1]))

        channels = small.astype(np.float32)
        channel_delta = float(
            np.mean(np.max(channels, axis=2) - np.min(channels, axis=2))
        )
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        low_chroma = (
            median_saturation <= self.config.low_chroma_saturation
            and channel_delta <= self.config.low_chroma_channel_delta
        )
        if median_brightness < self.config.low_light_median or dark_fraction >= self.config.dark_fraction_degraded:
            state = "LOW_LIGHT"
        elif clipped >= self.config.glare_fraction_threshold:
            state = "GLARE"
        elif low_chroma and median_brightness >= self.config.usable_ir_median:
            state = "NIGHT_IR"
        elif blur_score < self.config.blur_laplacian_threshold:
            state = "DEGRADED"
        elif median_brightness >= self.config.daylight_median:
            state = "DAY"
        else:
            # Dim but still usable colour footage is retained as degraded so it
            # can be reported independently from both daylight and true IR.
            state = "DEGRADED"

        sample = VisionQualitySample(
            state=state,
            median_brightness=median_brightness,
            dark_fraction=dark_fraction,
            clipped_highlight_fraction=clipped,
            median_saturation=median_saturation,
            channel_delta=channel_delta,
            blur_score=blur_score,
        )
        self.last_sample = sample
        return sample


__all__ = [
    "VisionQualityConfig",
    "VisionQualityMonitor",
    "VisionQualitySample",
]

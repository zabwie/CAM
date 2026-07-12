"""Core domain records shared across perception and incident analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

BBox = Tuple[int, int, int, int]


@dataclass(slots=True)
class Detection:
    """A trusted canonical vehicle observation emitted by ``TrafficEngine``.

    ``track_id`` is the stable canonical vehicle ID used by speed and incident
    analyzers.  ``raw_track_id`` exposes the underlying tracker handle for
    diagnostics only.  A raw tracker ID may change after an occlusion while the
    canonical ID remains stable.

    ``bbox`` remains the raw detector/tracker geometry for incident analysis.
    ``filtered_bbox`` is a lightly filtered presentation/road-anchor box used to
    reduce visible jitter without hiding genuine impact-time innovations.
    """

    frame: int
    track_id: int
    class_name: str
    confidence: float
    bbox: BBox
    speed: float = 0.0
    speed_valid: bool = False
    invalid_reason: str = "VALID"
    measurement_confidence: float = 0.0
    cal_confidence: float = 0.0
    traj_confidence: float = 0.0
    vis_confidence: float = 0.0
    zone_confidence: float = 0.0
    capture_timestamp: float = 0.0
    track_quality: float = 1.0
    track_confirmed: bool = True
    raw_track_id: int | None = None
    identity_generation: int = 1
    identity_confidence: float = 1.0
    identity_lifecycle: str = "CONTINUING"
    filtered_bbox: BBox | None = None

    @property
    def speed_mph(self) -> float:
        """Compatibility alias used by older UI code."""
        return self.speed

    @property
    def display_bbox(self) -> BBox:
        """Stable box for annotation while preserving raw ``bbox`` analytics."""
        return self.filtered_bbox or self.bbox

    @property
    def tracker_id(self) -> int:
        """Underlying tracker ID, falling back to canonical ID for old records."""
        return self.track_id if self.raw_track_id is None else self.raw_track_id


__all__ = ["BBox", "Detection"]

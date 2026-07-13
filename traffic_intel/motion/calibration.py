"""Camera calibration and image-to-road-plane projection."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

PathLike = Union[str, Path]


@dataclass
class Calibration:
    """ROI filtering plus planar homography calibration.

    The point smoother operates on both bottom-centre coordinates.  Smoothing
    only the vertical coordinate is insufficient for oblique cameras where
    lateral box jitter also maps to substantial world-space error.
    """

    roi_polygon: Optional[list] = None
    image_points: Optional[list] = None
    world_points: Optional[list] = None
    homography_matrix: Optional[list] = None
    calibration_quality: Optional[dict] = None
    smoothing_window: int = 5
    _anchor_ring: dict[int, deque[tuple[float, float]]] = field(
        default_factory=lambda: defaultdict(deque), init=False, repr=False
    )

    def __post_init__(self) -> None:
        self.smoothing_window = max(1, int(self.smoothing_window))
        if self.calibration_quality is None and self.homography_matrix:
            self.compute_quality()

    def reset_track_state(self) -> None:
        self._anchor_ring.clear()

    @property
    def H(self) -> Optional[np.ndarray]:
        if self.homography_matrix:
            return np.asarray(self.homography_matrix, dtype=np.float32)
        return None

    @property
    def quality_grade(self) -> str:
        if self.calibration_quality:
            return str(self.calibration_quality.get("quality_grade", "UNKNOWN"))
        return "UNKNOWN"

    def compute_quality(self) -> dict:
        if not self.homography_matrix or not self.image_points or not self.world_points:
            self.calibration_quality = {}
            return {}

        H = np.asarray(self.homography_matrix, dtype=np.float32)
        img_pts = np.asarray(self.image_points, dtype=np.float32)
        world_pts = np.asarray(self.world_points, dtype=np.float32)
        if len(img_pts) != len(world_pts) or len(img_pts) < 4:
            self.calibration_quality = {}
            return {}

        projected = cv2.perspectiveTransform(img_pts.reshape(1, -1, 2), H)[0]
        errors = np.linalg.norm(projected - world_pts, axis=1)
        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))
        std_err = float(np.std(errors))
        med = float(np.median(errors))
        mad = float(np.median(np.abs(errors - med)))
        threshold = max(0.01, med + 3.0 * 1.4826 * mad)
        inlier_ratio = float(np.mean(errors <= threshold))

        if mean_err < 0.10:
            grade = "EXCELLENT"
        elif mean_err < 0.30:
            grade = "GOOD"
        elif mean_err < 0.50:
            grade = "FAIR"
        else:
            grade = "POOR"

        self.calibration_quality = {
            "point_count": int(len(img_pts)),
            "mean_reprojection_error_m": mean_err,
            "max_reprojection_error_m": max_err,
            "std_reprojection_error_m": std_err,
            "mean_reprojection_residual_m": mean_err,
            "max_reprojection_residual_m": max_err,
            "std_reprojection_residual_m": std_err,
            "inlier_ratio": inlier_ratio,
            "quality_grade": grade,
        }
        return self.calibration_quality

    def compute_homography(self) -> None:
        if not self.image_points or not self.world_points:
            return
        if len(self.image_points) < 4 or len(self.world_points) < 4:
            return
        image = np.asarray(self.image_points, dtype=np.float32)
        world = np.asarray(self.world_points, dtype=np.float32)
        H, _ = cv2.findHomography(image, world, cv2.RANSAC, 5.0)
        if H is not None:
            self.homography_matrix = H.tolist()
            self.compute_quality()

    def world_from_image(self, cx: float, cy: float) -> Optional[tuple[float, float]]:
        H = self.H
        if H is None:
            return None
        wx, wy = map(
            float,
            cv2.perspectiveTransform(
                np.array([[[cx, cy]]], dtype=np.float32), H
            )[0, 0],
        )
        if not self.world_point_is_calibrated(wx, wy):
            return None
        return wx, wy

    def smoothed_world_from_image(
        self, track_id: int, cx: float, cy: float
    ) -> Optional[tuple[float, float]]:
        ring = self._anchor_ring[track_id]
        ring.append((float(cx), float(cy)))
        while len(ring) > self.smoothing_window:
            ring.popleft()
        if len(ring) >= 3:
            arr = np.asarray(ring, dtype=np.float64)
            cx, cy = map(float, np.median(arr, axis=0))
        return self.world_from_image(cx, cy)

    def world_point_is_calibrated(self, x_m: float, y_m: float) -> bool:
        if not self.world_points:
            return False
        pts = np.asarray(self.world_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
            return False
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        lo = lo - np.array([0.50 * span[0], 0.20 * span[1]])
        hi = hi + np.array([0.50 * span[0], 0.20 * span[1]])
        return bool(lo[0] <= x_m <= hi[0] and lo[1] <= y_m <= hi[1])

    def save(self, path: PathLike) -> None:
        payload = {
            "roi_polygon": self.roi_polygon or [],
            "image_points": self.image_points or [],
            "world_points": self.world_points or [],
            "homography_matrix": self.homography_matrix or [],
            "calibration_quality": self.calibration_quality or {},
            "smoothing_window": self.smoothing_window,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: PathLike) -> "Calibration":
        data = json.loads(Path(path).read_text())
        return cls(
            roi_polygon=data.get("roi_polygon") or None,
            image_points=data.get("image_points") or None,
            world_points=data.get("world_points") or None,
            homography_matrix=data.get("homography_matrix") or None,
            calibration_quality=data.get("calibration_quality") or None,
            smoothing_window=int(data.get("smoothing_window", 5)),
        )

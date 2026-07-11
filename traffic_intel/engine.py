"""
Traffic Intelligence Engine
Core pipeline: YOLO detection -> ByteTrack tracking -> ROI filtering

Usage:
    engine = TrafficEngine()
    summary = engine.process_video("traffic.mp4", output_path="annotated.mp4")
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

try:
    from .speed import RobustSpeedEstimator
except ImportError:  # script execution: python traffic_intel/engine.py
    from speed import RobustSpeedEstimator


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    """Camera calibration: ROI filtering + homography speed estimation."""
    roi_polygon: Optional[list] = None       # [[x,y], ...] drivable area
    image_points: Optional[list] = None      # [[x,y], ...] pixel coords
    world_points: Optional[list] = None      # [[x,y], ...] meter coords
    homography_matrix: Optional[list] = None # 3x3 list from cv2.findHomography

    # Per-track ring buffers for sub-pixel cy estimation via running median.
    _cy_ring: Optional[dict] = None  # track_id → list of recent cy values
    _ring_size: int = 5

    def __post_init__(self):
        self._cy_ring = {}

    def world_from_image(self, cx: float, cy: float) -> Optional[tuple[float, float]]:
        return self._world_from_homography(cx, cy)

    def _world_from_homography(self, cx: float, cy: float) -> Optional[tuple[float, float]]:
        H = self.H
        if H is None:
            return None
        wx, wy = map(float, cv2.perspectiveTransform(
            np.array([[[cx, cy]]], dtype=np.float32), H)[0, 0])
        if not self.world_point_is_calibrated(wx, wy):
            return None
        return (wx, wy)

    def smoothed_world_from_image(
        self, track_id: int, cx: float, cy: float
    ) -> Optional[tuple[float, float]]:
        H = self.H
        if H is None:
            return None

        ring = self._cy_ring.setdefault(track_id, [])
        ring.append(cy)
        if len(ring) > self._ring_size:
            ring.pop(0)

        # Running median kills single-frame quantization jitter (1-2px) on cy
        # before the homography amplifies it to meters of depth noise.
        fitted_cy = float(np.median(ring)) if len(ring) >= 3 else cy

        wx, wy = map(float, cv2.perspectiveTransform(
            np.array([[[cx, fitted_cy]]], dtype=np.float32), H)[0, 0])
        if not self.world_point_is_calibrated(wx, wy):
            return None
        return (wx, wy)

    def compute_homography(self):
        """Compute H from image_points → world_points. Requires 4+ points."""
        if not self.image_points or not self.world_points:
            return
        if len(self.image_points) < 4 or len(self.world_points) < 4:
            return
        img = np.float32(self.image_points)
        wld = np.float32(self.world_points)
        # findHomography supports 4+ points with RANSAC
        H, _ = cv2.findHomography(img, wld, cv2.RANSAC, 5.0)
        if H is not None:
            self.homography_matrix = H.tolist()

    def save(self, path: str | Path):
        import json
        data = dict(
            roi_polygon=self.roi_polygon or [],
            image_points=self.image_points or [],
            world_points=self.world_points or [],
            homography_matrix=self.homography_matrix or [],
        )
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path):
        import json
        data = json.loads(Path(path).read_text())
        return cls(
            roi_polygon=data.get("roi_polygon"),
            image_points=data.get("image_points"),
            world_points=data.get("world_points"),
            homography_matrix=data.get("homography_matrix"),
        )

    @property
    def H(self) -> Optional[np.ndarray]:
        """Get homography matrix as numpy array, or None."""
        if self.homography_matrix:
            return np.array(self.homography_matrix, dtype=np.float32)
        return None

    def world_point_is_calibrated(self, x_m: float, y_m: float) -> bool:
        """Return whether a world point is near the measured calibration area.

        A homography can mathematically extrapolate across the entire image, but
        speed estimates far outside the measured road patch are not trustworthy.
        Keep a moderate margin for nearby lanes while rejecting distant roads and
        other perspective planes.
        """
        if not self.world_points:
            return False
        pts = np.asarray(self.world_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 4:
            return False
        lo = pts.min(axis=0)
        hi = pts.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        # Wider lateral allowance covers adjacent lanes; longitudinal allowance
        # stays tighter because depth extrapolation is where speed scale fails fast.
        lo = lo - np.array([0.50 * span[0], 0.20 * span[1]])
        hi = hi + np.array([0.50 * span[0], 0.20 * span[1]])
        return bool(lo[0] <= x_m <= hi[0] and lo[1] <= y_m <= hi[1])


@dataclass
class Detection:
    """A single detection event for one vehicle in one frame."""
    frame: int
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple          # x1, y1, x2, y2
    speed: float = 0.0   # mph
    speed_valid: bool = False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

VEHICLE_IDS = {2, 3, 5, 7}       # car, motorcycle, bus, truck (COCO)


class TrafficEngine:
    """Video-in -> detections + tracks + speed."""

    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        calibration: Optional[Calibration] = None,
        confidence: float = 0.25,
        fps: float = 30.0,
        imgsz: int = 1280,
    ):
        self.model = YOLO(model_path)
        self.conf_thresh = confidence
        self.imgsz = imgsz
        self.fps = fps
        self.calibration = calibration

        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence,
            lost_track_buffer=int(fps),
            minimum_matching_threshold=0.8,
            frame_rate=fps,
        )

        self.results: list[Detection] = []
        self.frame_count = 0

        # Robust trajectory-based speed estimation.
        self.speed_estimator = RobustSpeedEstimator(fps=fps)

    # ---- public API -------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one frame -> annotated copy."""
        self.frame_count += 1

        # Detect
        yolo_out = self.model(frame, verbose=False, conf=self.conf_thresh,
                              imgsz=self.imgsz)[0]
        dets = sv.Detections.from_ultralytics(yolo_out)

        # Keep only vehicles
        dets = dets[np.isin(dets.class_id, list(VEHICLE_IDS))]

        # ROI filter: remove detections outside the drivable polygon
        roi_pts = getattr(self.calibration, "roi_polygon", None) if self.calibration else None
        if roi_pts:
            roi_arr = np.array(roi_pts, dtype=np.int32)
            keep = []
            for i in range(len(dets)):
                x1, y1, x2, y2 = dets.xyxy[i].astype(int)
                cx, cy = (x1 + x2) // 2, y2   # bottom-centre
                if cv2.pointPolygonTest(roi_arr, (float(cx), float(cy)), False) >= 0:
                    keep.append(i)
            dets = dets[keep] if keep else dets[:0]

        # Track
        tracks = self.tracker.update_with_detections(dets)
        annotated = frame.copy()

        # Draw ROI polygon
        if roi_pts:
            cv2.polylines(annotated, [np.array(roi_pts, dtype=np.int32)],
                          True, (0, 255, 255), 2)

        cal = self.calibration

        for i in range(len(tracks)):
            tid = int(tracks.tracker_id[i])
            x1, y1, x2, y2 = map(int, tracks.xyxy[i])
            cls_id = int(tracks.class_id[i])
            conf = float(tracks.confidence[i])
            label = yolo_out.names.get(cls_id, "?")

            bbox_h = y2 - y1
            speed = None

            # Skip detections too small for reliable depth — a few pixels of
            # bbox-bottom jitter at distance maps to tens of meters of depth
            # noise regardless of the calibration or smoothing technique.
            if bbox_h < 12:
                pass
            elif cal is not None:
                cx, cy = (x1 + x2) / 2.0, float(y2)
                world_pt = cal.smoothed_world_from_image(tid, cx, cy)
                if world_pt is not None:
                    wx, wy = world_pt
                    speed = self.speed_estimator.update(tid, self.frame_count, wx, wy)

            speed_valid = speed is not None
            speed_value = float(speed) if speed_valid else 0.0
            self.results.append(Detection(
                frame=self.frame_count, track_id=tid,
                class_name=label, confidence=conf,
                bbox=(x1, y1, x2, y2), speed=speed_value,
                speed_valid=speed_valid,
            ))

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self._draw_vehicle_label(
                annotated, x1=x1, y1=y1, track_id=tid, class_name=label,
                speed_mph=speed_value if speed_valid else None,
            )

        self.speed_estimator.forget_stale(self.frame_count)
        return annotated

    @staticmethod
    def _draw_vehicle_label(
        frame: np.ndarray,
        *,
        x1: int,
        y1: int,
        track_id: int,
        class_name: str,
        speed_mph: Optional[float],
    ) -> None:
        """Draw one per-vehicle label: classification and that track's own MPH."""
        speed_text = f"{speed_mph:.0f} MPH" if speed_mph is not None else "-- MPH"
        text = f"{class_name} | {speed_text} | #{track_id}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.55
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

        frame_h, frame_w = frame.shape[:2]
        left = max(0, min(int(x1), max(0, frame_w - text_w - 10)))
        text_y = max(text_h + baseline + 6, int(y1) - 8)
        top = max(0, text_y - text_h - baseline - 6)
        right = min(frame_w - 1, left + text_w + 10)
        bottom = min(frame_h - 1, text_y + baseline + 3)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (left + 5, text_y), font, scale, (0, 255, 0), thickness,
            cv2.LINE_AA,
        )

    def process_video(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        max_frames: Optional[int] = None,
        progress_callback=None,
    ) -> dict:
        """Process entire video file. Returns summary dict."""
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        self.fps = fps

        # A TrafficEngine instance may process more than one video. Never leak
        # tracks, speed history, or detections across video boundaries.
        self.results.clear()
        self.frame_count = 0
        self.speed_estimator.reset()
        self.speed_estimator.set_fps(fps)
        if self.calibration is not None:
            self.calibration._cy_ring.clear()

        self.tracker = sv.ByteTrack(
            track_activation_threshold=self.conf_thresh,
            lost_track_buffer=int(fps),
            minimum_matching_threshold=0.8,
            frame_rate=fps,
        )

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = None
        if output_path:
            out = cv2.VideoWriter(str(output_path),
                                  cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            annotated = self.process_frame(frame)
            if out:
                out.write(annotated)
            count += 1
            if max_frames and count >= max_frames:
                break
            if progress_callback and count % 30 == 0:
                progress_callback(count, total)

        cap.release()
        if out:
            out.release()
        return self._summary()

    def results_csv(self, path: str | Path):
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "track_id", "class", "confidence",
                         "x1", "y1", "x2", "y2", "speed_mph", "speed_valid"])
            for d in self.results:
                w.writerow([
                    d.frame, d.track_id, d.class_name,
                    f"{d.confidence:.3f}", *d.bbox,
                    f"{d.speed:.1f}", int(d.speed_valid),
                ])

    # ---- summary ----------------------------------------------------------

    def _summary(self) -> dict:
        tracks = set(d.track_id for d in self.results)
        return dict(
            frames_processed=self.frame_count,
            unique_vehicles=len(tracks),
            total_detections=len(self.results),
        )

"""
Traffic Intelligence Engine
Core pipeline: YOLO detection → ByteTrack tracking → Homography speed estimation → Violation logic

Usage:
    from engine import TrafficEngine
    engine = TrafficEngine(calibration=Calibration.load("calib.json"))
    summary = engine.process_video("traffic.mp4", output_path="annotated.mp4")
    print(summary)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Calibration:
    """Homography mapping image pixels → ground-plane metres."""
    H: np.ndarray  # 3×3 homography (image → ground plane)
    width_m: float  # road width perpendicular to traffic
    length_m: float  # road length along traffic direction
    points_uv: list  # 4 image points [u,v] in TL/TR/BR/BL order

    def save(self, path: str | Path):
        data = dict(H=self.H.tolist(), width_m=self.width_m,
                    length_m=self.length_m, points_uv=self.points_uv)
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text())
        return cls(H=np.array(data["H"]), width_m=data["width_m"],
                   length_m=data["length_m"], points_uv=data["points_uv"])


@dataclass
class Detection:
    """A single detection event for one vehicle in one frame."""
    frame: int
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple  # x1, y1, x2, y2
    speed_kmh: Optional[float]
    is_violation: bool


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# Vehicle classes in COCO: car=2, motorcycle=3, bus=5, truck=7
VEHICLE_IDS = {2, 3, 5, 7}
# Simpler alias for speed checks
SPEED_BUFFER_FRAMES = 15  # ~0.5 s at 30 fps


class TrafficEngine:
    """Video-in → detections + tracks + speeds + violations."""

    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        calibration: Optional[Calibration] = None,
        speed_limit_kmh: float = 50,
        confidence: float = 0.25,
        fps: float = 30.0,
        imgsz: int = 1280,
    ):
        self.model = YOLO(model_path)
        self.cal = calibration
        self.speed_limit_kmh = speed_limit_kmh
        self.conf_thresh = confidence
        self.imgsz = imgsz

        self.fps = fps
        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence,
            lost_track_buffer=int(fps),
            minimum_matching_threshold=0.8,
            frame_rate=fps,
        )

        # per-track position buffer: track_id → [(frame, x_world, y_world)]
        self._pos_buffer: dict[int, list] = {}
        self.results: list[Detection] = []
        self.frame_count = 0

    # ---- public API -------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one frame → return annotated copy.

        All detections are appended to self.results.
        """
        self.frame_count += 1
        h, w = frame.shape[:2]

        # Detect
        yolo_out = self.model(frame, verbose=False, conf=self.conf_thresh, imgsz=self.imgsz)[0]
        dets = sv.Detections.from_ultralytics(yolo_out)

        # Keep only vehicles
        vehicle_mask = np.isin(dets.class_id, list(VEHICLE_IDS))
        dets = dets[vehicle_mask]

        # Track
        tracks = self.tracker.update_with_detections(dets)

        # Annotate
        annotated = frame.copy()

        for i in range(len(tracks)):
            tid = int(tracks.tracker_id[i])
            x1, y1, x2, y2 = map(int, tracks.xyxy[i])
            cls_id = int(tracks.class_id[i])
            conf = float(tracks.confidence[i])
            label = yolo_out.names.get(cls_id, "?")

            # Speed
            speed_kmh = self._estimate_speed(tid, (x1 + x2) // 2, y2)

            violation = speed_kmh is not None and speed_kmh > self.speed_limit_kmh

            self.results.append(Detection(
                frame=self.frame_count, track_id=tid,
                class_name=label, confidence=conf,
                bbox=(x1, y1, x2, y2), speed_kmh=speed_kmh,
                is_violation=violation,
            ))

            # Draw
            color = (0, 0, 255) if violation else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            text = f"#{tid} {label}"
            if speed_kmh is not None:
                text += f" {speed_kmh:.0f} km/h"
            cv2.putText(annotated, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return annotated

    def process_video(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        max_frames: Optional[int] = None,
        progress_callback=None,
    ) -> dict:
        """Process entire video file. Returns summary dict.

        If progress_callback is set, calls it as callback(frame_count, total_frames).
        """
        cap = cv2.VideoCapture(str(video_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        self.fps = fps
        # Re-create tracker with real fps
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
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

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

    def get_violations(self) -> list[Detection]:
        """Return all violation events (one per frame per vehicle)."""
        return [d for d in self.results if d.is_violation]

    def violations_by_track(self) -> dict[int, list[Detection]]:
        """Group violations by track ID (vehicle)."""
        by_track: dict[int, list[Detection]] = {}
        for d in self.results:
            if d.is_violation:
                by_track.setdefault(d.track_id, []).append(d)
        return by_track

    def results_csv(self, path: str | Path):
        """Write all detections to CSV."""
        import csv
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "track_id", "class", "confidence",
                         "x1", "y1", "x2", "y2", "speed_kmh", "violation"])
            for d in self.results:
                w.writerow([
                    d.frame, d.track_id, d.class_name,
                    f"{d.confidence:.3f}",
                    *d.bbox,
                    f"{d.speed_kmh:.1f}" if d.speed_kmh is not None else "",
                    "YES" if d.is_violation else "",
                ])

    # ---- internals --------------------------------------------------------

    def _image_to_world(self, u: float, v: float) -> Optional[tuple[float, float]]:
        """Map image pixel → ground-plane metres via homography."""
        if self.cal is None:
            return None
        p = self.cal.H @ np.array([u, v, 1.0])
        if p[2] == 0:
            return None
        return (float(p[0] / p[2]), float(p[1] / p[2]))

    def _estimate_speed(self, track_id: int, u: int, v: int) -> Optional[float]:
        """Compute speed in km/h from position history."""
        if self.cal is None:
            return None

        world = self._image_to_world(u, v)
        if world is None:
            return None

        buf = self._pos_buffer.setdefault(track_id, [])
        buf.append((self.frame_count, *world))
        if len(buf) > SPEED_BUFFER_FRAMES:
            buf.pop(0)
        if len(buf) < 5:
            return None

        # Speed from oldest → newest in buffer
        t0, x0, y0 = buf[0]
        t1, x1, y1 = buf[-1]
        dt = (t1 - t0) / self.fps
        if dt <= 0:
            return None
        dist = np.hypot(x1 - x0, y1 - y0)
        # m/s → km/h
        return dist / dt * 3.6

    def _summary(self) -> dict:
        """Aggregate statistics."""
        speeds = [d.speed_kmh for d in self.results if d.speed_kmh is not None]
        violations = sum(1 for d in self.results if d.is_violation)
        tracks = set(d.track_id for d in self.results)
        return dict(
            frames_processed=self.frame_count,
            unique_vehicles=len(tracks),
            total_detections=len(self.results),
            violations=violations,
            avg_speed_kmh=round(float(np.mean(speeds)), 1) if speeds else 0.0,
            max_speed_kmh=round(float(max(speeds)), 1) if speeds else 0.0,
            speed_limit_kmh=self.speed_limit_kmh,
        )

"""
Traffic Intelligence Engine
Core pipeline: YOLO detection → ByteTrack tracking → speed trap → violation logic

Usage:
    engine = TrafficEngine(calibration=Calibration.load("calib.json"))
    summary = engine.process_video("traffic.mp4", output_path="annotated.mp4")
"""

import json
from dataclasses import dataclass
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
    """Camera calibration for ROI, homography, and speed trap."""
    H: Optional[np.ndarray] = None       # 3x3 homography (image -> ground plane)
    width_m: float = 3.7                 # road width perpendicular to traffic
    length_m: float = 10.0               # road length along traffic
    points_uv: list = None               # 4 homography calibration points
    roi_polygon: Optional[list] = None   # [[x,y], ...] drivable area
    speed_trap: Optional[dict] = None    # {"line_a": [[x1,y1],[x2,y2]], "line_b": ..., "distance_m": float}
    speed_unit: str = "mph"              # "mph" or "kmh"

    def save(self, path: str | Path):
        data = dict(
            H=self.H.tolist() if self.H is not None else None,
            width_m=self.width_m, length_m=self.length_m,
            points_uv=self.points_uv or [],
            roi_polygon=self.roi_polygon or [],
            speed_trap=self.speed_trap,
            speed_unit=self.speed_unit,
        )
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path):
        data = json.loads(Path(path).read_text())
        H = np.array(data["H"]) if data.get("H") else None
        return cls(
            H=H, width_m=data.get("width_m", 3.7),
            length_m=data.get("length_m", 10.0),
            points_uv=data.get("points_uv") or [],
            roi_polygon=data.get("roi_polygon"),
            speed_trap=data.get("speed_trap"),
            speed_unit=data.get("speed_unit", "mph"),
        )


@dataclass
class Detection:
    """A single detection event for one vehicle in one frame."""
    frame: int
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple          # x1, y1, x2, y2
    speed: Optional[float]   # in the engine's active unit (mph or kmh)
    speed_unit: str
    is_violation: bool


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

VEHICLE_IDS = {2, 3, 5, 7}       # car, motorcycle, bus, truck (COCO)
SPEED_BUFFER_FRAMES = 15          # ~0.5 s at 30 fps (legacy homography)


class TrafficEngine:
    """Video-in -> detections + tracks + speed trap readings + violations."""

    def __init__(
        self,
        model_path: str = "yolo11m.pt",
        calibration: Optional[Calibration] = None,
        speed_limit: float = 50,
        speed_unit: str = "mph",
        confidence: float = 0.25,
        fps: float = 30.0,
        imgsz: int = 1280,
    ):
        self.model = YOLO(model_path)
        self.cal = calibration
        self.conf_thresh = confidence
        self.imgsz = imgsz
        self.fps = fps

        # Speed limit in the active unit
        self.speed_limit = speed_limit
        self.speed_unit = speed_unit if calibration is None else calibration.speed_unit

        self.tracker = sv.ByteTrack(
            track_activation_threshold=confidence,
            lost_track_buffer=int(fps),
            minimum_matching_threshold=0.8,
            frame_rate=fps,
        )

        # Per-track history
        self._pos_buffer: dict[int, list] = {}   # track_id -> [(frame, x_world, y_world)]
        self._prev_pos: dict[int, tuple] = {}    # track_id -> (cx, cy) previous frame
        self._trap_state: dict[int, dict] = {}   # track_id -> {"la": frame, "lb": frame, "fired": bool}

        self.results: list[Detection] = []
        self.frame_count = 0

    # ---- public API -------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one frame -> annotated copy with speed trap readings."""
        self.frame_count += 1

        # Detect
        yolo_out = self.model(frame, verbose=False, conf=self.conf_thresh,
                              imgsz=self.imgsz)[0]
        dets = sv.Detections.from_ultralytics(yolo_out)

        # Keep only vehicles
        dets = dets[np.isin(dets.class_id, list(VEHICLE_IDS))]

        # ROI filter: remove detections outside the drivable polygon
        if self.cal and self.cal.roi_polygon:
            roi = np.array(self.cal.roi_polygon, dtype=np.int32)
            keep = []
            for i in range(len(dets)):
                x1, y1, x2, y2 = dets.xyxy[i].astype(int)
                cx, cy = (x1 + x2) // 2, y2   # bottom-centre
                if cv2.pointPolygonTest(roi, (float(cx), float(cy)), False) >= 0:
                    keep.append(i)
            dets = dets[keep] if keep else dets[:0]

        # Track
        tracks = self.tracker.update_with_detections(dets)
        annotated = frame.copy()

        # Draw ROI + speed trap lines on every frame
        if self.cal:
            if self.cal.roi_polygon:
                cv2.polylines(annotated, [np.array(self.cal.roi_polygon, dtype=np.int32)],
                              True, (0, 255, 255), 2)
            st = self.cal.speed_trap
            if st:
                for line_key, colour in [("line_a", (255, 0, 0)), ("line_b", (0, 0, 255))]:
                    pts = st.get(line_key)
                    if pts and len(pts) == 2:
                        cv2.line(annotated, tuple(pts[0]), tuple(pts[1]), colour, 2)

        for i in range(len(tracks)):
            tid = int(tracks.tracker_id[i])
            x1, y1, x2, y2 = map(int, tracks.xyxy[i])
            cls_id = int(tracks.class_id[i])
            conf = float(tracks.confidence[i])
            label = yolo_out.names.get(cls_id, "?")
            cx, cy = (x1 + x2) // 2, y2

            # Speed: try speed trap first, fall back to homography
            speed = self._speed_trap(tid, cx, cy)
            if speed is None:
                speed = self._estimate_speed(tid, cx, cy)

            speed_display = speed
            violation = speed_display is not None and speed_display > self.speed_limit

            self.results.append(Detection(
                frame=self.frame_count, track_id=tid,
                class_name=label, confidence=conf,
                bbox=(x1, y1, x2, y2), speed=speed_display,
                speed_unit=self.speed_unit, is_violation=violation,
            ))

            colour = (0, 0, 255) if violation else (0, 255, 0)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

            unit_label = "mph" if self.speed_unit == "mph" else "km/h"
            text = f"#{tid} {label}"
            if speed_display is not None:
                text += f" {speed_display:.0f} {unit_label}"
            cv2.putText(annotated, text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        return annotated

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

    def get_violations(self) -> list[Detection]:
        return [d for d in self.results if d.is_violation]

    def violations_by_track(self) -> dict[int, list[Detection]]:
        by_track: dict[int, list[Detection]] = {}
        for d in self.results:
            if d.is_violation:
                by_track.setdefault(d.track_id, []).append(d)
        return by_track

    def results_csv(self, path: str | Path):
        import csv
        unit = self.speed_unit
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "track_id", "class", "confidence",
                         "x1", "y1", "x2", "y2", f"speed_{unit}", "violation"])
            for d in self.results:
                w.writerow([
                    d.frame, d.track_id, d.class_name,
                    f"{d.confidence:.3f}", *d.bbox,
                    f"{d.speed:.1f}" if d.speed is not None else "",
                    "YES" if d.is_violation else "",
                ])

    # ---- speed trap (line-crossing) ---------------------------------------

    def _line_side(self, p1, p2, p):
        """Cross product sign: which side of the line (p1->p2) is point p on?"""
        return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])

    def _crossed_line(self, prev, curr, line_p1, line_p2):
        """Did the point cross the line between prev and curr frames?"""
        s1 = self._line_side(line_p1, line_p2, prev)
        s2 = self._line_side(line_p1, line_p2, curr)
        return (s1 > 0) != (s2 > 0)  # sign changed (ignore exactly-on-line cases)

    def _speed_trap(self, track_id: int, cx: int, cy: int) -> Optional[float]:
        """Check speed trap crossing -> speed in active unit, or None."""
        if not (self.cal and self.cal.speed_trap):
            return None

        st = self.cal.speed_trap
        line_a = st.get("line_a")
        line_b = st.get("line_b")
        dist_m = st.get("distance_m", 10.0)
        if not (line_a and line_b and dist_m > 0):
            return None

        prev = self._prev_pos.get(track_id)
        self._prev_pos[track_id] = (cx, cy)
        if prev is None:
            return None

        state = self._trap_state.setdefault(track_id, {"la": None, "lb": None, "fired": False})
        if state["fired"]:
            return None

        # Check both lines
        if state["la"] is None and self._crossed_line(prev, (cx, cy), *line_a):
            state["la"] = self.frame_count
        if state["lb"] is None and self._crossed_line(prev, (cx, cy), *line_b):
            state["lb"] = self.frame_count

        # If both crossed, compute speed
        if state["la"] is not None and state["lb"] is not None:
            state["fired"] = True
            dt = abs(state["la"] - state["lb"]) / self.fps
            if dt <= 0:
                return None
            speed_ms = dist_m / dt
            if self.speed_unit == "mph":
                return speed_ms * 2.23694   # m/s -> mph
            else:
                return speed_ms * 3.6       # m/s -> km/h

        return None

    # ---- homography speed (fallback) --------------------------------------

    def _image_to_world(self, u: float, v: float) -> Optional[tuple[float, float]]:
        if self.cal is None or self.cal.H is None:
            return None
        p = self.cal.H @ np.array([u, v, 1.0])
        if p[2] == 0:
            return None
        return (float(p[0] / p[2]), float(p[1] / p[2]))

    def _estimate_speed(self, track_id: int, u: int, v: int) -> Optional[float]:
        """Per-frame homography-based speed (km/h). Converted to active unit."""
        if self.cal is None or self.cal.H is None:
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

        t0, x0, y0 = buf[0]
        t1, x1, y1 = buf[-1]
        dt = (t1 - t0) / self.fps
        if dt <= 0:
            return None
        dist = np.hypot(x1 - x0, y1 - y0)
        speed_ms = dist / dt
        if self.speed_unit == "mph":
            return speed_ms * 2.23694
        return speed_ms * 3.6

    # ---- summary ----------------------------------------------------------

    def _summary(self) -> dict:
        speeds = [d.speed for d in self.results if d.speed is not None]
        violations = sum(1 for d in self.results if d.is_violation)
        tracks = set(d.track_id for d in self.results)
        return dict(
            frames_processed=self.frame_count,
            unique_vehicles=len(tracks),
            total_detections=len(self.results),
            violations=violations,
            avg_speed=round(float(np.mean(speeds)), 1) if speeds else 0.0,
            max_speed=round(float(max(speeds)), 1) if speeds else 0.0,
            speed_limit=self.speed_limit,
            speed_unit=self.speed_unit,
        )

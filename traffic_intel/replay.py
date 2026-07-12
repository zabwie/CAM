"""Replay tracked detections onto a video with robust speed estimation.

This is useful when detector/tracker output has already been exported to CSV.
It reuses the real frame-by-frame boxes and track IDs, recomputes speed from
calibrated trajectories, and writes an annotated video plus a new speed CSV.

Example:
    python -m traffic_intel.replay \
        --video clear.mp4 --detections results.csv --calibration calib.json \
        --output annotated.mp4 --output-csv stable_results.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

try:
    from .calibration import Calibration
    from .speed import RobustSpeedEstimator
except ImportError:
    from calibration import Calibration
    from speed import RobustSpeedEstimator


@dataclass(frozen=True)
class CsvDetection:
    frame: int
    track_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


def load_detections(path: str | Path) -> dict[int, list[CsvDetection]]:
    """Load common tracking CSV formats and group rows by 1-based frame number."""
    grouped: dict[int, list[CsvDetection]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"frame", "track_id", "confidence", "x1", "y1", "x2", "y2"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Detection CSV missing columns: {', '.join(sorted(missing))}")

        class_column = "class_name" if "class_name" in (reader.fieldnames or []) else "class"
        if class_column not in (reader.fieldnames or []):
            raise ValueError("Detection CSV needs a 'class' or 'class_name' column")

        for row in reader:
            det = CsvDetection(
                frame=int(row["frame"]),
                track_id=int(row["track_id"]),
                class_name=row[class_column],
                confidence=float(row["confidence"]),
                bbox=(int(float(row["x1"])), int(float(row["y1"])),
                      int(float(row["x2"])), int(float(row["y2"]))),
            )
            grouped[det.frame].append(det)
    return dict(grouped)


def _world_from_bbox(calibration: Calibration, bbox: tuple[int, int, int, int]) -> Optional[tuple[float, float]]:
    x1, _y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    return calibration.world_from_image(cx, float(y2))


def _draw_vehicle_label(
    frame: np.ndarray,
    *,
    x1: int,
    y1: int,
    track_id: int,
    class_name: str,
    speed_mph: Optional[float],
) -> None:
    speed_text = f"{speed_mph:.0f} MPH" if speed_mph is not None else "-- MPH"
    text = f"{class_name} | {speed_text} | #{track_id}"
    cv2.putText(
        frame, text, (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA,
    )


def replay_video(
    video_path: str | Path,
    detections_csv: str | Path,
    calibration_path: str | Path,
    output_path: str | Path,
    output_csv: str | Path | None = None,
    progress_every: int = 300,
) -> dict:
    """Render real tracked detections with recomputed stable MPH values."""
    detections = load_detections(detections_csv)
    calibration = Calibration.load(calibration_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    estimator = RobustSpeedEstimator(fps=fps)
    output_rows: list[tuple] = []
    frame_no = 0
    valid_speed_count = 0
    valid_tracks: set[int] = set()
    speed_values: list[float] = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            annotated = frame.copy()

            for det in detections.get(frame_no, []):
                x1, y1, x2, y2 = det.bbox
                mph = None
                world_pt = _world_from_bbox(calibration, det.bbox)
                if world_pt is not None:
                    wx, wy = world_pt
                    mph = estimator.update(det.track_id, frame_no, wx, wy)

                speed_valid = mph is not None
                speed_value = float(mph) if speed_valid else 0.0
                if speed_valid:
                    valid_speed_count += 1
                    valid_tracks.add(det.track_id)
                    speed_values.append(speed_value)

                output_rows.append((
                    frame_no, det.track_id, det.class_name, det.confidence,
                    x1, y1, x2, y2, speed_value, int(speed_valid),
                ))

                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                _draw_vehicle_label(
                    annotated, x1=x1, y1=y1, track_id=det.track_id,
                    class_name=det.class_name,
                    speed_mph=speed_value if speed_valid else None,
                )

            estimator.forget_stale(frame_no)
            writer.write(annotated)

            if progress_every and frame_no % progress_every == 0:
                print(f"Rendered {frame_no}/{total_frames} frames", flush=True)
    finally:
        cap.release()
        writer.release()

    if output_csv:
        with open(output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "frame", "track_id", "class", "confidence",
                "x1", "y1", "x2", "y2", "speed_mph", "speed_valid",
            ])
            for row in output_rows:
                w.writerow([
                    row[0], row[1], row[2], f"{row[3]:.3f}",
                    row[4], row[5], row[6], row[7], f"{row[8]:.1f}", row[9],
                ])

    summary = {
        "frames_processed": frame_no,
        "detection_rows": len(output_rows),
        "tracks_with_valid_speed": len(valid_tracks),
        "valid_speed_rows": valid_speed_count,
        "average_speed_mph": float(np.mean(speed_values)) if speed_values else None,
        "max_speed_mph": float(np.max(speed_values)) if speed_values else None,
    }
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", required=True)
    p.add_argument("--detections", required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--output-csv", default=None)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    summary = replay_video(
        video_path=args.video,
        detections_csv=args.detections,
        calibration_path=args.calibration,
        output_path=args.output,
        output_csv=args.output_csv,
    )
    print(summary)


if __name__ == "__main__":
    main()

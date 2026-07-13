"""
Live camera feed — detection, tracking, speed estimation, event capture.

Runs the full pipeline against a live camera (or any CV2-compatible source).
Maintains a rolling video + telemetry buffer for event capture.

Controls:
  q  Quit
  m  Manual event save (saves 20s before + 10s after)

Usage:
    python3 -m traffic_intel.live --camera 0
    python3 -m traffic_intel.live --camera rtsp://192.168.1.100:554/stream1
    python3 -m traffic_intel.live --camera 0 --calibration calib.json --speed-limit 50
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

try:
    from .core.engine import TrafficEngine
    from .motion.calibration import Calibration
    from .recording.event_recorder import RollingSegmentBuffer
    from .core.pipeline import TrafficIncidentPipeline
except ImportError:  # direct script execution from repository root
    from traffic_intel.core.engine import TrafficEngine
    from traffic_intel.motion.calibration import Calibration
    from traffic_intel.recording.event_recorder import RollingSegmentBuffer
    from traffic_intel.core.pipeline import TrafficIncidentPipeline


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Live traffic camera")
    ap.add_argument("--camera", default="0",
                    help="Camera index (0) or RTSP URL")
    ap.add_argument("--calibration", default=None,
                    help="Path to calibration JSON (optional — no speed without it)")
    ap.add_argument("--model", default="models/yolo11n.pt",
                    help="YOLO model path")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Inference resolution (lower = faster)")
    ap.add_argument("--speed-limit", type=float, default=None,
                    help="Alert threshold in mph")
    ap.add_argument("--save-video", default=None,
                    help="Path to save annotated video (optional)")
    ap.add_argument("--event-dir", default="events",
                    help="Directory for saved event packages")
    ap.add_argument("--pre-event-seconds", type=float, default=20.0,
                    help="Seconds of video before trigger to include")
    ap.add_argument("--post-event-seconds", type=float, default=10.0,
                    help="Seconds of video after trigger to include")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()

    # Camera source — index or URL.
    source = int(args.camera) if args.camera.isdigit() else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera: {args.camera}")

    # Read actual camera FPS and resolution.
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {cam_w}x{cam_h} @ {cam_fps:.1f} fps")

    # Load calibration if provided.
    calib = None
    if args.calibration:
        calib = Calibration.load(args.calibration)
        print(f"Calibration: {calib.quality_grade}")

    # Build engine with live-friendly settings.
    engine = TrafficEngine(
        model_path=args.model,
        calibration=calib,
        fps=cam_fps,
        imgsz=args.imgsz,
        retain_history=False,
    )
    print(f"  imgsz={args.imgsz}px")

    event_buffer = RollingSegmentBuffer(
        fps=cam_fps,
        output_dir=args.event_dir,
        max_seconds=args.pre_event_seconds,
        post_event_seconds=args.post_event_seconds,
    )
    pipeline = TrafficIncidentPipeline(engine)
    print(f"Events: {args.event_dir}/  (press 'm' to save, crash detection active)")

    # Optional video writer.
    writer = None
    if args.save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_video, fourcc, cam_fps, (cam_w, cam_h))

    manual_trigger_pending = False
    seen_track_ids: set[int] = set()
    print("\nLive — press 'q' to quit, 'm' to save event.\n")
    fps_display = "?"

    while True:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            print("Camera disconnected.")
            break

        result = pipeline.process_frame(frame, optical_flow=True)
        annotated = result.annotated
        current_detections = result.detections
        candidates = result.crashes
        seen_track_ids.update(d.track_id for d in current_detections)
        for c in candidates:
            print(f"\n  Crash candidate: {c.description}")

        # Buffer the fully annotated current frame before triggering the event.
        event_buffer.write_frame(annotated, current_detections, engine.frame_count)
        for c in candidates:
            event_buffer.trigger_auto(
                c.reason,
                engine,
                trigger_frame=c.trigger_frame,
                event_metadata={
                    "detected_frame": c.detected_frame,
                    "score": round(c.score, 6),
                    "involved_tracks": c.involved_tracks,
                    "description": c.description,
                    "evidence": c.evidence,
                },
            )

        # Handle manual trigger (deferred so buffer has the frame).
        if manual_trigger_pending:
            manual_trigger_pending = False
            event_buffer.trigger_manual(engine)

        # Overlay FPS and calibration status.
        info = f"FPS: {fps_display}  |  Frame: {engine.frame_count}"
        if calib:
            info += f"  |  Cal: {calib.quality_grade}"
        cv2.putText(annotated, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Speed violation alert: inspect every valid vehicle, not only the
        # final detection returned by the tracker.  Show the strongest current
        # violation to avoid stacking unreadable alerts.
        if args.speed_limit and current_detections:
            violators = [
                d for d in current_detections
                if d.speed_valid and d.speed > args.speed_limit
            ]
            if violators:
                fastest = max(violators, key=lambda d: d.speed)
                alert = f"SPEEDING: {fastest.speed:.0f} mph  #{fastest.track_id}"
                (tw, th), _ = cv2.getTextSize(
                    alert, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                )
                x_off = (annotated.shape[1] - tw) // 2
                cv2.putText(
                    annotated, alert, (x_off, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2,
                )

        cv2.imshow("Traffic Intelligence — Live", annotated)

        if writer:
            writer.write(annotated)

        # Compute moving-average FPS.
        elapsed = time.perf_counter() - t0
        if elapsed > 0:
            fps_display = f"{1.0 / elapsed:.1f}"

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("Stopped by user.")
            break
        elif key == ord("m"):
            manual_trigger_pending = True

    cap.release()
    if writer:
        writer.release()
    event_buffer.close()
    cv2.destroyAllWindows()

    print(f"Frames processed: {engine.frame_count}")
    print(f"Vehicles tracked: {len(seen_track_ids)}")


if __name__ == "__main__":
    main()

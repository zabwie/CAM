"""Live traffic camera runtime with pilot analytics persistence.

The camera is read continuously on a background thread while inference consumes
only the newest frame. Speed uses the frame's monotonic receipt timestamp, so a
slow model or skipped frames do not masquerade as higher vehicle speed.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

try:
    from . import __version__
    from .analytics import CameraHealthAccumulator, VehiclePassageAggregator
    from .analytics_store import AnalyticsStore
    from .capture import LatestFrameCapture
    from .core.engine import TrafficEngine
    from .core.pipeline import TrafficIncidentPipeline
    from .motion.calibration import Calibration
    from .recording.event_recorder import RollingSegmentBuffer
    from .vision_quality import VisionQualityMonitor
except ImportError:  # direct script execution from repository root
    from traffic_intel import __version__
    from traffic_intel.analytics import CameraHealthAccumulator, VehiclePassageAggregator
    from traffic_intel.analytics_store import AnalyticsStore
    from traffic_intel.capture import LatestFrameCapture
    from traffic_intel.core.engine import TrafficEngine
    from traffic_intel.core.pipeline import TrafficIncidentPipeline
    from traffic_intel.motion.calibration import Calibration
    from traffic_intel.recording.event_recorder import RollingSegmentBuffer
    from traffic_intel.vision_quality import VisionQualityMonitor


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Live traffic camera")
    ap.add_argument("--camera", default="0", help="Camera index or RTSP URL")
    ap.add_argument("--camera-id", default="camera-1", help="Stable analytics camera ID")
    ap.add_argument("--municipality", default="", help="Municipality recorded in analytics")
    ap.add_argument("--location-id", default="", help="Road/site identifier")
    ap.add_argument("--calibration", default=None, help="Calibration JSON")
    ap.add_argument("--model", default="models/yolo11n.pt", help="YOLO model")
    ap.add_argument("--imgsz", type=int, default=1280, help="Inference resolution")
    ap.add_argument("--speed-limit", type=float, default=None, help="Configured limit in mph")
    ap.add_argument("--save-video", default=None, help="Annotated video output")
    ap.add_argument("--event-dir", default="events", help="Saved event packages")
    ap.add_argument("--pre-event-seconds", type=float, default=20.0)
    ap.add_argument("--post-event-seconds", type=float, default=10.0)
    ap.add_argument("--analytics-db", default="analytics.db", help="SQLite pilot ledger")
    ap.add_argument("--disable-analytics", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    source = int(args.camera) if str(args.camera).isdigit() else args.camera
    capture = LatestFrameCapture(source)
    cam_fps = capture.fps
    cam_w, cam_h = capture.width, capture.height
    print(f"Camera: {cam_w}x{cam_h} @ nominal {cam_fps:.1f} fps")

    calibration = Calibration.load(args.calibration) if args.calibration else None
    if calibration is not None:
        print(f"Calibration: {calibration.quality_grade}")

    engine = TrafficEngine(
        model_path=args.model,
        calibration=calibration,
        fps=cam_fps,
        imgsz=args.imgsz,
        retain_history=False,
    )
    pipeline = TrafficIncidentPipeline(engine)
    event_buffer = RollingSegmentBuffer(
        fps=cam_fps,
        output_dir=args.event_dir,
        max_seconds=args.pre_event_seconds,
        post_event_seconds=args.post_event_seconds,
    )
    quality_monitor = VisionQualityMonitor()

    store: AnalyticsStore | None = None
    passage_aggregator: VehiclePassageAggregator | None = None
    health: CameraHealthAccumulator | None = None
    if not args.disable_analytics:
        store = AnalyticsStore(args.analytics_db)
        passage_aggregator = VehiclePassageAggregator(
            camera_id=args.camera_id,
            municipality=args.municipality,
            location_id=args.location_id,
            speed_limit_mph=args.speed_limit,
            calibration_id=(Path(args.calibration).name if args.calibration else ""),
            software_version=__version__,
        )
        health = CameraHealthAccumulator(args.camera_id)
        print(f"Analytics: {args.analytics_db} ({args.camera_id})")

    writer = None
    if args.save_video:
        writer = cv2.VideoWriter(
            args.save_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            cam_fps,
            (cam_w, cam_h),
        )

    manual_trigger_pending = False
    seen_track_ids: set[int] = set()
    fps_display = "?"
    last_sequence = 0
    last_capture_timestamp = time.time()
    print("\nLive — press 'q' to quit, 'm' to save event.\n")

    try:
        while True:
            t0 = time.perf_counter()
            packet = capture.read_packet(after_sequence=last_sequence, timeout=5.0)
            if packet is None:
                print(f"No new camera frame. {capture.last_error or ''}")
                continue

            sequence_gap = max(0, packet.sequence - last_sequence - 1)
            last_sequence = packet.sequence
            last_capture_timestamp = packet.capture_timestamp
            frame = packet.image
            quality = quality_monitor.update(frame)

            result = pipeline.process_frame(
                frame,
                optical_flow=True,
                capture_timestamp=packet.capture_timestamp,
                monotonic_timestamp=packet.monotonic_timestamp,
                vision_state=quality.state,
            )
            annotated = result.annotated
            detections = result.detections
            candidates = result.crashes
            seen_track_ids.update(d.track_id for d in detections)

            if store is not None and passage_aggregator is not None and health is not None:
                finalized = passage_aggregator.update(
                    detections,
                    capture_timestamp=packet.capture_timestamp,
                    monotonic_timestamp=packet.monotonic_timestamp,
                    vision_state=quality.state,
                )
                store.write_passages(finalized)
                store.write_camera_health(
                    health.update(
                        capture_timestamp=packet.capture_timestamp,
                        monotonic_timestamp=packet.monotonic_timestamp,
                        sequence_gap=sequence_gap,
                        detections=detections,
                        quality=quality,
                    )
                )

            event_buffer.write_frame(annotated, detections, engine.frame_count)
            for candidate in candidates:
                print(f"\nCrash candidate: {candidate.description}")
                event_buffer.trigger_auto(
                    candidate.reason,
                    engine,
                    trigger_frame=candidate.trigger_frame,
                    event_metadata={
                        "detected_frame": candidate.detected_frame,
                        "score": round(candidate.score, 6),
                        "involved_tracks": candidate.involved_tracks,
                        "description": candidate.description,
                        "evidence": candidate.evidence,
                        "vision_state": quality.state,
                    },
                )
                if store is not None:
                    event_id = (
                        f"{args.camera_id}:{packet.capture_timestamp:.6f}:"
                        + "-".join(map(str, candidate.involved_tracks))
                    )
                    store.write_incident(
                        event_id=event_id,
                        camera_id=args.camera_id,
                        occurred_at=packet.capture_timestamp,
                        incident_type=candidate.reason,
                        score=candidate.score,
                        involved_tracks=candidate.involved_tracks,
                        metadata={
                            "description": candidate.description,
                            "trigger_frame": candidate.trigger_frame,
                            "detected_frame": candidate.detected_frame,
                            "vision_state": quality.state,
                        },
                    )

            if manual_trigger_pending:
                manual_trigger_pending = False
                event_buffer.trigger_manual(engine)

            info = (
                f"FPS: {fps_display} | Frame: {engine.frame_count} | "
                f"Vision: {quality.state}"
            )
            if calibration:
                info += f" | Cal: {calibration.quality_grade}"
            cv2.putText(
                annotated,
                info,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            if args.speed_limit and detections:
                violators = [
                    d for d in detections if d.speed_valid and d.speed > args.speed_limit
                ]
                if violators:
                    fastest = max(violators, key=lambda d: d.speed)
                    alert = f"SPEEDING: {fastest.speed:.0f} mph #{fastest.track_id}"
                    (tw, _), _ = cv2.getTextSize(
                        alert, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
                    )
                    cv2.putText(
                        annotated,
                        alert,
                        ((annotated.shape[1] - tw) // 2, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2,
                    )

            cv2.imshow("Traffic Intelligence — Live", annotated)
            if writer:
                writer.write(annotated)

            elapsed = time.perf_counter() - t0
            if elapsed > 0:
                fps_display = f"{1.0 / elapsed:.1f}"

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("m"):
                manual_trigger_pending = True
    finally:
        try:
            if store is not None and passage_aggregator is not None:
                store.write_passages(passage_aggregator.flush())
            if store is not None and health is not None:
                final_health = health.flush(last_capture_timestamp)
                if final_health is not None:
                    store.write_camera_health([final_health])
        finally:
            if store is not None:
                store.close()
            capture.release()
            if writer:
                writer.release()
            event_buffer.close()
            cv2.destroyAllWindows()

    print(f"Frames processed: {engine.frame_count}")
    print(f"Vehicles tracked: {len(seen_track_ids)}")


if __name__ == "__main__":
    main()

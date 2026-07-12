"""Headless crash-detector validation for recorded videos.

Example:
    python -m traffic_intel.validate_crashes crash.mp4 \
        --model yolo11n.mlpackage --imgsz 1280 \
        --output crash_validated.mp4 --events-json crash_events.json

The command uses the same detector/tracker/crash logic as live.py, but does not
open a GUI and writes machine-readable event timing/evidence for regression
comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

try:
    from .engine import TrafficEngine
    from .pipeline import TrafficIncidentPipeline
except ImportError:  # pragma: no cover - direct script execution
    from traffic_intel.engine import TrafficEngine
    from traffic_intel.pipeline import TrafficIncidentPipeline


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate crash detection on a video")
    ap.add_argument("video")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--output", default=None, help="Optional annotated MP4")
    ap.add_argument("--events-json", default=None, help="Event JSON output path")
    ap.add_argument("--no-flow", action="store_true", help="Disable optical-flow support")
    ap.add_argument("--max-frames", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = _args()
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    engine = TrafficEngine(
        model_path=args.model,
        fps=fps,
        imgsz=args.imgsz,
        retain_history=False,
    )
    pipeline = TrafficIncidentPipeline(engine)

    writer = None
    if args.output:
        writer = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise SystemExit(f"Cannot open output writer: {args.output}")

    events: list[dict] = []
    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        result = pipeline.process_frame(frame, optical_flow=not args.no_flow)
        annotated = result.annotated
        for c in result.crashes:
            events.append(
                {
                    "trigger_frame": c.trigger_frame,
                    "detected_frame": c.detected_frame,
                    "trigger_seconds": round(c.trigger_frame / fps, 4),
                    "detected_seconds": round(c.detected_frame / fps, 4),
                    "score": round(c.score, 6),
                    "reason": c.reason,
                    "description": c.description,
                    "involved_tracks": c.involved_tracks,
                    "evidence": c.evidence,
                }
            )

        if writer is not None:
            writer.write(annotated)

        if frame_count % 30 == 0:
            print(f"{frame_count}/{total or '?'} frames | events={len(events)}", flush=True)
        if args.max_frames and frame_count >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()

    payload = {
        "video": str(Path(args.video)),
        "fps": fps,
        "frames_processed": frame_count,
        "imgsz": args.imgsz,
        "model": args.model,
        "optical_flow": not args.no_flow,
        "events": events,
    }
    json_path = args.events_json or f"{Path(args.video).stem}_crash_events.json"
    Path(json_path).write_text(json.dumps(payload, indent=2))

    print(f"\nEvents: {len(events)}")
    for event in events:
        print(
            f"  impact f{event['trigger_frame']} -> detected f{event['detected_frame']} "
            f"tracks={event['involved_tracks']} score={event['score']:.3f}"
        )
    print(f"Event JSON: {json_path}")


if __name__ == "__main__":
    main()

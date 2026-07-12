"""Replay cached YOLO outputs through production tracking/identity/crash logic.

This isolates association and incident-algorithm regressions from model runtime
and hardware differences.  When the source video is available, the replay also
uses real frame pixels for canonical identity appearance matching and production
scene-cut resets.  It does not replace full 1280/Core ML validation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack

from traffic_intel.config import SceneChangeConfig, TrackingConfig
from traffic_intel.crash_detector import CrashDetector
from traffic_intel.domain import Detection
from traffic_intel.identity import CanonicalIdentityManager, RawTrackObservation
from traffic_intel.scene import SceneChangeDetector
from traffic_intel.tracking import TrackQualityGate

VEHICLE_IDS = {2, 3, 5, 7}


def _new_tracker(cfg: TrackingConfig, fps: float) -> ByteTrack:
    return ByteTrack(
        track_activation_threshold=cfg.activation_threshold,
        lost_track_buffer=max(1, int(round(cfg.lost_track_seconds * fps))),
        minimum_matching_threshold=cfg.minimum_matching_threshold,
        frame_rate=max(1, int(round(fps))),
    )


def replay(cache_path: Path, video_path: Path | None = None) -> dict:
    cache = json.loads(cache_path.read_text())
    fps = float(cache["fps"])
    cfg = TrackingConfig()
    scene_cfg = SceneChangeConfig()

    tracker = _new_tracker(cfg, fps)
    identity = CanonicalIdentityManager(fps, cfg)
    gate = TrackQualityGate(fps, cfg)
    crash = CrashDetector(fps=fps)
    scene = SceneChangeDetector(scene_cfg)
    warmup_remaining = 0
    events = []
    scene_cuts: list[int] = []
    stitch_events: list[dict] = []
    discontinuity_events: list[dict] = []

    cap = cv2.VideoCapture(str(video_path)) if video_path is not None else None
    if cap is not None and not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source video: {video_path}")

    try:
        for frame_payload in cache["frames"]:
            frame = int(frame_payload["frame"])
            image = None
            if cap is not None:
                ok, image = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Video ended before cached frame {frame}: {video_path}"
                    )
                if scene.update(image):
                    stitch_events.extend(identity.stitch_events)
                    discontinuity_events.extend(identity.discontinuity_events)
                    scene_cuts.append(frame)
                    tracker = _new_tracker(cfg, fps)
                    identity.reset(reset_counter=False)
                    gate.reset()
                    crash.reset()
                    crash.set_fps(fps)
                    warmup_remaining = max(
                        3, int(round(scene_cfg.warmup_seconds * fps))
                    )

            raw = [
                d for d in frame_payload["dets"]
                if d["cls"] in VEHICLE_IDS and d["conf"] >= cfg.detector_confidence
            ]
            if raw:
                detections = sv.Detections(
                    xyxy=np.asarray([d["xyxy"] for d in raw], dtype=np.float32),
                    confidence=np.asarray([d["conf"] for d in raw], dtype=np.float32),
                    class_id=np.asarray([d["cls"] for d in raw], dtype=int),
                ).with_nms(threshold=cfg.vehicle_nms_threshold, class_agnostic=True)
            else:
                detections = sv.Detections.empty()

            tracked = tracker.update_with_detections(detections)
            observations = [
                RawTrackObservation(
                    tracker_id=int(tracked.tracker_id[i]),
                    class_id=int(tracked.class_id[i]),
                    confidence=float(tracked.confidence[i]),
                    bbox=np.asarray(tracked.xyxy[i], dtype=np.float32),
                )
                for i in range(len(tracked))
                if tracked.tracker_id[i] is not None
            ]
            assignments = identity.assign_batch(
                frame=frame,
                observations=observations,
                image=image,
            )

            trusted: list[Detection] = []
            warming_up = warmup_remaining > 0
            warmup_remaining = max(0, warmup_remaining - 1)
            for raw_track, assignment in zip(observations, assignments):
                assessment = gate.update(
                    frame=frame,
                    track_id=assignment.canonical_id,
                    class_id=raw_track.class_id,
                    confidence=raw_track.confidence,
                    bbox=assignment.raw_bbox,
                    reidentified=assignment.lifecycle in {"STITCHED", "REACQUIRED"},
                    identity_confidence=assignment.identity_confidence,
                )
                if warming_up or assignment.provisional or not assessment.confirmed:
                    continue
                trusted.append(Detection(
                    frame=frame,
                    track_id=assignment.canonical_id,
                    raw_track_id=assignment.tracker_id,
                    identity_generation=assignment.generation,
                    identity_confidence=assignment.identity_confidence,
                    identity_lifecycle=assignment.lifecycle,
                    class_name=str(assessment.stable_class_id),
                    confidence=raw_track.confidence,
                    bbox=tuple(map(float, assignment.raw_bbox)),
                    filtered_bbox=tuple(map(float, assignment.filtered_bbox)),
                    track_quality=assessment.quality,
                    track_confirmed=True,
                ))

            gate.forget_stale(frame)
            identity.forget_stale(frame)
            events.extend(crash.update(frame, trusted, frame=image))
    finally:
        if cap is not None:
            cap.release()

    stitch_events.extend(identity.stitch_events)
    discontinuity_events.extend(identity.discontinuity_events)
    return {
        "events": [
            {
                "trigger_frame": event.trigger_frame,
                "detected_frame": event.detected_frame,
                "involved_tracks": event.involved_tracks,
                "score": round(event.score, 6),
                "evidence": event.evidence,
            }
            for event in events
        ],
        "scene_cuts": scene_cuts,
        "identity_stitches": stitch_events,
        "identity_discontinuities": discontinuity_events,
    }


def main() -> None:
    expected = json.loads((ROOT / "validation" / "expected.json").read_text())
    output = {}
    failures = []
    for name in ("crash", "crash2"):
        result = replay(
            ROOT / "validation" / "cached" / f"{name}_yolo640.json",
            ROOT / "validation" / "videos" / f"{name}.mp4",
        )
        output[name] = result
        events = result["events"]
        spec = expected[name]
        if len(events) != spec["expected_event_count"]:
            failures.append(
                f"{name}: expected {spec['expected_event_count']} event(s), got {len(events)}"
            )
            continue
        event = events[0]
        lo, hi = spec["trigger_frame_range"]
        if not lo <= event["trigger_frame"] <= hi:
            failures.append(f"{name}: trigger frame {event['trigger_frame']} outside {lo}-{hi}")
        lo, hi = spec["detected_frame_range"]
        if not lo <= event["detected_frame"] <= hi:
            failures.append(f"{name}: detected frame {event['detected_frame']} outside {lo}-{hi}")
        print(
            f"{name}: impact={event['trigger_frame']} detected={event['detected_frame']} "
            f"canonical_tracks={event['involved_tracks']} score={event['score']:.3f} "
            f"stitches={len(result['identity_stitches'])} cuts={result['scene_cuts']}"
        )

    result_path = ROOT / "validation" / "latest_cached_results.json"
    result_path.write_text(json.dumps(output, indent=2))
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS — results written to {result_path}")


if __name__ == "__main__":
    main()

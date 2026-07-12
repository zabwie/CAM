"""Render cached-model identity/crash validation videos with diagnostic IDs.

The large label is the canonical physical-vehicle ID used by analytics.  The
small ``raw`` value is the current ByteTrack handle.  A canonical ID should stay
constant even when ``raw`` changes after an occlusion.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack

from traffic_intel.config import SceneChangeConfig, TrackingConfig
from traffic_intel.crash_detector import CrashDetector
from traffic_intel.crash_visuals import (
    draw_crash_boxes,
    reset_crash_visuals,
    update_crash_visuals,
)
from traffic_intel.domain import Detection
from traffic_intel.identity import CanonicalIdentityManager, RawTrackObservation
from traffic_intel.scene import SceneChangeDetector
from traffic_intel.tracking import TrackQualityGate

VEHICLE_IDS = {2, 3, 5, 7}
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def _new_tracker(cfg: TrackingConfig, fps: float) -> ByteTrack:
    return ByteTrack(
        track_activation_threshold=cfg.activation_threshold,
        lost_track_buffer=max(1, int(round(cfg.lost_track_seconds * fps))),
        minimum_matching_threshold=cfg.minimum_matching_threshold,
        frame_rate=max(1, int(round(fps))),
    )


def _text_box(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.52,
    fg: tuple[int, int, int] = (255, 255, 255),
    bg: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thickness)
    h, w = frame.shape[:2]
    x = max(0, min(int(x), max(0, w - tw - 10)))
    y = max(th + base + 5, min(int(y), h - base - 2))
    cv2.rectangle(frame, (x, y - th - base - 5), (x + tw + 9, y + base + 2), bg, -1)
    cv2.putText(frame, text, (x + 4, y), font, scale, fg, thickness, cv2.LINE_AA)


def render(name: str, output: Path, timeline_path: Path) -> dict:
    cache_path = ROOT / "validation" / "cached" / f"{name}_yolo640.json"
    video_path = ROOT / "validation" / "videos" / f"{name}.mp4"
    cache = json.loads(cache_path.read_text())
    fps = float(cache["fps"])
    cfg = TrackingConfig()
    scene_cfg = SceneChangeConfig()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create {output}")

    tracker = _new_tracker(cfg, fps)
    identity = CanonicalIdentityManager(fps, cfg)
    gate = TrackQualityGate(fps, cfg)
    crash = CrashDetector(fps=fps)
    scene = SceneChangeDetector(scene_cfg)
    reset_crash_visuals()

    warmup_remaining = 0
    scene_epoch = 0
    timeline: list[dict] = []
    events: list[dict] = []
    all_stitches: list[dict] = []
    all_breaks: list[dict] = []
    active_reid_messages: deque[tuple[int, str]] = deque()
    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=24))
    previous_stitch_count = 0
    canonical_seen_raw: dict[int, set[int]] = defaultdict(set)
    duplicate_canonical_frames: list[int] = []
    scene_cuts: list[int] = []

    try:
        for payload in cache["frames"]:
            frame_no = int(payload["frame"])
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(f"Video ended before frame {frame_no}")

            cut = scene.update(frame)
            if cut:
                all_stitches.extend(identity.stitch_events)
                all_breaks.extend(identity.discontinuity_events)
                tracker = _new_tracker(cfg, fps)
                identity.reset(reset_counter=False)
                gate.reset()
                crash.reset()
                crash.set_fps(fps)
                reset_crash_visuals()
                trails.clear()
                previous_stitch_count = 0
                warmup_remaining = max(3, int(round(scene_cfg.warmup_seconds * fps)))
                scene_epoch += 1
                scene_cuts.append(frame_no)

            raw = [
                d for d in payload["dets"]
                if d["cls"] in VEHICLE_IDS and d["conf"] >= cfg.detector_confidence
            ]
            if raw:
                dets = sv.Detections(
                    xyxy=np.asarray([d["xyxy"] for d in raw], dtype=np.float32),
                    confidence=np.asarray([d["conf"] for d in raw], dtype=np.float32),
                    class_id=np.asarray([d["cls"] for d in raw], dtype=int),
                ).with_nms(threshold=cfg.vehicle_nms_threshold, class_agnostic=True)
            else:
                dets = sv.Detections.empty()

            tracked = tracker.update_with_detections(dets)
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
                frame=frame_no,
                observations=observations,
                image=frame,
            )

            if len(identity.stitch_events) > previous_stitch_count:
                for event in identity.stitch_events[previous_stitch_count:]:
                    active_reid_messages.append((
                        frame_no + int(round(1.2 * fps)),
                        (
                            f"RE-ID: ID {event['canonical_id']} kept | raw "
                            f"{event['old_tracker_id']} -> {event['new_tracker_id']} "
                            f"| score {event['score']:.2f}"
                        ),
                    ))
                previous_stitch_count = len(identity.stitch_events)

            trusted: list[Detection] = []
            warming_up = warmup_remaining > 0
            warmup_remaining = max(0, warmup_remaining - 1)
            for raw_track, assignment in zip(observations, assignments):
                assessment = gate.update(
                    frame=frame_no,
                    track_id=assignment.canonical_id,
                    class_id=raw_track.class_id,
                    confidence=raw_track.confidence,
                    bbox=assignment.raw_bbox,
                    reidentified=assignment.lifecycle in {"STITCHED", "REACQUIRED"},
                    identity_confidence=assignment.identity_confidence,
                )
                if warming_up or assignment.provisional or not assessment.confirmed:
                    continue
                det = Detection(
                    frame=frame_no,
                    track_id=assignment.canonical_id,
                    raw_track_id=assignment.tracker_id,
                    identity_generation=assignment.generation,
                    identity_confidence=assignment.identity_confidence,
                    identity_lifecycle=assignment.lifecycle,
                    class_name=CLASS_NAMES.get(assessment.stable_class_id, "vehicle"),
                    confidence=raw_track.confidence,
                    bbox=tuple(map(float, assignment.raw_bbox)),
                    filtered_bbox=tuple(map(float, assignment.filtered_bbox)),
                    track_quality=assessment.quality,
                    track_confirmed=True,
                )
                trusted.append(det)
                canonical_seen_raw[det.track_id].add(det.tracker_id)
                x1, y1, x2, y2 = map(int, det.display_bbox)
                trails[det.track_id].append(((x1 + x2) // 2, y2))
                timeline.append({
                    "frame": frame_no,
                    "scene_epoch": scene_epoch,
                    "canonical_id": det.track_id,
                    "raw_track_id": det.tracker_id,
                    "generation": det.identity_generation,
                    "identity_confidence": round(det.identity_confidence, 6),
                    "identity_lifecycle": det.identity_lifecycle,
                    "track_quality": round(det.track_quality, 6),
                    "bbox": [float(v) for v in det.bbox],
                    "filtered_bbox": [float(v) for v in det.display_bbox],
                })

            ids = [d.track_id for d in trusted]
            if len(ids) != len(set(ids)):
                duplicate_canonical_frames.append(frame_no)

            annotated = frame.copy()
            # Light trajectory tails make ID continuity visually obvious.
            for det in trusted:
                pts = np.asarray(trails[det.track_id], dtype=np.int32)
                if len(pts) >= 2:
                    cv2.polylines(annotated, [pts], False, (80, 220, 80), 2, cv2.LINE_AA)

                x1, y1, x2, y2 = map(int, det.display_bbox)
                color = (0, 215, 255) if det.identity_lifecycle == "STITCHED" else (0, 255, 0)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                main = f"ID {det.track_id}  {det.class_name}"
                sub = (
                    f"raw {det.tracker_id}  gen {det.identity_generation}  "
                    f"IDQ {det.identity_confidence:.2f}  TQ {det.track_quality:.2f}"
                )
                _text_box(annotated, main, x1, y1 - 22, scale=0.62, fg=color)
                _text_box(annotated, sub, x1, y1 - 2, scale=0.40, fg=(220, 220, 220))

            new_events = crash.update(frame_no, trusted, frame=frame)
            for event in new_events:
                draw_crash_boxes(annotated, event, trusted, frame_no)
                events.append({
                    "trigger_frame": event.trigger_frame,
                    "detected_frame": event.detected_frame,
                    "involved_tracks": event.involved_tracks,
                    "score": round(event.score, 6),
                })
            update_crash_visuals(annotated, trusted, frame_no)

            while active_reid_messages and active_reid_messages[0][0] < frame_no:
                active_reid_messages.popleft()
            y = 34
            for _until, message in list(active_reid_messages)[-3:]:
                _text_box(
                    annotated,
                    message,
                    14,
                    y,
                    scale=0.55,
                    fg=(0, 215, 255),
                    bg=(20, 20, 20),
                )
                y += 30

            if cut:
                _text_box(
                    annotated,
                    f"SCENE RESET | epoch {scene_epoch}",
                    14,
                    height - 44,
                    scale=0.66,
                    fg=(0, 255, 255),
                )
            _text_box(
                annotated,
                f"Canonical ID validation | frame {frame_no} | green ID = physical identity | raw = ByteTrack handle",
                12,
                height - 12,
                scale=0.44,
                fg=(255, 255, 255),
            )
            writer.write(annotated)

            gate.forget_stale(frame_no)
            identity.forget_stale(frame_no)
    finally:
        cap.release()
        writer.release()

    all_stitches.extend(identity.stitch_events)
    all_breaks.extend(identity.discontinuity_events)
    timeline_path.write_text(json.dumps({
        "video": name,
        "scene_cuts": scene_cuts,
        "events": events,
        "identity_stitches": all_stitches,
        "identity_discontinuities": all_breaks,
        "duplicate_canonical_frames": duplicate_canonical_frames,
        "canonical_to_raw_ids": {
            str(cid): sorted(raw_ids) for cid, raw_ids in canonical_seen_raw.items()
        },
        "timeline": timeline,
    }, indent=2))

    return {
        "video": name,
        "output": str(output),
        "frames": len(cache["frames"]),
        "scene_cuts": scene_cuts,
        "crash_events": events,
        "identity_stitches": all_stitches,
        "identity_discontinuities": all_breaks,
        "duplicate_canonical_frames": duplicate_canonical_frames,
        "canonical_ids_with_raw_changes": {
            str(cid): sorted(raw_ids)
            for cid, raw_ids in canonical_seen_raw.items()
            if len(raw_ids) > 1
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=["crash", "crash2"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeline", required=True)
    args = parser.parse_args()
    result = render(args.name, Path(args.output), Path(args.timeline))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

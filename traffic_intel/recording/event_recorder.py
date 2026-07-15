"""Rolling annotated-video and telemetry recorder for incident capture.

This module records operational incident packages.  It is intentionally kept
separate from any future evidentiary signing/chain-of-custody layer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import socket
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PRE_EVENT_SECONDS = 20.0
POST_EVENT_SECONDS = 10.0
SEGMENT_DURATION = 2.0


@dataclass(frozen=True, slots=True)
class EventRecorderConfig:
    pre_event_seconds: float = PRE_EVENT_SECONDS
    post_event_seconds: float = POST_EVENT_SECONDS
    segment_duration_seconds: float = SEGMENT_DURATION
    camera_id: Optional[str] = None


@dataclass(slots=True)
class TelemetryRow:
    frame: int
    track_id: int
    raw_track_id: int
    identity_generation: int
    identity_confidence: float
    identity_lifecycle: str
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    speed_mph: float
    speed_valid: bool
    invalid_reason: str
    capture_ts: float
    meas_confidence: float
    track_quality: float


@dataclass(slots=True)
class _Segment:
    video_path: Path
    telemetry_rows: list[TelemetryRow] = field(default_factory=list)
    first_frame: int = 0
    last_frame: int = 0
    frame_count: int = 0
    start_time: float = 0.0


class RollingSegmentBuffer:
    """Bounded pre-event ring plus post-trigger recording window.

    ``max_seconds`` is retained as a compatibility alias for
    ``pre_event_seconds``.  Segment filenames use a monotonic counter so an
    active event can never overwrite its own captured pre-event footage.
    """

    def __init__(
        self,
        fps: float,
        output_dir: str = "events",
        segment_duration: float = SEGMENT_DURATION,
        max_seconds: float = PRE_EVENT_SECONDS,
        post_event_seconds: float = POST_EVENT_SECONDS,
        *,
        config: EventRecorderConfig | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        cfg = config or EventRecorderConfig(
            pre_event_seconds=max_seconds,
            post_event_seconds=post_event_seconds,
            segment_duration_seconds=segment_duration,
        )
        self.fps = float(fps)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pre_event_seconds = max(0.0, float(cfg.pre_event_seconds))
        self.post_event_seconds = max(0.0, float(cfg.post_event_seconds))
        self.segment_duration = max(0.25, float(cfg.segment_duration_seconds))
        self.camera_id = cfg.camera_id or socket.gethostname()

        self.segment_frames = max(1, int(round(self.fps * self.segment_duration)))
        self.max_segments = max(
            1, int(np.ceil(max(self.pre_event_seconds, self.segment_duration) / self.segment_duration))
        )
        self._tmpdir = Path(tempfile.mkdtemp(prefix="event_buffer_"))
        self._segments: deque[_Segment] = deque(maxlen=self.max_segments)
        self._cur: Optional[_Segment] = None
        self._writer: Optional[cv2.VideoWriter] = None
        self._frame_w = 0
        self._frame_h = 0
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._segment_counter = 0

        self._post_trigger: Optional[dict] = None
        self._post_frames_remaining = 0
        self.last_saved_event: Optional[Path] = None

    def write_frame(self, annotated: np.ndarray, detections: list, frame_count: int) -> None:
        if annotated is None or annotated.size == 0:
            raise ValueError("annotated frame must be non-empty")
        self._init_encoding(annotated)

        height, width = annotated.shape[:2]
        if (width, height) != (self._frame_w, self._frame_h):
            self._close_segment()
            self._frame_w, self._frame_h = width, height

        if self._cur is None or self._cur.frame_count >= self.segment_frames:
            self._start_segment()
        assert self._cur is not None

        if self._writer is not None:
            self._writer.write(annotated)

        self._cur.frame_count += 1
        if self._cur.frame_count == 1:
            self._cur.first_frame = frame_count
            self._cur.start_time = frame_count / self.fps
        self._cur.last_frame = frame_count

        for d in detections:
            if d.frame != frame_count:
                continue
            self._cur.telemetry_rows.append(TelemetryRow(
                frame=int(d.frame),
                track_id=int(d.track_id),
                raw_track_id=int(getattr(d, "raw_track_id", d.track_id) or d.track_id),
                identity_generation=int(getattr(d, "identity_generation", 1)),
                identity_confidence=float(getattr(d, "identity_confidence", 1.0)),
                identity_lifecycle=str(getattr(d, "identity_lifecycle", "CONTINUING")),
                class_name=str(d.class_name),
                confidence=float(d.confidence),
                x1=int(d.bbox[0]), y1=int(d.bbox[1]),
                x2=int(d.bbox[2]), y2=int(d.bbox[3]),
                speed_mph=float(d.speed),
                speed_valid=bool(d.speed_valid),
                invalid_reason=str(d.invalid_reason),
                capture_ts=float(d.capture_timestamp),
                meas_confidence=float(d.measurement_confidence),
                track_quality=float(getattr(d, "track_quality", 1.0)),
            ))

        if self._post_trigger is not None:
            self._post_frames_remaining -= 1
            if self._post_frames_remaining <= 0:
                self._finalize_event()

    def trigger_manual(
        self,
        engine=None,
        *,
        event_metadata: Optional[dict] = None,
    ) -> Optional[Path]:
        return self._begin_event("manual", engine, event_metadata=event_metadata)

    def trigger_auto(
        self,
        reason: str,
        engine=None,
        *,
        trigger_frame: Optional[int] = None,
        event_metadata: Optional[dict] = None,
    ) -> Optional[Path]:
        return self._begin_event(
            f"auto_{reason}",
            engine,
            trigger_frame=trigger_frame,
            event_metadata=event_metadata,
        )

    def close(self) -> None:
        if self._post_trigger is not None:
            self._finalize_event()
        self._close_segment()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _init_encoding(self, frame: np.ndarray) -> None:
        if self._frame_w == 0:
            height, width = frame.shape[:2]
            self._frame_w, self._frame_h = width, height

    def _start_segment(self) -> None:
        self._close_segment()
        path = self._tmpdir / f"seg_{self._segment_counter:08d}.mp4"
        self._segment_counter += 1
        writer = cv2.VideoWriter(
            str(path), self._fourcc, self.fps, (self._frame_w, self._frame_h)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create event-buffer segment: {path}")
        self._writer = writer
        self._cur = _Segment(video_path=path)

    def _close_segment(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._cur is not None and self._cur.frame_count > 0:
            self._segments.append(self._cur)
        self._cur = None

    def _begin_event(
        self,
        trigger_type: str,
        engine,
        *,
        trigger_frame: Optional[int] = None,
        event_metadata: Optional[dict] = None,
    ) -> Optional[Path]:
        if self._post_trigger is not None:
            return None

        self._close_segment()
        if not self._segments:
            return None

        detected_frame = int(getattr(engine, "frame_count", 0)) if engine else 0
        self._post_trigger = {
            "trigger_type": trigger_type,
            "trigger_frame": int(trigger_frame) if trigger_frame is not None else detected_frame,
            "detected_frame": detected_frame,
            "event_metadata": dict(event_metadata or {}),
            "trigger_time": time.time(),
            "segments": list(self._segments),
            "num_pre_segments": len(self._segments),
            "event_id": str(uuid.uuid4()),
        }
        self._post_frames_remaining = max(1, int(round(self.post_event_seconds * self.fps)))
        self._segments.clear()
        print(
            f"\n  Event triggered ({trigger_type}) — "
            f"recording {self.post_event_seconds:g}s post-event..."
        )
        return None

    def _finalize_event(self) -> None:
        trigger = self._post_trigger
        if trigger is None:
            return
        self._post_trigger = None
        self._close_segment()

        all_segments: list[_Segment] = list(trigger["segments"]) + list(self._segments)
        self._segments.clear()
        if not all_segments:
            return

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        short_id = trigger["event_id"].split("-")[0]
        event_dir = self.output_dir / f"{timestamp}_{short_id}_{trigger['trigger_type']}"
        event_dir.mkdir(parents=True, exist_ok=False)
        clip_path = event_dir / "clip.mp4"
        telemetry_path = event_dir / "telemetry.csv"
        json_path = event_dir / "event.json"

        telemetry_rows: list[TelemetryRow] = []
        sources: list[str] = []
        for segment in all_segments:
            if segment.video_path.exists() and segment.video_path.stat().st_size > 0:
                sources.append(str(segment.video_path))
            telemetry_rows.extend(segment.telemetry_rows)

        if sources:
            self._concat_videos(sources, str(clip_path))
        self._write_telemetry(telemetry_path, telemetry_rows)

        metadata = {
            "event_id": trigger["event_id"],
            "camera_id": self.camera_id,
            "trigger_time_unix": trigger["trigger_time"],
            "trigger_type": trigger["trigger_type"],
            "trigger_frame": trigger["trigger_frame"],
            "detected_frame": trigger["detected_frame"],
            "pre_event_seconds": self.pre_event_seconds,
            "post_event_seconds": self.post_event_seconds,
            "segment_duration_seconds": self.segment_duration,
            "segments_pre": trigger["num_pre_segments"],
            "segments_post": len(all_segments) - trigger["num_pre_segments"],
            "total_frames": sum(s.frame_count for s in all_segments),
            "total_detections": len(telemetry_rows),
        }
        metadata.update(trigger["event_metadata"])
        if clip_path.exists():
            digest = self._sha256(clip_path)
            (event_dir / "clip.sha256").write_text(digest + "\n")
            metadata["video_sha256"] = digest
        json_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
        self.last_saved_event = event_dir
        print(
            f"\n  Event saved: {event_dir}\n"
            f"     Frames: {metadata['total_frames']}  "
            f"Detections: {metadata['total_detections']}"
        )

    @staticmethod
    def _write_telemetry(path: Path, rows: list[TelemetryRow]) -> None:
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "frame", "track_id", "raw_track_id", "identity_generation",
                "identity_confidence", "identity_lifecycle", "class", "confidence",
                "x1", "y1", "x2", "y2", "speed_mph", "speed_valid",
                "invalid_reason", "capture_ts", "meas_confidence", "track_quality",
            ])
            for r in rows:
                writer.writerow([
                    r.frame, r.track_id, r.raw_track_id, r.identity_generation,
                    f"{r.identity_confidence:.3f}", r.identity_lifecycle,
                    r.class_name, f"{r.confidence:.3f}",
                    r.x1, r.y1, r.x2, r.y2, f"{r.speed_mph:.1f}",
                    int(r.speed_valid), r.invalid_reason, f"{r.capture_ts:.3f}",
                    f"{r.meas_confidence:.3f}", f"{r.track_quality:.3f}",
                ])

    def _concat_videos(self, sources: list[str], output: str) -> None:
        cap = cv2.VideoCapture(sources[0])
        fps = float(cap.get(cv2.CAP_PROP_FPS) or self.fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        writer = cv2.VideoWriter(
            output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create event clip: {output}")
        try:
            for source in sources:
                cap = cv2.VideoCapture(source)
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        writer.write(frame)
                finally:
                    cap.release()
        finally:
            writer.release()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

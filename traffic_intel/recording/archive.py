"""Continuous surveillance archive writer.

The archive is intentionally separate from incident evidence capture:
- every incoming raw frame can be written into time-bounded MP4 segments;
- each segment receives a JSON sidecar with place/camera/time metadata;
- each completed segment receives a SHA-256 digest;
- an optional retention window removes expired continuous footage.

Incident packages remain managed by :mod:`traffic_intel.event_recorder`.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class ArchiveRecorderConfig:
    output_dir: str = "archive"
    segment_seconds: float = 300.0
    retention_days: int = 30
    municipality: str = ""
    location_name: str = ""
    camera_name: str = ""
    source_label: str = ""


class ContinuousArchiveRecorder:
    """Write raw surveillance footage into searchable, bounded MP4 segments."""

    def __init__(self, fps: float, config: ArchiveRecorderConfig) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.segment_seconds = max(5.0, float(config.segment_seconds))
        self.segment_frames = max(1, int(round(self.segment_seconds * self.fps)))
        self.retention_days = max(0, int(config.retention_days))

        self._writer: cv2.VideoWriter | None = None
        self._video_path: Path | None = None
        self._frame_count = 0
        self._width = 0
        self._height = 0
        self._start_time_unix = 0.0
        self._last_time_unix = 0.0
        self.last_saved_segment: Path | None = None
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self._cleanup_expired()

    def write_frame(
        self,
        frame: np.ndarray,
        *,
        frame_number: int | None = None,
        timestamp: float | None = None,
    ) -> None:
        if frame is None or frame.size == 0:
            raise ValueError("frame must be non-empty")
        ts = float(timestamp if timestamp is not None else time.time())
        height, width = frame.shape[:2]

        if self._writer is None:
            self._start_segment(width, height, ts)
        elif (width, height) != (self._width, self._height):
            self._finalize_segment()
            self._start_segment(width, height, ts)

        assert self._writer is not None
        self._writer.write(frame)
        self._frame_count += 1
        self._last_time_unix = ts

        if self._frame_count >= self.segment_frames:
            self._finalize_segment()

    def close(self) -> None:
        self._finalize_segment()

    def _start_segment(self, width: int, height: int, ts: float) -> None:
        dt = time.localtime(ts)
        day_dir = self.output_dir / time.strftime("%Y-%m-%d", dt)
        day_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d_%H-%M-%S", dt)
        short_id = uuid.uuid4().hex[:8]
        self._video_path = day_dir / f"{stamp}_{short_id}.mp4"
        writer = cv2.VideoWriter(
            str(self._video_path), self._fourcc, self.fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot create archive segment: {self._video_path}")
        self._writer = writer
        self._frame_count = 0
        self._width = width
        self._height = height
        self._start_time_unix = ts
        self._last_time_unix = ts

    def _finalize_segment(self) -> None:
        if self._writer is None or self._video_path is None:
            return

        self._writer.release()
        self._writer = None
        video_path = self._video_path
        self._video_path = None

        if self._frame_count <= 0 or not video_path.exists():
            video_path.unlink(missing_ok=True)
            return

        digest = self._sha256(video_path)
        metadata = {
            "recording_id": video_path.stem,
            "type": "surveillance",
            "municipality": self.config.municipality,
            "location": self.config.location_name,
            "camera": self.config.camera_name,
            "source": self.config.source_label,
            "start_time_unix": self._start_time_unix,
            "end_time_unix": self._last_time_unix,
            "duration_seconds": max(0.0, self._last_time_unix - self._start_time_unix),
            "fps": self.fps,
            "width": self._width,
            "height": self._height,
            "frames": self._frame_count,
            "video_sha256": digest,
            "video_file": video_path.name,
        }
        video_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        video_path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
        self.last_saved_segment = video_path
        self._cleanup_expired()

    def _cleanup_expired(self) -> None:
        if self.retention_days <= 0 or not self.output_dir.exists():
            return
        cutoff = time.time() - (self.retention_days * 86400)
        for video_path in self.output_dir.rglob("*.mp4"):
            try:
                if video_path.stat().st_mtime >= cutoff:
                    continue
            except OSError:
                continue
            for sidecar in (
                video_path,
                video_path.with_suffix(".json"),
                video_path.with_suffix(".sha256"),
            ):
                sidecar.unlink(missing_ok=True)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

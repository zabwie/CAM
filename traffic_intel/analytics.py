"""Downstream pilot analytics built from trusted canonical detections.

Per-frame detections are intentionally not treated as independent vehicles.
``VehiclePassageAggregator`` produces one durable record per canonical vehicle,
so a slow car visible for 300 frames does not outweigh a fast car visible for
30 frames in municipal statistics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable
from uuid import uuid4

import numpy as np

from .domain import Detection
from .vision_quality import VisionQualitySample


@dataclass(frozen=True, slots=True)
class VehiclePassageAggregatorConfig:
    finalization_gap_seconds: float = 1.75
    min_valid_speed_samples: int = 3
    min_measurement_confidence: float = 0.50


@dataclass(frozen=True, slots=True)
class VehiclePassage:
    passage_id: str
    session_id: str
    municipality: str
    location_id: str
    camera_id: str
    canonical_track_id: int
    vehicle_class: str
    first_seen_at: float
    last_seen_at: float
    observed_seconds: float
    valid_speed_samples: int
    representative_speed_mph: float | None
    max_speed_mph: float | None
    measurement_confidence: float
    speed_limit_mph: float | None
    speeding: bool | None
    vision_state: str
    calibration_id: str
    software_version: str


@dataclass(slots=True)
class _ActivePassage:
    canonical_track_id: int
    first_seen_at: float
    last_seen_at: float
    first_monotonic: float
    last_monotonic: float
    class_votes: Counter[str] = field(default_factory=Counter)
    vision_votes: Counter[str] = field(default_factory=Counter)
    speed_samples: list[float] = field(default_factory=list)
    confidence_samples: list[float] = field(default_factory=list)


class VehiclePassageAggregator:
    """Convert a trusted detection stream into one row per physical vehicle."""

    def __init__(
        self,
        *,
        camera_id: str,
        municipality: str = "",
        location_id: str = "",
        speed_limit_mph: float | None = None,
        calibration_id: str = "",
        software_version: str = "",
        session_id: str | None = None,
        config: VehiclePassageAggregatorConfig | None = None,
    ) -> None:
        self.camera_id = str(camera_id)
        self.municipality = str(municipality)
        self.location_id = str(location_id)
        self.speed_limit_mph = (
            None if speed_limit_mph is None else float(speed_limit_mph)
        )
        self.calibration_id = str(calibration_id)
        self.software_version = str(software_version)
        self.session_id = session_id or uuid4().hex
        self.config = config or VehiclePassageAggregatorConfig()
        self._active: dict[int, _ActivePassage] = {}

    @property
    def active_count(self) -> int:
        return len(self._active)

    def update(
        self,
        detections: Iterable[Detection],
        *,
        capture_timestamp: float,
        monotonic_timestamp: float,
        vision_state: str = "UNKNOWN",
    ) -> list[VehiclePassage]:
        """Ingest one analyzed frame and return passages finalized this frame."""
        capture_timestamp = float(capture_timestamp)
        monotonic_timestamp = float(monotonic_timestamp)
        finalized = self.finalize_stale(monotonic_timestamp)

        for detection in detections:
            tid = int(detection.track_id)
            observed_at = (
                float(detection.capture_timestamp)
                if detection.capture_timestamp > 0
                else capture_timestamp
            )
            observed_monotonic = (
                float(detection.monotonic_timestamp)
                if detection.monotonic_timestamp > 0
                else monotonic_timestamp
            )
            state = self._active.get(tid)
            if state is None:
                state = _ActivePassage(
                    canonical_track_id=tid,
                    first_seen_at=observed_at,
                    last_seen_at=observed_at,
                    first_monotonic=observed_monotonic,
                    last_monotonic=observed_monotonic,
                )
                self._active[tid] = state

            state.last_seen_at = max(state.last_seen_at, observed_at)
            state.last_monotonic = max(state.last_monotonic, observed_monotonic)
            state.class_votes[str(detection.class_name or "vehicle")] += 1
            state.vision_votes[str(detection.vision_state or vision_state or "UNKNOWN")] += 1

            if (
                detection.speed_valid
                and np.isfinite(detection.speed)
                and detection.measurement_confidence >= self.config.min_measurement_confidence
            ):
                state.speed_samples.append(float(detection.speed))
                state.confidence_samples.append(float(detection.measurement_confidence))

        return finalized

    def finalize_stale(self, current_monotonic: float) -> list[VehiclePassage]:
        stale_ids = [
            tid
            for tid, state in self._active.items()
            if current_monotonic - state.last_monotonic
            > self.config.finalization_gap_seconds
        ]
        return [self._finalize(tid) for tid in stale_ids]

    def flush(self) -> list[VehiclePassage]:
        return [self._finalize(tid) for tid in list(self._active)]

    def _finalize(self, track_id: int) -> VehiclePassage:
        state = self._active.pop(track_id)
        valid_samples = len(state.speed_samples)
        has_speed = valid_samples >= self.config.min_valid_speed_samples
        representative = (
            float(median(state.speed_samples)) if has_speed else None
        )
        maximum = float(max(state.speed_samples)) if has_speed else None
        confidence = (
            float(median(state.confidence_samples))
            if state.confidence_samples
            else 0.0
        )
        speeding = (
            None
            if representative is None or self.speed_limit_mph is None
            else representative > self.speed_limit_mph
        )
        vehicle_class = (
            state.class_votes.most_common(1)[0][0]
            if state.class_votes
            else "vehicle"
        )
        vision_state = (
            state.vision_votes.most_common(1)[0][0]
            if state.vision_votes
            else "UNKNOWN"
        )
        return VehiclePassage(
            passage_id=uuid4().hex,
            session_id=self.session_id,
            municipality=self.municipality,
            location_id=self.location_id,
            camera_id=self.camera_id,
            canonical_track_id=state.canonical_track_id,
            vehicle_class=vehicle_class,
            first_seen_at=state.first_seen_at,
            last_seen_at=state.last_seen_at,
            observed_seconds=max(0.0, state.last_monotonic - state.first_monotonic),
            valid_speed_samples=valid_samples,
            representative_speed_mph=representative,
            max_speed_mph=maximum,
            measurement_confidence=confidence,
            speed_limit_mph=self.speed_limit_mph,
            speeding=speeding,
            vision_state=vision_state,
            calibration_id=self.calibration_id,
            software_version=self.software_version,
        )


@dataclass(frozen=True, slots=True)
class CameraHealthRecord:
    camera_id: str
    bucket_start: float
    bucket_end: float
    frames_received: int
    frames_analyzed: int
    frame_gap_count: int
    analysis_fps: float
    detections: int
    valid_speed_detections: int
    valid_speed_rate: float
    median_brightness: float
    dark_fraction: float
    clipped_highlight_fraction: float
    median_saturation: float
    blur_score: float
    vision_state: str


class CameraHealthAccumulator:
    """Aggregate explainable camera/runtime health into fixed time buckets."""

    def __init__(self, camera_id: str, bucket_seconds: float = 60.0) -> None:
        if bucket_seconds <= 0:
            raise ValueError("bucket_seconds must be positive")
        self.camera_id = str(camera_id)
        self.bucket_seconds = float(bucket_seconds)
        self._reset()

    def _reset(self) -> None:
        self._bucket_start: float | None = None
        self._last_monotonic: float | None = None
        self._frames_received = 0
        self._frames_analyzed = 0
        self._frame_gap_count = 0
        self._detections = 0
        self._valid_speed_detections = 0
        self._samples: list[VisionQualitySample] = []
        self._states: Counter[str] = Counter()

    def update(
        self,
        *,
        capture_timestamp: float,
        monotonic_timestamp: float,
        sequence_gap: int,
        detections: Iterable[Detection],
        quality: VisionQualitySample,
    ) -> list[CameraHealthRecord]:
        records: list[CameraHealthRecord] = []
        capture_timestamp = float(capture_timestamp)
        if self._bucket_start is None:
            self._bucket_start = capture_timestamp - (capture_timestamp % self.bucket_seconds)
        while capture_timestamp >= self._bucket_start + self.bucket_seconds:
            records.append(self._finalize(self._bucket_start + self.bucket_seconds))
            self._bucket_start += self.bucket_seconds

        dets = list(detections)
        self._frames_received += max(1, int(sequence_gap) + 1)
        self._frames_analyzed += 1
        self._frame_gap_count += max(0, int(sequence_gap))
        self._detections += len(dets)
        self._valid_speed_detections += sum(1 for d in dets if d.speed_valid)
        self._samples.append(quality)
        self._states[quality.state] += 1
        if self._last_monotonic is None:
            self._last_monotonic = float(monotonic_timestamp)
        return records

    def flush(self, end_timestamp: float | None = None) -> CameraHealthRecord | None:
        if self._bucket_start is None or self._frames_analyzed == 0:
            return None
        end = (
            float(end_timestamp)
            if end_timestamp is not None
            else self._bucket_start + self.bucket_seconds
        )
        record = self._finalize(end)
        self._reset()
        return record

    def _finalize(self, bucket_end: float) -> CameraHealthRecord:
        start = float(self._bucket_start or bucket_end - self.bucket_seconds)
        elapsed = max(bucket_end - start, 1e-6)
        samples = self._samples

        def med(attr: str) -> float:
            if not samples:
                return 0.0
            return float(median(getattr(sample, attr) for sample in samples))

        record = CameraHealthRecord(
            camera_id=self.camera_id,
            bucket_start=start,
            bucket_end=float(bucket_end),
            frames_received=self._frames_received,
            frames_analyzed=self._frames_analyzed,
            frame_gap_count=self._frame_gap_count,
            analysis_fps=self._frames_analyzed / elapsed,
            detections=self._detections,
            valid_speed_detections=self._valid_speed_detections,
            valid_speed_rate=(
                self._valid_speed_detections / self._detections
                if self._detections
                else 0.0
            ),
            median_brightness=med("median_brightness"),
            dark_fraction=med("dark_fraction"),
            clipped_highlight_fraction=med("clipped_highlight_fraction"),
            median_saturation=med("median_saturation"),
            blur_score=med("blur_score"),
            vision_state=(
                self._states.most_common(1)[0][0]
                if self._states
                else "UNKNOWN"
            ),
        )
        # Prepare next bucket while preserving the caller-managed start value.
        self._frames_received = 0
        self._frames_analyzed = 0
        self._frame_gap_count = 0
        self._detections = 0
        self._valid_speed_detections = 0
        self._samples = []
        self._states = Counter()
        return record


__all__ = [
    "CameraHealthAccumulator",
    "CameraHealthRecord",
    "VehiclePassage",
    "VehiclePassageAggregator",
    "VehiclePassageAggregatorConfig",
]

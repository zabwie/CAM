"""High-level incident pipeline composed from independent subsystems."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..domain import Detection
from ..incident.crash_detector import (
    CrashCandidate,
    CrashDetector,
    draw_crash_boxes,
    reset_crash_visuals,
    update_crash_visuals,
)
from .engine import TrafficEngine


@dataclass(slots=True)
class PipelineFrame:
    frame_number: int
    annotated: np.ndarray
    detections: list[Detection]
    crashes: list[CrashCandidate]
    scene_cut: bool = False


class TrafficIncidentPipeline:
    """Compose perception and crash analysis without merging their state.

    ``TrafficEngine`` owns detection/tracking/speed. ``CrashDetector`` owns only
    incident semantics.  This coordinator handles source resets and annotation,
    giving live, validation, and future service adapters one canonical path.
    """

    def __init__(
        self,
        engine: TrafficEngine,
        crash_detector: CrashDetector | None = None,
        *,
        draw_crash_annotations: bool = True,
    ) -> None:
        self.engine = engine
        self.crash_detector = crash_detector or CrashDetector(fps=engine.fps)
        self.draw_crash_annotations = bool(draw_crash_annotations)
        reset_crash_visuals()

    def process_frame(
        self,
        frame: np.ndarray,
        *,
        optical_flow: bool = True,
        capture_timestamp: float | None = None,
        monotonic_timestamp: float | None = None,
        vision_state: str = "UNKNOWN",
    ) -> PipelineFrame:
        annotated = self.engine.process_frame(
            frame,
            capture_timestamp=capture_timestamp,
            monotonic_timestamp=monotonic_timestamp,
            vision_state=vision_state,
        )
        if self.engine.last_scene_cut:
            self.crash_detector.reset()
            self.crash_detector.set_fps(self.engine.fps)
            reset_crash_visuals()

        detections = list(self.engine.current_detections)
        crashes = self.crash_detector.update(
            self.engine.frame_count,
            detections,
            frame=frame if optical_flow else None,
        )
        if self.draw_crash_annotations:
            for candidate in crashes:
                draw_crash_boxes(
                    annotated,
                    candidate,
                    detections,
                    self.engine.frame_count,
                )
            update_crash_visuals(annotated, detections, self.engine.frame_count)

        return PipelineFrame(
            frame_number=self.engine.frame_count,
            annotated=annotated,
            detections=detections,
            crashes=crashes,
            scene_cut=self.engine.last_scene_cut,
        )

    def reset(self) -> None:
        self.engine.reset()
        self.crash_detector.reset()
        self.crash_detector.set_fps(self.engine.fps)
        reset_crash_visuals()

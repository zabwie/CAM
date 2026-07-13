"""Temporal quality gate for ByteTrack outputs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from traffic_intel.config import TrackingConfig


@dataclass
class _TrackQualityState:
    first_frame: int
    last_frame: int
    hit_streak: int = 0
    total_hits: int = 0
    conf_ema: float = 0.0
    class_scores: dict[int, float] = field(default_factory=lambda: defaultdict(float))
    last_bbox: np.ndarray | None = None
    confirmed: bool = False
    reacquire_remaining: int = 0
    instability: float = 0.0


@dataclass(frozen=True, slots=True)
class TrackAssessment:
    quality: float
    confirmed: bool
    stable_class_id: int
    instability: float


class TrackQualityGate:
    """Separates tracker continuity from downstream analytical trust.

    ByteTrack can deliberately retain weak detections to bridge occlusions.  The
    analytics layer should not treat every tracker output as equally reliable.
    This gate scores temporal maturity, detector confidence, geometry stability,
    and class consistency before a track is emitted.
    """

    def __init__(self, fps: float, config: TrackingConfig | None = None) -> None:
        self.fps = float(fps or 30.0)
        self.config = config or TrackingConfig()
        self._states: dict[int, _TrackQualityState] = {}

    def reset(self) -> None:
        self._states.clear()

    def set_fps(self, fps: float) -> None:
        self.fps = float(fps or 30.0)

    def update(
        self,
        *,
        frame: int,
        track_id: int,
        class_id: int,
        confidence: float,
        bbox: np.ndarray,
        reidentified: bool = False,
        identity_confidence: float = 1.0,
    ) -> TrackAssessment:
        cfg = self.config
        box = np.asarray(bbox, dtype=np.float32)
        state = self._states.get(track_id)

        if state is None:
            state = _TrackQualityState(
                first_frame=frame,
                last_frame=frame,
                hit_streak=1,
                total_hits=1,
                conf_ema=float(confidence),
                last_bbox=box.copy(),
            )
            state.class_scores[class_id] += float(confidence)
            self._states[track_id] = state
        else:
            gap = frame - state.last_frame
            state.hit_streak = state.hit_streak + 1 if gap <= 2 else 1
            state.total_hits += 1

            jump_norm = 0.0
            size_ratio = 1.0
            if state.last_bbox is not None and gap <= 2:
                old = state.last_bbox
                old_center = np.array([(old[0] + old[2]) * 0.5, (old[1] + old[3]) * 0.5])
                new_center = np.array([(box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5])
                old_w, old_h = max(1.0, old[2] - old[0]), max(1.0, old[3] - old[1])
                new_w, new_h = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
                old_diag = max(10.0, float(np.hypot(old_w, old_h)))
                jump_norm = float(np.linalg.norm(new_center - old_center)) / old_diag
                size_ratio = max(new_w * new_h, old_w * old_h) / max(1.0, min(new_w * new_h, old_w * old_h))

            long_gap = gap > 2
            trusted_reidentification = bool(
                reidentified
                and identity_confidence >= cfg.identity_min_stitch_score
                and gap <= max(2, int(round(cfg.identity_stitch_seconds * self.fps)))
            )
            unstable = (
                jump_norm > cfg.max_normalized_jump
                or size_ratio > cfg.max_size_ratio_step
                or (long_gap and not trusted_reidentification)
            )
            if unstable:
                state.hit_streak = 1
                state.confirmed = False
                state.reacquire_remaining = max(state.reacquire_remaining, cfg.reacquire_hits)
                state.instability = min(1.0, state.instability + 0.55)
            elif trusted_reidentification:
                # Canonical identity has already performed a stricter,
                # multi-cue re-identification.  Preserve analytical maturity so
                # the vehicle returns immediately with the same canonical ID.
                state.hit_streak = max(state.hit_streak, cfg.min_hits + 1)
                state.reacquire_remaining = 0
                state.instability = min(0.55, state.instability * 0.70 + 0.08)
            else:
                state.instability *= 0.82

            state.conf_ema = 0.72 * state.conf_ema + 0.28 * float(confidence)
            for key in list(state.class_scores):
                state.class_scores[key] *= 0.94
                if state.class_scores[key] < 1e-4:
                    del state.class_scores[key]
            state.class_scores[class_id] += float(confidence)
            state.last_frame = frame
            state.last_bbox = box.copy()

            if state.reacquire_remaining > 0:
                state.reacquire_remaining -= 1

        stable_class = max(state.class_scores, key=state.class_scores.get)
        class_total = sum(state.class_scores.values()) or 1.0
        class_consistency = state.class_scores[stable_class] / class_total
        maturity = min(state.hit_streak / max(cfg.min_hits + 2, 1), 1.0)
        stability = 1.0 - state.instability
        quality = float(np.clip(
            0.52 * state.conf_ema
            + 0.22 * maturity
            + 0.16 * class_consistency
            + 0.10 * stability,
            0.0,
            1.0,
        ))

        state.confirmed = bool(
            state.hit_streak >= cfg.min_hits
            and state.conf_ema >= cfg.min_confidence
            and state.reacquire_remaining <= 0
            and state.instability < 0.70
        )
        return TrackAssessment(
            quality=quality,
            confirmed=state.confirmed,
            stable_class_id=int(stable_class),
            instability=float(state.instability),
        )

    def forget_stale(self, current_frame: int) -> None:
        stale_after = max(15, int(round(self.config.stale_seconds * self.fps)))
        stale = [
            tid for tid, state in self._states.items()
            if current_frame - state.last_frame > stale_after
        ]
        for tid in stale:
            del self._states[tid]

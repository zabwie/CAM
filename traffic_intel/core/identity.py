"""Canonical vehicle identity across short tracker fragmentation.

ByteTrack IDs are association handles, not durable physical identities.  This
module sits between the tracker and downstream analytics and provides a stable
canonical ID that can survive a short occlusion or a tracker re-initialisation.

The matcher is deliberately conservative.  A new raw tracker ID is stitched to
an existing canonical vehicle only when motion/scale are plausible and, when
available, the vehicle appearance is also compatible.  Ambiguous matches are
left as new identities rather than guessed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from math import exp, log
from typing import Iterable

import cv2
import numpy as np

from traffic_intel.config import TrackingConfig
from traffic_intel.core.appearance import appearance_descriptor, appearance_similarity, normalise_hist
from traffic_intel.core.geometry import box_center_size, center_size_box, robust_velocity, size_ratio


@dataclass(frozen=True, slots=True)
class RawTrackObservation:
    """One tracker output before canonical identity assignment."""

    tracker_id: int
    class_id: int
    confidence: float
    bbox: np.ndarray


@dataclass(frozen=True, slots=True)
class IdentityAssignment:
    """Canonical identity assigned to one raw tracker observation."""

    canonical_id: int
    tracker_id: int
    generation: int
    identity_confidence: float
    lifecycle: str
    raw_bbox: np.ndarray
    filtered_bbox: np.ndarray
    provisional: bool = False
    stitched: bool = False
    discontinuity: bool = False


@dataclass
class _IdentityState:
    canonical_id: int
    first_frame: int
    last_frame: int
    current_tracker_id: int
    generation: int = 1
    history: deque[tuple[int, float, float, float, float]] = field(
        default_factory=lambda: deque(maxlen=24)
    )
    class_scores: dict[int, float] = field(default_factory=dict)
    appearance: np.ndarray | None = None
    appearance_samples: int = 0
    filtered_center: np.ndarray | None = None
    filtered_size: np.ndarray | None = None
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    last_identity_confidence: float = 1.0


class CanonicalIdentityManager:
    """Map ephemeral tracker IDs to stable per-camera canonical vehicle IDs."""

    def __init__(self, fps: float, config: TrackingConfig | None = None) -> None:
        self.fps = float(fps or 30.0)
        self.config = config or TrackingConfig()
        self._states: dict[int, _IdentityState] = {}
        self._raw_to_canonical: dict[int, int] = {}
        self._next_canonical_id = 1
        self._last_frame = 0
        self.stitch_events: list[dict[str, float | int]] = []
        self.discontinuity_events: list[dict[str, float | int]] = []

    def reset(self, *, reset_counter: bool = True) -> None:
        self._states.clear()
        self._raw_to_canonical.clear()
        if reset_counter:
            self._next_canonical_id = 1
        self._last_frame = 0
        self.stitch_events.clear()
        self.discontinuity_events.clear()

    def set_fps(self, fps: float) -> None:
        self.fps = float(fps or 30.0)

    def assign_batch(
        self,
        *,
        frame: int,
        observations: Iterable[RawTrackObservation],
        image: np.ndarray | None = None,
    ) -> list[IdentityAssignment]:
        """Assign canonical IDs to all tracker outputs in one frame.

        Matching is done as a batch so two new raw tracks cannot claim the same
        dormant canonical vehicle.  Existing healthy raw-ID mappings are kept
        first; genuinely new/reassigned raw IDs are then matched conservatively
        against recently lost canonical tracks.
        """
        obs = list(observations)
        self._last_frame = max(self._last_frame, int(frame))
        if not obs:
            self.forget_stale(frame)
            return []

        descriptors = {
            o.tracker_id: appearance_descriptor(image, o.bbox)
            for o in obs
        }

        assignments: dict[int, IdentityAssignment] = {}
        used_canonical: set[int] = set()
        # (observation, came_from_continuity_break, provisional_canonical_id)
        unmatched: list[tuple[RawTrackObservation, bool, int | None]] = []

        # First preserve existing raw->canonical mappings when continuity is
        # plausible.  A severe jump can indicate an ID hijack; in that case the
        # raw ID is detached and treated like a new observation below.
        for o in obs:
            cid = self._raw_to_canonical.get(o.tracker_id)
            if cid is None or cid not in self._states:
                unmatched.append((o, False, None))
                continue
            state = self._states[cid]
            # A newly created canonical ID is provisional until the raw track
            # has survived a few frames.  Re-evaluate it against older dormant
            # identities so a true reappearance can be stitched once the
            # minimum gap becomes observable, before analytics ever trust the
            # provisional ID.
            provisional_hits = max(2, int(self.config.identity_provisional_hits))
            if (
                state.generation == 1
                and frame - state.first_frame < provisional_hits
                and len(state.history) < provisional_hits
            ):
                unmatched.append((o, False, cid))
                continue

            continuity_score, severe_break = self._continuity_score(
                state, frame, o.bbox, descriptors[o.tracker_id]
            )
            if severe_break:
                self._raw_to_canonical.pop(o.tracker_id, None)
                self.discontinuity_events.append({
                    "frame": int(frame),
                    "tracker_id": int(o.tracker_id),
                    "canonical_id": int(cid),
                    "continuity_score": float(continuity_score),
                })
                unmatched.append((o, True, None))
                continue

            gap = frame - state.last_frame
            lifecycle = "REACQUIRED" if gap > 2 else "CONTINUING"
            assignment = self._update_state(
                state,
                frame=frame,
                observation=o,
                descriptor=descriptors[o.tracker_id],
                identity_confidence=max(0.55, continuity_score),
                lifecycle=lifecycle,
                stitched=False,
                discontinuity=False,
            )
            assignments[o.tracker_id] = assignment
            used_canonical.add(cid)

        # Build conservative candidate matches for raw IDs that are new to the
        # tracker (or were detached after a continuity break).
        candidates: dict[int, list[tuple[float, int]]] = {}
        max_gap_frames = max(1, int(round(self.config.identity_stitch_seconds * self.fps)))
        for o, _was_break, provisional_cid in unmatched:
            rows: list[tuple[float, int]] = []
            provisional_state = self._states.get(provisional_cid) if provisional_cid else None
            for cid, state in self._states.items():
                if cid in used_canonical or cid == provisional_cid:
                    continue
                if provisional_state is not None and state.last_frame >= provisional_state.first_frame:
                    # Only merge a young provisional identity backward into a
                    # vehicle that disappeared before it first appeared.
                    continue
                gap = frame - state.last_frame
                if gap <= 0 or gap > max_gap_frames:
                    continue
                score = self._stitch_score(
                    state,
                    frame=frame,
                    bbox=o.bbox,
                    class_id=o.class_id,
                    descriptor=descriptors[o.tracker_id],
                )
                if score is not None:
                    rows.append((score, cid))
            rows.sort(reverse=True)
            candidates[o.tracker_id] = rows

        # Global greedy assignment by best score.  Each raw/canonical identity
        # can participate in at most one stitch in the frame.
        proposals: list[tuple[float, int, int]] = []
        for raw_id, rows in candidates.items():
            if not rows:
                continue
            best_score, best_cid = rows[0]
            second = rows[1][0] if len(rows) > 1 else 0.0
            if best_score < self.config.identity_min_stitch_score:
                continue
            if best_score - second < self.config.identity_ambiguity_margin:
                continue
            proposals.append((best_score, raw_id, best_cid))
        proposals.sort(reverse=True)

        stitched_raw: set[int] = set()
        for score, raw_id, cid in proposals:
            if raw_id in stitched_raw or cid in used_canonical:
                continue
            o, _was_break, provisional_cid = next(
                row for row in unmatched if row[0].tracker_id == raw_id
            )
            state = self._states[cid]
            old_raw = state.current_tracker_id
            if provisional_cid is not None and provisional_cid in self._states:
                provisional = self._states.pop(provisional_cid)
                if self._raw_to_canonical.get(provisional.current_tracker_id) == provisional_cid:
                    self._raw_to_canonical.pop(provisional.current_tracker_id, None)
            state.generation += 1
            self._raw_to_canonical[raw_id] = cid
            # Remove any stale reverse mapping to make the active association
            # unambiguous.  ByteTrack IDs are not expected to be reused within
            # one tracker instance, but this protects custom tracker adapters.
            if self._raw_to_canonical.get(old_raw) == cid and old_raw != raw_id:
                self._raw_to_canonical.pop(old_raw, None)
            assignment = self._update_state(
                state,
                frame=frame,
                observation=o,
                descriptor=descriptors[raw_id],
                identity_confidence=score,
                lifecycle="STITCHED",
                stitched=True,
                discontinuity=False,
            )
            assignments[raw_id] = assignment
            used_canonical.add(cid)
            stitched_raw.add(raw_id)
            self.stitch_events.append({
                "frame": int(frame),
                "canonical_id": int(cid),
                "old_tracker_id": int(old_raw),
                "new_tracker_id": int(raw_id),
                "score": float(score),
            })

        # Anything still unmatched becomes a new physical identity.  This is
        # intentionally safer than forcing a weak/ambiguous re-identification.
        for o, was_break, provisional_cid in unmatched:
            if o.tracker_id in assignments:
                continue
            if provisional_cid is not None and provisional_cid in self._states:
                state = self._states[provisional_cid]
                continuity_score, severe_break = self._continuity_score(
                    state, frame, o.bbox, descriptors[o.tracker_id]
                )
                if severe_break:
                    self._states.pop(provisional_cid, None)
                    self._raw_to_canonical.pop(o.tracker_id, None)
                    state = self._new_state(frame, o, descriptors[o.tracker_id])
                    lifecycle = "NEW_AFTER_BREAK"
                    confidence = 0.72
                    discontinuity = True
                else:
                    lifecycle = "CONTINUING"
                    confidence = max(0.60, continuity_score)
                    discontinuity = False
            else:
                state = self._new_state(frame, o, descriptors[o.tracker_id])
                lifecycle = "NEW_AFTER_BREAK" if was_break else "NEW"
                confidence = 0.72 if was_break else 0.80
                discontinuity = was_break

            assignment = self._update_state(
                state,
                frame=frame,
                observation=o,
                descriptor=descriptors[o.tracker_id],
                identity_confidence=confidence,
                lifecycle=lifecycle,
                stitched=False,
                discontinuity=discontinuity,
            )
            assignments[o.tracker_id] = assignment
            used_canonical.add(state.canonical_id)

        self.forget_stale(frame)
        return [assignments[o.tracker_id] for o in obs]

    def forget_stale(self, current_frame: int) -> None:
        stale_frames = max(1, int(round(self.config.identity_state_seconds * self.fps)))
        stale = [
            cid
            for cid, state in self._states.items()
            if current_frame - state.last_frame > stale_frames
        ]
        for cid in stale:
            state = self._states.pop(cid)
            if self._raw_to_canonical.get(state.current_tracker_id) == cid:
                self._raw_to_canonical.pop(state.current_tracker_id, None)

    def _new_state(
        self,
        frame: int,
        observation: RawTrackObservation,
        descriptor: np.ndarray | None,
    ) -> _IdentityState:
        cid = self._next_canonical_id
        self._next_canonical_id += 1
        box = np.asarray(observation.bbox, dtype=np.float64)
        center, size = box_center_size(box)
        state = _IdentityState(
            canonical_id=cid,
            first_frame=int(frame),
            last_frame=int(frame),
            current_tracker_id=int(observation.tracker_id),
            appearance=descriptor.copy() if descriptor is not None else None,
            appearance_samples=1 if descriptor is not None else 0,
            filtered_center=center.copy(),
            filtered_size=size.copy(),
        )
        self._states[cid] = state
        self._raw_to_canonical[observation.tracker_id] = cid
        return state

    def _update_state(
        self,
        state: _IdentityState,
        *,
        frame: int,
        observation: RawTrackObservation,
        descriptor: np.ndarray | None,
        identity_confidence: float,
        lifecycle: str,
        stitched: bool,
        discontinuity: bool,
    ) -> IdentityAssignment:
        box = np.asarray(observation.bbox, dtype=np.float64)
        center, size = box_center_size(box)
        gap = max(1, int(frame - state.last_frame))

        if state.filtered_center is None or state.filtered_size is None:
            state.filtered_center = center.copy()
            state.filtered_size = size.copy()
            state.velocity = np.zeros(2, dtype=np.float64)
        else:
            # Predict with the prior velocity, then correct toward the new
            # measurement.  Reacquisitions use a stronger correction so the
            # display does not visibly lag after an occlusion.
            predicted = state.filtered_center + state.velocity * gap
            pos_alpha = 0.82 if stitched or gap > 2 else self.config.identity_position_alpha
            size_alpha = 0.72 if stitched or gap > 2 else self.config.identity_size_alpha
            corrected = predicted + pos_alpha * (center - predicted)
            measured_velocity = (corrected - state.filtered_center) / gap
            vel_alpha = self.config.identity_velocity_alpha
            state.velocity = (1.0 - vel_alpha) * state.velocity + vel_alpha * measured_velocity
            state.filtered_center = corrected
            state.filtered_size = (
                (1.0 - size_alpha) * state.filtered_size + size_alpha * size
            )

        state.last_frame = int(frame)
        state.current_tracker_id = int(observation.tracker_id)
        state.last_identity_confidence = float(np.clip(identity_confidence, 0.0, 1.0))
        state.history.append((
            int(frame),
            float(center[0]),
            float(center[1]),
            float(size[0]),
            float(size[1]),
        ))
        for key in list(state.class_scores):
            state.class_scores[key] *= 0.96
            if state.class_scores[key] < 1e-5:
                del state.class_scores[key]
        state.class_scores[observation.class_id] = (
            state.class_scores.get(observation.class_id, 0.0) + float(observation.confidence)
        )

        if descriptor is not None:
            if state.appearance is None:
                state.appearance = descriptor.copy()
            else:
                # Appearance changes slowly; using a conservative EMA prevents
                # a brief occlusion or glare from rewriting the vehicle model.
                beta = 0.10 if state.appearance_samples >= 4 else 0.22
                state.appearance = normalise_hist(
                    (1.0 - beta) * state.appearance + beta * descriptor
                )
            state.appearance_samples += 1

        self._raw_to_canonical[observation.tracker_id] = state.canonical_id
        provisional = bool(
            not stitched
            and state.generation == 1
            and len(state.history) < max(2, int(self.config.identity_provisional_hits))
        )
        return IdentityAssignment(
            canonical_id=state.canonical_id,
            tracker_id=int(observation.tracker_id),
            generation=state.generation,
            identity_confidence=state.last_identity_confidence,
            lifecycle=lifecycle,
            raw_bbox=box.astype(np.float32),
            filtered_bbox=center_size_box(
                state.filtered_center, state.filtered_size
            ).astype(np.float32),
            provisional=provisional,
            stitched=stitched,
            discontinuity=discontinuity,
        )

    def _continuity_score(
        self,
        state: _IdentityState,
        frame: int,
        bbox: np.ndarray,
        descriptor: np.ndarray | None,
    ) -> tuple[float, bool]:
        gap = max(1, frame - state.last_frame)
        pred_center, pred_size = self._predict(state, gap)
        center, size = box_center_size(bbox)
        scale = max(20.0, float(np.hypot(*pred_size)))
        error = float(np.linalg.norm(center - pred_center)) / scale
        area_ratio = size_ratio(pred_size, size)
        motion_score = exp(-0.5 * (error / 0.95) ** 2)
        size_score = exp(-abs(log(max(area_ratio, 1e-6))) / 0.80)
        appearance = appearance_similarity(state.appearance, descriptor)
        score = 0.60 * motion_score + 0.25 * size_score + 0.15 * appearance

        severe_break = bool(
            error > self.config.identity_hijack_max_jump
            or area_ratio > self.config.identity_hijack_max_size_ratio
            or (
                error > self.config.identity_hijack_appearance_jump
                and appearance < self.config.identity_hijack_min_appearance
                and state.appearance_samples >= 3
                and descriptor is not None
            )
        )
        return float(np.clip(score, 0.0, 1.0)), severe_break

    def _stitch_score(
        self,
        state: _IdentityState,
        *,
        frame: int,
        bbox: np.ndarray,
        class_id: int,
        descriptor: np.ndarray | None,
    ) -> float | None:
        gap = frame - state.last_frame
        min_gap = max(1, int(round(self.config.identity_min_stitch_gap_seconds * self.fps)))
        if gap < min_gap:
            # A raw ID that appears immediately after another disappears is
            # often a second vehicle at contact/occlusion, not a clean re-ID.
            # Waiting a few frames also gives the old ByteTrack ID a chance to
            # return before canonical identity is reassigned.
            return None
        pred_center, pred_size = self._predict(state, gap)
        center, size = box_center_size(bbox)
        scale = max(18.0, float(np.hypot(*pred_size)))
        error = float(np.linalg.norm(center - pred_center)) / scale
        # Prediction uncertainty grows during a gap.  The hard gate therefore
        # relaxes gradually but remains bounded to avoid cross-lane guessing.
        max_error = self.config.identity_max_prediction_error + min(
            0.75, 0.035 * gap
        )
        if error > max_error:
            return None

        area_ratio = size_ratio(pred_size, size)
        if area_ratio > self.config.identity_max_size_ratio:
            return None

        appearance = appearance_similarity(state.appearance, descriptor)
        appearance_available = (
            state.appearance is not None
            and descriptor is not None
            and state.appearance_samples >= 2
        )
        if appearance_available and appearance < self.config.identity_min_appearance:
            return None
        if (
            appearance_available
            and area_ratio > self.config.identity_large_scale_ratio
            and appearance < self.config.identity_large_scale_min_appearance
        ):
            # Large scale changes are common after occlusion/contact, but they
            # are also a common way a background false-positive gets merged
            # into a newly arriving vehicle.  Demand stronger appearance
            # agreement before preserving identity across that scale jump.
            return None

        source_velocity = robust_velocity(state.history)
        source_speed_norm = float(np.linalg.norm(source_velocity)) / max(
            18.0, float(np.hypot(*pred_size))
        )
        if (
            source_speed_norm < self.config.identity_static_speed_norm
            and error > self.config.identity_static_relocation_error
            and (
                not appearance_available
                or appearance < self.config.identity_static_min_appearance
            )
        ):
            # A long-lived nearly static source is often a fixed-scene false
            # positive.  Do not let it inherit a different vehicle merely
            # because that vehicle later crosses the same image region.
            return None

        motion_score = exp(-0.5 * (error / max(0.72, max_error * 0.52)) ** 2)
        size_score = exp(-abs(log(max(area_ratio, 1e-6))) / 0.65)
        gap_seconds = gap / self.fps
        gap_score = exp(-gap_seconds / max(self.config.identity_stitch_seconds, 1e-6))
        class_score = self._class_compatibility(state, class_id)

        if appearance_available:
            score = (
                0.38 * motion_score
                + 0.18 * size_score
                + 0.28 * appearance
                + 0.09 * gap_score
                + 0.07 * class_score
            )
        else:
            score = (
                0.54 * motion_score
                + 0.24 * size_score
                + 0.13 * gap_score
                + 0.09 * class_score
            )
        return float(np.clip(score, 0.0, 1.0))

    def _predict(self, state: _IdentityState, gap: int) -> tuple[np.ndarray, np.ndarray]:
        if state.filtered_center is None or state.filtered_size is None:
            last = state.history[-1]
            return np.array(last[1:3], dtype=np.float64), np.array(last[3:5], dtype=np.float64)

        # Blend the state filter velocity with a robust trajectory slope.  This
        # makes short gaps predictable without allowing one noisy last frame to
        # fling the expected position across the image.
        robust = robust_velocity(state.history)
        velocity = 0.55 * state.velocity + 0.45 * robust
        return state.filtered_center + velocity * gap, state.filtered_size.copy()

    def _class_compatibility(self, state: _IdentityState, class_id: int) -> float:
        if not state.class_scores:
            return 0.8
        stable = max(state.class_scores, key=state.class_scores.get)
        if stable == class_id:
            return 1.0
        # COCO car/truck/bus labels can flicker for the same road vehicle.  A
        # motorcycle mismatch is much less plausible and receives a penalty.
        road_heavy = {2, 5, 7}
        if stable in road_heavy and class_id in road_heavy:
            return 0.78
        return 0.25

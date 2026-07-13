"""Public crash detector — pair-attributed impact detection.

The detector is intentionally conservative about *who* crashed.  A vehicle can
brake, turn sharply, or suffer a noisy track without being labelled as a crash.
A crash event is emitted only for an interacting vehicle pair with recent
contact/near-contact evidence plus an impact-time motion discontinuity.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Any

import cv2
import numpy as np

from traffic_intel.incident.crash_fsm import (
    PairEvent,
    _Observation,
    _PairMetric,
    _PairState,
    PairCrashFSM,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HISTORY_SECONDS = 4.0
MIN_BBOX_H = 24
MIN_TRACK_QUALITY = 0.30

TTC_WINDOW_SECONDS = 1.6
PAIR_HISTORY_SECONDS = 2.0
PAIR_STALE_SECONDS = 2.0
SYNC_SECONDS = 0.22
PAIR_DEBOUNCE_SECONDS = 3.0
SHARED_TRACK_LOCK_SECONDS = 6.0

CONTACT_GAP_NORM = 0.48
NEAR_GAP_NORM = 1.25
IMPACT_THRESHOLD = 0.64


@dataclass(frozen=True, slots=True)
class CrashDetectorConfig:
    """Time-based crash detector configuration.

    Durations are expressed in seconds so replaying the same physical event at
    15, 30, or 60 FPS does not silently change the detector semantics.
    """

    history_seconds: float = HISTORY_SECONDS
    min_bbox_height: int = MIN_BBOX_H
    min_track_quality: float = MIN_TRACK_QUALITY
    ttc_window_seconds: float = TTC_WINDOW_SECONDS
    pair_history_seconds: float = PAIR_HISTORY_SECONDS
    pair_stale_seconds: float = PAIR_STALE_SECONDS
    sync_seconds: float = SYNC_SECONDS
    pair_debounce_seconds: float = PAIR_DEBOUNCE_SECONDS
    shared_track_lock_seconds: float = SHARED_TRACK_LOCK_SECONDS
    contact_gap_norm: float = CONTACT_GAP_NORM
    near_gap_norm: float = NEAR_GAP_NORM
    impact_threshold: float = IMPACT_THRESHOLD
    discontinuity_pre_start_seconds: float = 0.50
    discontinuity_pre_end_seconds: float = 0.20
    discontinuity_post_seconds: float = 0.17
    discontinuity_fresh_seconds: float = 0.45
    discontinuity_replace_seconds: float = 0.35
    maturity_seconds: float = 0.50
    pair_recent_seconds: float = 0.75
    pair_context_seconds: float = 1.25
    contact_dropout_seconds: float = 0.30
    impact_alignment_seconds: float = 0.20
    recent_contact_onset_seconds: float = 0.25
    flow_fresh_seconds: float = 0.12
    converging_timeout_seconds: float = 1.20
    candidate_timeout_seconds: float = 1.00
    confirmed_to_aftermath_seconds: float = 2.00
    near_miss_timeout_seconds: float = 1.00
    aftermath_pre_start_seconds: float = 0.47
    aftermath_pre_end_seconds: float = 0.13
    aftermath_post_seconds: float = 0.27

    # Runtime/input hardening.
    min_supported_fps: float = 5.0
    max_pair_gap_norm: float = 3.0

    # Pair-level impact validation.  A real impact should change the vehicles'
    # *relative* motion, not merely make two nearby vehicles brake together.
    # This gate is intentionally independent of detector-box contact so common
    # road motion and synchronized braking cannot become a crash on 1 px jitter.
    min_pair_relative_dv_norm: float = 0.035
    speed_only_relative_dv_norm: float = 0.050
    common_braking_cosine: float = 0.94
    common_heading_cosine: float = 0.92
    common_braking_min_drop: float = 0.35
    common_braking_max_relative_dv_norm: float = 0.12
    stale_contact_relative_dv_norm: float = 0.090

    # Occlusion/merge impact pathway.  These settings preserve detector-level
    # evidence that never became a trusted tracker identity.
    merge_enabled: bool = True
    merge_raw_min_confidence: float = 0.08
    merge_raw_max_confidence: float = 0.48
    merge_match_seconds: float = 0.30
    merge_memory_seconds: float = 0.90
    merge_min_missing_seconds: float = 0.067
    merge_survivor_maturity_seconds: float = 0.45
    merge_min_area_expansion: float = 1.55
    merge_min_width_expansion: float = 1.18
    merge_max_pre_overlap: float = 0.42
    merge_absorb_margin_norm: float = 0.45
    merge_event_cluster_seconds: float = 1.25
    merge_threshold: float = 0.72


@dataclass
class _WeakRawTrack:
    weak_id: int
    observations: deque = field(default_factory=lambda: deque(maxlen=24))
    last_frame: int = 0
    matched_this_frame: bool = False
    emitted: bool = False


@dataclass
class _MergeEventMemory:
    frame: int
    cx: float
    cy: float
    scale: float


@dataclass
class CrashCandidate:
    score: float
    reason: str
    description: str
    trigger_frame: int
    involved_tracks: list[int] = field(default_factory=list)
    detected_frame: int = 0
    evidence: dict[str, float] = field(default_factory=dict)
    involved_bboxes: dict[int, tuple[float, float, float, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public detector
# ---------------------------------------------------------------------------

class CrashDetector:
    """Crash detector that emits only pair-attributed events."""

    def __init__(
        self,
        fps: float = 30.0,
        impact_threshold: float | None = None,
        config: CrashDetectorConfig | None = None,
    ):
        self.config = config or CrashDetectorConfig()
        self.fps = self._validated_fps(fps)
        self.impact_threshold = float(
            self.config.impact_threshold if impact_threshold is None else impact_threshold
        )
        self._frame_count = 0
        max_history = max(20, self._frames(self.config.history_seconds))
        pair_history = max(15, self._frames(self.config.pair_history_seconds))
        self.fsm = PairCrashFSM(self.fps, self.config, max_history, pair_history)
        self._prev_gray: Optional[np.ndarray] = None
        self._track_box_history: dict[int, deque] = {}
        self._weak_raw_tracks: dict[int, _WeakRawTrack] = {}
        self._next_weak_id = 1
        self._merge_events: deque[_MergeEventMemory] = deque(maxlen=32)

    def reset(self) -> None:
        self.fsm.reset()
        self._prev_gray = None
        self._frame_count = 0
        self._track_box_history.clear()
        self._weak_raw_tracks.clear()
        self._next_weak_id = 1
        self._merge_events.clear()

    def set_fps(self, fps: float) -> None:
        new_fps = self._validated_fps(fps)
        if math.isclose(new_fps, self.fps, rel_tol=0.0, abs_tol=1e-9):
            return
        # Changing the time base while retaining old frame-indexed histories
        # corrupts every temporal threshold.  Reset rather than mixing units.
        self.reset()
        self.fps = new_fps
        self.fsm.set_fps(self.fps)

    def _validated_fps(self, fps: float) -> float:
        value = float(fps)
        if not math.isfinite(value) or value < self.config.min_supported_fps:
            raise ValueError(
                f"fps must be finite and >= {self.config.min_supported_fps:g}; got {fps!r}"
            )
        return value

    def _frames(self, seconds: float, minimum: int = 1) -> int:
        return max(minimum, int(round(float(seconds) * self.fps)))

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        frame_count: int,
        all_detections: list,
        frame: Optional[np.ndarray] = None,
        raw_detections: Optional[list] = None,
    ) -> list[CrashCandidate]:
        frame_count = int(frame_count)
        if frame_count < 0:
            raise ValueError(f"frame_count must be non-negative; got {frame_count}")
        if self._frame_count and frame_count <= self._frame_count:
            raise ValueError(
                f"frame_count must increase monotonically; previous={self._frame_count}, "
                f"current={frame_count}"
            )
        self._frame_count = frame_count
        self.fsm.frame_count = self._frame_count

        # Deduplicate tracker IDs deterministically and drop malformed upstream
        # observations.  A bad detector row must not poison temporal state.
        frame_dets_by_id: dict[int, tuple[float, float, object]] = {}
        for d in all_detections or []:
            if d.frame != frame_count:
                continue
            if not getattr(d, "track_confirmed", True):
                continue
            quality = float(getattr(d, "track_quality", 1.0))
            confidence = float(getattr(d, "confidence", 1.0))
            if not math.isfinite(quality) or quality < self.config.min_track_quality:
                continue
            if not math.isfinite(confidence):
                continue
            try:
                x1, y1, x2, y2 = map(float, d.bbox)
                tid = int(d.track_id)
            except (TypeError, ValueError, OverflowError):
                continue
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            if y2 - y1 < self.config.min_bbox_height:
                continue
            rank = (quality, confidence)
            previous = frame_dets_by_id.get(tid)
            if previous is None or rank > previous[:2]:
                frame_dets_by_id[tid] = (quality, confidence, d)
        frame_dets = [entry[2] for entry in frame_dets_by_id.values()]

        tracks_now: dict[int, _Observation] = {}
        for d in frame_dets:
            x1, y1, x2, y2 = map(float, d.bbox)
            w, h = x2 - x1, y2 - y1
            obs = _Observation(
                frame=frame_count,
                cx=(x1 + x2) * 0.5,
                cy=(y1 + y2) * 0.5,
                anchor_x=(x1 + x2) * 0.5,
                anchor_y=y2,
                w=w,
                h=h,
                bbox=(x1, y1, x2, y2),
                confidence=float(getattr(d, "confidence", 1.0)),
                quality=float(getattr(d, "track_quality", 1.0)),
            )
            tid = int(d.track_id)
            tracks_now[tid] = obs
            self.fsm.append_track(tid, obs)
            hist = self._track_box_history.setdefault(
                tid, deque(maxlen=max(20, self._frames(1.5)))
            )
            hist.append(obs)

        for tid in tracks_now:
            self.fsm.update_discontinuity(tid)

        gray = self._prepare_gray(frame)
        if gray is not None and self._prev_gray is not None and gray.shape != self._prev_gray.shape:
            # RTSP reconnects and source renegotiation can change resolution.
            # Optical flow requires equal image sizes, so safely restart only
            # the flow reference while preserving kinematic histories.
            self._prev_gray = None
            for ps in self.fsm.pairs.values():
                ps.flow_score = 0.0
                ps.flow_frame = -9999

        # Update pair geometry for currently visible tracks.
        current_pairs: list[tuple[tuple[int, int], float]] = []
        tids = sorted(tracks_now)
        velocity_cache = {tid: self.fsm.recent_velocity(tid) for tid in tids}
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = tids[i], tids[j]
                if not self._pair_may_interact(
                    tracks_now[a], tracks_now[b], velocity_cache[a], velocity_cache[b]
                ):
                    continue
                metric = self._pair_metric(
                    a, b, tracks_now[a], tracks_now[b],
                    va=velocity_cache[a], vb=velocity_cache[b],
                )
                if (
                    metric.gap_norm > self.config.near_gap_norm
                    and metric.risk < 0.15
                    and metric.contact < 0.10
                ):
                    continue
                key = (a, b)
                ps = self.fsm.pairs.get(key)
                if ps is None:
                    ps = _PairState(metrics=deque(maxlen=(
                        max(15, self._frames(self.config.pair_history_seconds))
                    )))
                    self.fsm.pairs[key] = ps
                ps.metrics.append(metric)
                ps.last_seen = frame_count
                if metric.gap_norm < ps.closest_gap:
                    ps.closest_gap = metric.gap_norm
                    ps.closest_frame = frame_count
                current_pairs.append((key, max(metric.risk, metric.contact)))

        # Candidate-only optical flow: run only for the strongest few interactions.
        if gray is not None and self._prev_gray is not None:
            for key, _ in sorted(current_pairs, key=lambda kv: kv[1], reverse=True)[:3]:
                ps = self.fsm.pairs[key]
                if not ps.metrics:
                    continue
                m = ps.metrics[-1]
                if max(m.risk, m.contact) < 0.42:
                    continue
                a, b = key
                if a in tracks_now and b in tracks_now:
                    ps.flow_score = PairCrashFSM.local_flow_score(
                        self._prev_gray, gray,
                        tracks_now[a].bbox, tracks_now[b].bbox,
                    )
                    ps.flow_frame = frame_count

        candidates: list[CrashCandidate] = []

        if self.config.merge_enabled and raw_detections is not None:
            candidates.extend(
                self._update_merge_occlusion_path(
                    frame_count=frame_count,
                    tracks_now=tracks_now,
                    raw_detections=raw_detections,
                )
            )

        for key, ps in list(self.fsm.pairs.items()):
            result = self.fsm.advance_pair_state(key, ps)
            if result is not None:
                candidates.append(CrashCandidate(
                    score=result["score"],
                    reason=result["reason"],
                    description=result["description"],
                    trigger_frame=result["trigger_frame"],
                    detected_frame=result["detected_frame"],
                    involved_tracks=result["involved_tracks"],
                    evidence=result["evidence"],
                    involved_bboxes=result["involved_bboxes"],
                ))

        if gray is not None:
            self._prev_gray = gray

        self.fsm.forget_stale()
        return candidates

    @staticmethod
    def _prepare_gray(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame is None:
            return None
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("frame must be a non-empty numpy array")
        if frame.ndim == 2:
            return frame
        if frame.ndim == 3 and frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        raise ValueError(f"unsupported frame shape for optical flow: {frame.shape}")

    # ------------------------------------------------------------------
    # Occlusion / merge impact pathway
    # ------------------------------------------------------------------

    @staticmethod
    def _box_area(bbox: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        return max(1.0, x2 - x1) * max(1.0, y2 - y1)

    @staticmethod
    def _box_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @staticmethod
    def _contains_with_margin(
        outer: tuple[float, float, float, float],
        inner: tuple[float, float, float, float],
        margin_norm: float,
    ) -> float:
        """Return how well ``outer`` spatially explains ``inner`` after expansion."""
        ox1, oy1, ox2, oy2 = outer
        iw = max(1.0, inner[2] - inner[0])
        ih = max(1.0, inner[3] - inner[1])
        margin = margin_norm * max(iw, ih)
        ex = (ox1 - margin, oy1 - margin, ox2 + margin, oy2 + margin)
        cx, cy = CrashDetector._box_center(inner)
        center_inside = ex[0] <= cx <= ex[2] and ex[1] <= cy <= ex[3]
        iou = PairCrashFSM._iou(outer, inner)
        return max(float(center_inside), min(1.0, iou / 0.20))

    def _trusted_overlap(self, bbox, tracks_now: dict[int, _Observation]) -> float:
        return max(
            (PairCrashFSM._iou(tuple(map(float, bbox)), obs.bbox) for obs in tracks_now.values()),
            default=0.0,
        )

    def _update_merge_occlusion_path(
        self,
        *,
        frame_count: int,
        tracks_now: dict[int, _Observation],
        raw_detections: list,
    ) -> list[CrashCandidate]:
        cfg = self.config
        for weak in self._weak_raw_tracks.values():
            weak.matched_this_frame = False

        # Build ephemeral tracks only for detector observations that are not
        # already explained by a trusted vehicle track.  This keeps ordinary
        # tracker output on the strong pair-attributed path.
        candidates = []
        for raw in raw_detections:
            conf = float(getattr(raw, "confidence", 0.0))
            if conf < cfg.merge_raw_min_confidence or conf > cfg.merge_raw_max_confidence:
                continue
            bbox = tuple(map(float, getattr(raw, "bbox")))
            if self._trusted_overlap(bbox, tracks_now) > 0.30:
                continue
            candidates.append((raw, bbox, conf))

        match_gap = self._frames(cfg.merge_match_seconds)
        for raw, bbox, conf in candidates:
            best_id = None
            best_score = -1.0
            rcx, rcy = self._box_center(bbox)
            rdiag = max(20.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
            for wid, weak in self._weak_raw_tracks.items():
                if weak.emitted or not weak.observations:
                    continue
                if frame_count - weak.last_frame > match_gap:
                    continue
                prev = weak.observations[-1]
                pb = prev[2]
                pcx, pcy = self._box_center(pb)
                pdiag = max(20.0, math.hypot(pb[2] - pb[0], pb[3] - pb[1]))
                dist = math.hypot(rcx - pcx, rcy - pcy) / max(rdiag, pdiag)
                iou = PairCrashFSM._iou(bbox, pb)
                score = max(iou, 1.0 - dist)
                if score > 0.20 and score > best_score:
                    best_id, best_score = wid, score
            if best_id is None:
                best_id = self._next_weak_id
                self._next_weak_id += 1
                self._weak_raw_tracks[best_id] = _WeakRawTrack(weak_id=best_id)
            weak = self._weak_raw_tracks[best_id]
            weak.observations.append((frame_count, conf, bbox))
            weak.last_frame = frame_count
            weak.matched_this_frame = True

        emitted: list[CrashCandidate] = []
        min_missing = self._frames(cfg.merge_min_missing_seconds)
        max_missing = self._frames(cfg.merge_memory_seconds)
        mature_frames = self._frames(cfg.merge_survivor_maturity_seconds)

        for weak in list(self._weak_raw_tracks.values()):
            if weak.emitted or weak.matched_this_frame or len(weak.observations) < 2:
                continue
            missing = frame_count - weak.last_frame
            if missing < min_missing or missing > max_missing:
                continue

            last_frame, last_conf, lost_box = weak.observations[-1]
            first_frame = weak.observations[0][0]
            weak_persistence = min(1.0, len(weak.observations) / 4.0)

            best = None
            for tid, current in tracks_now.items():
                hist = self._track_box_history.get(tid)
                if not hist or len(hist) < mature_frames:
                    continue

                # Compare the survivor to its geometry immediately before the
                # weak vehicle disappeared.  A collision merge should be a
                # sudden innovation, not ordinary perspective growth.
                pre = None
                for obs in reversed(hist):
                    if obs.frame <= last_frame - 1:
                        pre = obs
                        break
                if pre is None:
                    continue
                pre_area = self._box_area(pre.bbox)
                cur_area = self._box_area(current.bbox)
                area_ratio = cur_area / max(pre_area, 1.0)
                pre_w = max(1.0, pre.bbox[2] - pre.bbox[0])
                cur_w = max(1.0, current.bbox[2] - current.bbox[0])
                width_ratio = cur_w / pre_w
                if area_ratio < cfg.merge_min_area_expansion:
                    continue
                if width_ratio < cfg.merge_min_width_expansion:
                    continue

                # B must have been genuinely distinct from A before vanishing.
                pre_overlap = PairCrashFSM._iou(pre.bbox, lost_box)
                if pre_overlap > cfg.merge_max_pre_overlap:
                    continue

                pdiag = max(20.0, math.hypot(pre.w, pre.h))
                lcx, lcy = self._box_center(lost_box)
                near_pre = math.hypot(lcx - pre.cx, lcy - pre.cy) / pdiag
                proximity = float(np.clip(1.0 - near_pre / 1.60, 0.0, 1.0))
                if proximity <= 0.05:
                    continue

                absorption = self._contains_with_margin(
                    current.bbox, lost_box, cfg.merge_absorb_margin_norm
                )
                if absorption <= 0.0:
                    continue

                expansion = float(np.clip((area_ratio - 1.0) / 1.25, 0.0, 1.0))
                width_jump = float(np.clip((width_ratio - 1.0) / 0.60, 0.0, 1.0))
                weak_conf = float(np.clip(last_conf / cfg.merge_raw_max_confidence, 0.0, 1.0))
                disappearance = float(np.clip(missing / max(min_missing + 1, self._frames(0.20)), 0.0, 1.0))
                score = (
                    0.24 * expansion
                    + 0.13 * width_jump
                    + 0.20 * absorption
                    + 0.17 * proximity
                    + 0.11 * weak_conf
                    + 0.08 * weak_persistence
                    + 0.07 * disappearance
                )
                if score < cfg.merge_threshold:
                    continue

                if best is None or score > best[0]:
                    best = (score, tid, current, pre, area_ratio, width_ratio, proximity, absorption)

            if best is None:
                continue

            score, tid, current, pre, area_ratio, width_ratio, proximity, absorption = best
            cx, cy = self._box_center(current.bbox)
            scale = max(20.0, math.hypot(current.w, current.h))
            clustered = False
            cluster_frames = self._frames(cfg.merge_event_cluster_seconds)
            for evt in self._merge_events:
                if frame_count - evt.frame > cluster_frames:
                    continue
                if math.hypot(cx - evt.cx, cy - evt.cy) <= 1.25 * max(scale, evt.scale):
                    clustered = True
                    break
            weak.emitted = True
            if clustered:
                continue

            self._merge_events.append(_MergeEventMemory(frame_count, cx, cy, scale))
            emitted.append(CrashCandidate(
                score=float(np.clip(score, 0.0, 1.0)),
                reason="merge_occlusion_impact",
                description=(
                    "Probable collision: a recent vehicle observation disappeared "
                    "beside a mature track as the surviving vehicle box abruptly expanded."
                ),
                trigger_frame=int(last_frame),
                detected_frame=int(frame_count),
                involved_tracks=[int(tid)],
                evidence={
                    "merge_area_ratio": float(area_ratio),
                    "merge_width_ratio": float(width_ratio),
                    "lost_vehicle_confidence": float(last_conf),
                    "lost_vehicle_persistence": float(weak_persistence),
                    "lost_vehicle_proximity": float(proximity),
                    "merged_box_absorption": float(absorption),
                    "missing_frames": float(missing),
                },
                involved_bboxes={int(tid): tuple(map(float, current.bbox))},
            ))

        # Bound memory without deleting newly missing evidence too early.
        stale = self._frames(cfg.merge_memory_seconds * 1.5)
        for wid, weak in list(self._weak_raw_tracks.items()):
            if weak.emitted or frame_count - weak.last_frame > stale:
                self._weak_raw_tracks.pop(wid, None)
        while self._merge_events and frame_count - self._merge_events[0].frame > self._frames(4.0):
            self._merge_events.popleft()
        return emitted

    # ------------------------------------------------------------------
    # Pair interaction geometry
    # ------------------------------------------------------------------

    def _pair_may_interact(
        self,
        oa: _Observation,
        ob: _Observation,
        va: tuple[float, float],
        vb: tuple[float, float],
    ) -> bool:
        """Cheap broad phase before the full pair metric.

        Nearby boxes always pass.  Distant boxes pass only when their current
        relative velocity predicts a close approach inside the TTC horizon.
        This preserves fast collision approaches while avoiding expensive full
        scoring for obviously unrelated vehicles in crowded scenes.
        """
        diag_a = math.hypot(oa.w, oa.h)
        diag_b = math.hypot(ob.w, ob.h)
        scale = max(20.0, 0.5 * (diag_a + diag_b))
        if PairCrashFSM._bbox_gap(oa.bbox, ob.bbox) / scale <= self.config.max_pair_gap_norm:
            return True

        rx, ry = ob.cx - oa.cx, ob.cy - oa.cy
        rvx, rvy = vb[0] - va[0], vb[1] - va[1]
        rv2 = rvx * rvx + rvy * rvy
        dot = rx * rvx + ry * rvy
        if dot >= 0.0 or rv2 <= 0.02:
            return False
        ttc_frames = -dot / rv2
        if ttc_frames > self.config.ttc_window_seconds * 30.0:
            return False
        px, py = rx + rvx * ttc_frames, ry + rvy * ttc_frames
        return math.hypot(px, py) / scale <= 1.35

    def _pair_metric(
        self,
        a: int,
        b: int,
        oa: _Observation,
        ob: _Observation,
        *,
        va: tuple[float, float] | None = None,
        vb: tuple[float, float] | None = None,
    ):
        gap_px = PairCrashFSM._bbox_gap(oa.bbox, ob.bbox)
        diag_a = math.hypot(oa.w, oa.h)
        diag_b = math.hypot(ob.w, ob.h)
        scale = max(20.0, 0.5 * (diag_a + diag_b))
        gap_norm = gap_px / scale
        iou = PairCrashFSM._iou(oa.bbox, ob.bbox)

        va = self.fsm.recent_velocity(a) if va is None else va
        vb = self.fsm.recent_velocity(b) if vb is None else vb
        rx, ry = ob.cx - oa.cx, ob.cy - oa.cy
        rvx, rvy = vb[0] - va[0], vb[1] - va[1]
        rv2 = rvx * rvx + rvy * rvy
        dot = rx * rvx + ry * rvy

        ttc_frames = 999.0
        dmin_norm = math.hypot(rx, ry) / scale
        closing = dot < 0.0 and rv2 > 0.02
        if closing:
            ttc_frames = max(0.0, -dot / rv2)
            px, py = rx + rvx * ttc_frames, ry + rvy * ttc_frames
            dmin_norm = math.hypot(px, py) / scale

        ttc_limit = self.config.ttc_window_seconds * 30.0
        ttc_score = 0.0
        if closing and 0.0 <= ttc_frames <= ttc_limit:
            ttc_score = 1.0 - ttc_frames / max(ttc_limit, 1.0)
        dmin_score = float(np.clip(1.0 - dmin_norm / 1.0, 0.0, 1.0))
        gap_score = float(np.clip(1.0 - gap_norm / 1.8, 0.0, 1.0))
        risk = 0.46 * ttc_score + 0.34 * dmin_score + 0.20 * gap_score
        if not closing:
            risk *= 0.30

        contact = max(
            float(np.clip(iou / 0.18, 0.0, 1.0)),
            float(np.clip(1.0 - gap_norm / self.config.contact_gap_norm, 0.0, 1.0)),
        )

        mean_h = max(18.0, 0.5 * (oa.h + ob.h))
        anchor_dist_norm = math.hypot(
            ob.anchor_x - oa.anchor_x, ob.anchor_y - oa.anchor_y
        ) / scale
        bottom_diff_norm = abs(ob.anchor_y - oa.anchor_y) / mean_h
        scale_ratio = max(oa.h, ob.h) / max(1.0, min(oa.h, ob.h))

        return _PairMetric(
            frame=self._frame_count,
            gap_norm=float(gap_norm),
            iou=float(iou),
            ttc_frames=float(ttc_frames),
            dmin_norm=float(dmin_norm),
            risk=float(np.clip(risk, 0.0, 1.0)),
            contact=float(np.clip(contact, 0.0, 1.0)),
            anchor_dist_norm=float(anchor_dist_norm),
            bottom_diff_norm=float(bottom_diff_norm),
            scale_ratio=float(scale_ratio),
        )


# Backward-compatible re-exports
from traffic_intel.incident.crash_visuals import (
    draw_crash_boxes,
    mark_crash_vehicles,
    reset_crash_visuals,
    update_crash_visuals,
)

__all__ = [
    "CrashCandidate", "CrashDetector", "CrashDetectorConfig", "PairEvent",
    "draw_crash_boxes", "mark_crash_vehicles", "reset_crash_visuals",
    "update_crash_visuals",
]

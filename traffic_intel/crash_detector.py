"""Pair-attributed crash detection.

The detector is intentionally conservative about *who* crashed.  A vehicle can
brake, turn sharply, or suffer a noisy track without being labelled as a crash.
A crash event is emitted only for an interacting vehicle pair with recent
contact/near-contact evidence plus an impact-time motion discontinuity.

Pipeline:
    confirmed tracks
      -> robust per-track kinematics
      -> pair interaction / TTC / closest approach
      -> temporal state machine
      -> synchronized (or contact-coupled) motion discontinuity
      -> optional candidate-only optical-flow support
      -> pair-attributed crash event

Post-impact stops are supporting evidence only.  They never create a crash by
themselves, which prevents unrelated stopped or braking vehicles from being
marked as crash participants.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import cv2
import numpy as np

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

# Geometry is normalized by average vehicle box diagonal, making it less depth
# sensitive than raw pixels.  0 means box overlap; 1 is roughly one vehicle
# diagonal of edge-to-edge separation.
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



class PairEvent(Enum):
    NORMAL = "NORMAL"
    CONVERGING = "CONVERGING"
    IMPACT_CANDIDATE = "IMPACT_CANDIDATE"
    CRASH_CONFIRMED = "CRASH_CONFIRMED"
    AFTERMATH = "AFTERMATH"
    NEAR_MISS = "NEAR_MISS"


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


@dataclass
class _Observation:
    frame: int
    cx: float
    cy: float
    anchor_x: float
    anchor_y: float
    w: float
    h: float
    bbox: tuple[float, float, float, float]
    confidence: float
    quality: float


@dataclass
class _TrackState:
    history: deque
    last_seen: int = -1
    last_disc_frame: int = -9999
    last_disc_score: float = 0.0
    last_disc_dv: float = 0.0
    last_disc_heading: float = 0.0
    last_disc_pred_error: float = 0.0
    last_disc_history_span: int = 0


@dataclass
class _PairMetric:
    frame: int
    gap_norm: float
    iou: float
    ttc_frames: float
    dmin_norm: float
    risk: float
    contact: float
    # Road-plane plausibility proxies.  Bottom-centre depth consistency is
    # substantially safer than box IoU alone for perspective CCTV footage.
    anchor_dist_norm: float
    bottom_diff_norm: float
    scale_ratio: float


@dataclass
class _PairState:
    metrics: deque
    event: PairEvent = PairEvent.NORMAL
    event_frame: int = -1
    last_seen: int = -1
    last_trigger_frame: int = -9999
    closest_frame: int = -1
    closest_gap: float = 999.0
    flow_score: float = 0.0
    flow_frame: int = -9999


class CrashDetector:
    """Crash detector that emits only pair-attributed events by default."""

    def __init__(
        self,
        fps: float = 30.0,
        impact_threshold: float | None = None,
        config: CrashDetectorConfig | None = None,
    ):
        self.fps = float(fps or 30.0)
        self.config = config or CrashDetectorConfig()
        self.impact_threshold = float(
            self.config.impact_threshold if impact_threshold is None else impact_threshold
        )
        self._frame_count = 0
        self._max_history = max(20, self._frames(self.config.history_seconds))
        self._pair_history = max(15, self._frames(self.config.pair_history_seconds))
        self._tracks: dict[int, _TrackState] = {}
        self._pairs: dict[tuple[int, int], _PairState] = {}
        self._claimed_until: dict[int, int] = {}
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._tracks.clear()
        self._pairs.clear()
        self._claimed_until.clear()
        self._prev_gray = None
        self._frame_count = 0

    def set_fps(self, fps: float) -> None:
        self.fps = float(fps or 30.0)
        self._max_history = max(20, self._frames(self.config.history_seconds))
        self._pair_history = max(15, self._frames(self.config.pair_history_seconds))

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
    ) -> list[CrashCandidate]:
        self._frame_count = int(frame_count)

        # Keep only confirmed, sufficiently visible tracks from this frame.
        frame_dets = []
        for d in all_detections:
            if d.frame != frame_count:
                continue
            if not getattr(d, "track_confirmed", True):
                continue
            if float(getattr(d, "track_quality", 1.0)) < self.config.min_track_quality:
                continue
            x1, y1, x2, y2 = map(float, d.bbox)
            if y2 - y1 < self.config.min_bbox_height:
                continue
            frame_dets.append(d)

        tracks_now: dict[int, _Observation] = {}
        for d in frame_dets:
            x1, y1, x2, y2 = map(float, d.bbox)
            w, h = x2 - x1, y2 - y1
            obs = _Observation(
                frame=frame_count,
                cx=(x1 + x2) * 0.5,
                cy=(y1 + y2) * 0.5,
                # Bottom-centre is more stable for road-plane motion than box centre.
                anchor_x=(x1 + x2) * 0.5,
                anchor_y=y2,
                w=w,
                h=h,
                bbox=(x1, y1, x2, y2),
                confidence=float(getattr(d, "confidence", 1.0)),
                quality=float(getattr(d, "track_quality", 1.0)),
            )
            tracks_now[int(d.track_id)] = obs
            self._append_track(int(d.track_id), obs)

        # Compute current per-track discontinuity evidence before pair fusion.
        for tid in tracks_now:
            self._update_discontinuity(tid)

        gray = None
        if frame is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Update pair geometry for currently visible tracks.
        current_pairs: list[tuple[tuple[int, int], float]] = []
        tids = sorted(tracks_now)
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = tids[i], tids[j]
                metric = self._pair_metric(a, b, tracks_now[a], tracks_now[b])
                # Do not maintain every pair in a crowded scene.  Far-separated
                # vehicles cannot be an impact pair in the immediate future.
                if metric.gap_norm > 3.0 and metric.risk < 0.15:
                    continue
                key = (a, b)
                ps = self._pairs.get(key)
                if ps is None:
                    ps = _PairState(metrics=deque(maxlen=self._pair_history))
                    self._pairs[key] = ps
                ps.metrics.append(metric)
                ps.last_seen = frame_count
                if metric.gap_norm < ps.closest_gap:
                    ps.closest_gap = metric.gap_norm
                    ps.closest_frame = frame_count
                current_pairs.append((key, max(metric.risk, metric.contact)))

        # Candidate-only optical flow: run only for the strongest few interactions.
        if gray is not None and self._prev_gray is not None:
            for key, _ in sorted(current_pairs, key=lambda kv: kv[1], reverse=True)[:3]:
                ps = self._pairs[key]
                if not ps.metrics:
                    continue
                m = ps.metrics[-1]
                if max(m.risk, m.contact) < 0.42:
                    continue
                a, b = key
                if a in tracks_now and b in tracks_now:
                    ps.flow_score = self._local_flow_score(
                        self._prev_gray, gray,
                        tracks_now[a].bbox, tracks_now[b].bbox,
                    )
                    ps.flow_frame = frame_count

        candidates: list[CrashCandidate] = []
        for key, ps in list(self._pairs.items()):
            candidate = self._advance_pair_state(key, ps)
            if candidate is not None:
                candidates.append(candidate)

        if gray is not None:
            self._prev_gray = gray

        self._forget_stale()
        return candidates

    # ------------------------------------------------------------------
    # Track state / kinematics
    # ------------------------------------------------------------------

    def _append_track(self, tid: int, obs: _Observation) -> None:
        ts = self._tracks.get(tid)
        if ts is None:
            ts = _TrackState(history=deque(maxlen=self._max_history))
            self._tracks[tid] = ts
        # Never duplicate a frame for the same track.
        if ts.history and ts.history[-1].frame == obs.frame:
            ts.history[-1] = obs
        else:
            ts.history.append(obs)
        ts.last_seen = obs.frame

    def _velocity(self, entries: list[_Observation]) -> tuple[float, float]:
        """Median velocity in 30-FPS-equivalent pixels per frame.

        The original detector was tuned at 30 FPS.  Scaling raw per-frame
        displacement by ``fps / 30`` preserves those thresholds while making
        the same physical motion comparable at 15, 30, and 60 FPS.
        """
        if len(entries) < 2:
            return 0.0, 0.0
        rate_scale = self.fps / 30.0
        vxs, vys = [], []
        for p, q in zip(entries[:-1], entries[1:]):
            df = max(1, q.frame - p.frame)
            vxs.append(((q.anchor_x - p.anchor_x) / df) * rate_scale)
            vys.append(((q.anchor_y - p.anchor_y) / df) * rate_scale)
        return float(np.median(vxs)), float(np.median(vys))

    def _recent_velocity(self, tid: int, seconds: float = 0.20) -> tuple[float, float]:
        ts = self._tracks.get(tid)
        if ts is None:
            return 0.0, 0.0
        cutoff = self._frame_count - self._frames(seconds)
        entries = [o for o in ts.history if o.frame >= cutoff]
        return self._velocity(entries)

    def _update_discontinuity(self, tid: int) -> None:
        ts = self._tracks[tid]
        hist = list(ts.history)
        if len(hist) < 9:
            return

        fc = self._frame_count
        # Roughly 0.45-0.20 s before vs. most recent 0.17 s.  Using frame
        # windows rather than list slices tolerates short detector dropouts.
        pre_start = self._frames(self.config.discontinuity_pre_start_seconds)
        pre_end = self._frames(self.config.discontinuity_pre_end_seconds)
        post_window = self._frames(self.config.discontinuity_post_seconds)
        pre = [o for o in hist if fc - pre_start <= o.frame <= fc - pre_end]
        post = [o for o in hist if fc - post_window <= o.frame <= fc]
        if len(pre) < 3 or len(post) < 3:
            return

        vb = self._velocity(pre)
        va = self._velocity(post)
        speed_b = math.hypot(*vb)
        speed_a = math.hypot(*va)
        scale = max(18.0, float(np.median([o.h for o in hist[-12:]])))
        dv_norm = math.hypot(va[0] - vb[0], va[1] - vb[1]) / scale

        heading_change = 0.0
        if speed_b / scale > 0.012 and speed_a / scale > 0.012:
            hb = math.degrees(math.atan2(vb[1], vb[0]))
            ha = math.degrees(math.atan2(va[1], va[0]))
            heading_change = abs(self._angle_delta(ha, hb))

        # One-step prediction error from a pre-impact local motion model.
        prior = hist[-7:-1]
        pred_error = 0.0
        if len(prior) >= 3:
            pv = self._velocity(prior)
            last = prior[-1]
            cur = hist[-1]
            dt = max(1, cur.frame - last.frame)
            frame_scale = 30.0 / self.fps
            px = last.anchor_x + pv[0] * dt * frame_scale
            py = last.anchor_y + pv[1] * dt * frame_scale
            pred_error = math.hypot(cur.anchor_x - px, cur.anchor_y - py) / scale

        speed_change = abs(speed_a - speed_b) / max(speed_b, 0.05 * scale)

        dv_c = float(np.clip((dv_norm - 0.015) / 0.085, 0.0, 1.0))
        heading_c = float(np.clip((heading_change - 18.0) / 72.0, 0.0, 1.0))
        pred_c = float(np.clip((pred_error - 0.04) / 0.42, 0.0, 1.0))
        speed_c = float(np.clip((speed_change - 0.25) / 1.25, 0.0, 1.0))
        score = 0.46 * dv_c + 0.20 * heading_c + 0.24 * pred_c + 0.10 * speed_c

        # Ignore weak noise; remember the strongest recent impulse for pair sync.
        if score >= 0.34:
            if score >= ts.last_disc_score or fc - ts.last_disc_frame > self._frames(self.config.discontinuity_replace_seconds):
                ts.last_disc_frame = fc
                ts.last_disc_score = float(score)
                ts.last_disc_dv = float(dv_norm)
                ts.last_disc_heading = float(heading_change)
                ts.last_disc_pred_error = float(pred_error)
                ts.last_disc_history_span = int(hist[-1].frame - hist[0].frame)

    # ------------------------------------------------------------------
    # Pair interaction geometry
    # ------------------------------------------------------------------

    def _pair_metric(
        self, a: int, b: int, oa: _Observation, ob: _Observation
    ) -> _PairMetric:
        gap_px = self._bbox_gap(oa.bbox, ob.bbox)
        diag_a = math.hypot(oa.w, oa.h)
        diag_b = math.hypot(ob.w, ob.h)
        scale = max(20.0, 0.5 * (diag_a + diag_b))
        gap_norm = gap_px / scale
        iou = self._iou(oa.bbox, ob.bbox)

        va = self._recent_velocity(a)
        vb = self._recent_velocity(b)
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

        # ttc_frames is expressed in 30-FPS-equivalent frames because the
        # relative velocity is rate-normalized above.
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

    # ------------------------------------------------------------------
    # State machine / fusion
    # ------------------------------------------------------------------

    def _advance_pair_state(
        self, key: tuple[int, int], ps: _PairState
    ) -> Optional[CrashCandidate]:
        if not ps.metrics:
            return None
        fc = self._frame_count
        a, b = key
        recent = [m for m in ps.metrics if fc - m.frame <= self._frames(self.config.pair_recent_seconds)]
        pre_window = [m for m in ps.metrics if fc - m.frame <= self._frames(self.config.pair_context_seconds)]
        if not recent:
            return None

        risk_peak = max(m.risk for m in pre_window)
        risk_persist = sum(m.risk >= 0.40 for m in pre_window)
        contact_peak = max(m.contact for m in recent)
        min_gap_metric = min(recent, key=lambda m: m.gap_norm)
        near_contact = min_gap_metric.gap_norm <= self.config.near_gap_norm

        ta = self._tracks.get(a)
        tb = self._tracks.get(b)
        if ta is None or tb is None:
            return None

        sync_frames = max(2, self._frames(self.config.sync_seconds))
        age_a = fc - ta.last_disc_frame
        age_b = fc - tb.last_disc_frame
        disc_age_limit = self._frames(self.config.discontinuity_fresh_seconds)
        disc_a = ta.last_disc_score if age_a <= disc_age_limit else 0.0
        disc_b = tb.last_disc_score if age_b <= disc_age_limit else 0.0
        # A newly-created track has not accumulated enough trajectory context
        # to distinguish a real impulse from detector settling.  This was a
        # major source of phantom crashes in the supplied clips.
        maturity_frames = max(8, self._frames(self.config.maturity_seconds))
        # Use the amount of history that existed *when the impulse was
        # measured*.  An immature detector-settling jump must not become valid
        # retroactively just because the track survives a few more frames.
        mature_a = ta.last_disc_history_span >= maturity_frames
        mature_b = tb.last_disc_history_span >= maturity_frames

        # A single vehicle may support a crash only with a directional
        # discontinuity.  Pure speed loss is deliberately *not* sufficient:
        # hard braking is common.  Two vehicles can still confirm a rear-end
        # style impact through synchronized strong delta-v.
        physical_a = mature_a and self._physical_impulse(ta) if disc_a > 0 else False
        physical_b = mature_b and self._physical_impulse(tb) if disc_b > 0 else False
        speed_impulse_a = mature_a and disc_a >= 0.34 and ta.last_disc_dv >= 0.090
        speed_impulse_b = mature_b and disc_b >= 0.34 and tb.last_disc_dv >= 0.090
        directional_sync = (
            disc_a >= 0.34 and disc_b >= 0.34
            and abs(ta.last_disc_frame - tb.last_disc_frame) <= sync_frames
            and physical_a and physical_b
        )
        speed_only_sync = (
            disc_a >= 0.34 and disc_b >= 0.34
            and abs(ta.last_disc_frame - tb.last_disc_frame) <= sync_frames
            and speed_impulse_a and speed_impulse_b
            and not (physical_a or physical_b)
        )
        synchronized = directional_sync or speed_only_sync

        if synchronized:
            kinematic = 0.62 * max(disc_a, disc_b) + 0.38 * min(disc_a, disc_b)
        else:
            # One vehicle can momentarily vanish/merge at the collision point.
            # A single strong impulse is accepted only when pair contact geometry
            # is already strong; it can never fire in isolation.
            physical_disc = max(
                disc_a if physical_a else 0.0,
                disc_b if physical_b else 0.0,
            )
            kinematic = 0.72 * physical_disc

        a_missing = fc - ta.last_seen
        b_missing = fc - tb.last_seen
        contact_dropout = (
            contact_peak >= 0.45
            and max(a_missing, b_missing) <= self._frames(self.config.contact_dropout_seconds)
            and max(a_missing, b_missing) >= 2
        )
        if contact_dropout and (physical_a or physical_b):
            kinematic = min(1.0, kinematic + 0.10)

        flow = ps.flow_score if fc - ps.flow_frame <= max(3, self._frames(self.config.flow_fresh_seconds)) else 0.0
        aftermath = self._aftermath_score(key, min_gap_metric.frame)

        if ps.event == PairEvent.NORMAL:
            if risk_persist >= 3 and risk_peak >= 0.40:
                ps.event = PairEvent.CONVERGING
                ps.event_frame = fc

        if ps.event in (PairEvent.NORMAL, PairEvent.CONVERGING):
            if near_contact and contact_peak >= 0.28 and kinematic >= 0.28:
                ps.event = PairEvent.IMPACT_CANDIDATE
                ps.event_frame = fc

        # Tie the impulse to the actual interaction in both time and apparent
        # road depth.  Box overlap alone is unreliable in perspective views:
        # vehicles at different depths frequently overlap in image space.
        impulse_frames = []
        if physical_a or speed_impulse_a:
            impulse_frames.append(ta.last_disc_frame)
        if physical_b or speed_impulse_b:
            impulse_frames.append(tb.last_disc_frame)
        impact_frame_hint = max(impulse_frames) if impulse_frames else fc
        align_frames = max(3, self._frames(self.config.impact_alignment_seconds))
        aligned = [
            m for m in pre_window
            if abs(m.frame - impact_frame_hint) <= align_frames
            and (m.contact >= 0.20 or m.gap_norm <= self.config.contact_gap_norm)
        ]
        if aligned:
            best_depth = min(m.bottom_diff_norm for m in aligned)
            best_anchor = min(m.anchor_dist_norm for m in aligned)
            contact_aligned = True
        else:
            best_depth = 999.0
            best_anchor = 999.0
            contact_aligned = False
        depth_ok = best_depth <= 0.90 and best_anchor <= 1.25

        # Pure synchronized slowing can be coordinated braking.  Admit it only
        # when image-space contact is newly established near the impulse, not
        # when two perspective-overlapping tracks have travelled together for
        # a long time.  Direction-changing impacts do not need this extra gate.
        contact_frames = [m.frame for m in pre_window if m.contact >= 0.50]
        if contact_frames:
            contact_onset = min(contact_frames)
            contact_age = impact_frame_hint - contact_onset
        else:
            contact_age = 9999
        recent_contact_onset = -2 <= contact_age <= max(3, self._frames(self.config.recent_contact_onset_seconds))

        # Impact-time fusion.  Geometry is mandatory; no single-track stop or
        # turn can create an event.  This is the key false-attribution guard.
        geometry_ok = (
            near_contact
            and contact_aligned
            and depth_ok
            and (contact_peak >= 0.28 or (risk_peak >= 0.62 and min_gap_metric.gap_norm <= 0.80))
            and (risk_peak >= 0.35 or contact_peak >= 0.70)
        )
        kinematics_ok = (
            physical_a
            or physical_b
            or directional_sync
            or (speed_only_sync and recent_contact_onset)
        )

        score = (
            0.31 * contact_peak
            + 0.22 * risk_peak
            + 0.38 * kinematic
            + 0.03 * flow
            + 0.06 * aftermath
        )
        if synchronized:
            score += 0.05
        elif physical_a or physical_b:
            score += 0.06
        score = float(np.clip(score, 0.0, 1.0))

        debounce = self._frames(self.config.pair_debounce_seconds)
        claimed = any(self._claimed_until.get(tid, -1) >= fc for tid in key)
        can_trigger = fc - ps.last_trigger_frame >= debounce and not claimed

        if geometry_ok and kinematics_ok and can_trigger and score >= self.impact_threshold:
            ps.event = PairEvent.CRASH_CONFIRMED
            ps.event_frame = fc
            ps.last_trigger_frame = fc
            lock = self._frames(self.config.shared_track_lock_seconds)
            self._claimed_until[a] = fc + lock
            self._claimed_until[b] = fc + lock

            # Timestamp the closest recent interaction, not the later
            # confirmation frame.  This backdates a 3-6 frame confirmation lag
            # to the actual contact moment in the saved metadata.
            physical_frames = []
            if physical_a or speed_impulse_a:
                physical_frames.append(ta.last_disc_frame)
            if physical_b or speed_impulse_b:
                physical_frames.append(tb.last_disc_frame)
            impact_frame = max(physical_frames) if physical_frames else min_gap_metric.frame
            evidence = {
                "contact": round(contact_peak, 4),
                "interaction_risk": round(risk_peak, 4),
                "kinematic": round(kinematic, 4),
                "flow": round(flow, 4),
                "aftermath": round(aftermath, 4),
                "gap_norm": round(min_gap_metric.gap_norm, 4),
                "disc_a": round(disc_a, 4),
                "disc_b": round(disc_b, 4),
                "physical_a": float(physical_a),
                "physical_b": float(physical_b),
                "depth_consistency": round(best_depth, 4),
                "anchor_distance": round(best_anchor, 4),
                "contact_aligned": float(contact_aligned),
                "mature_a": float(mature_a),
                "mature_b": float(mature_b),
                "speed_only_sync": float(speed_only_sync),
                "contact_age_frames": float(contact_age),
            }
            involved_bboxes = {}
            for tid in key:
                track_state = self._tracks.get(tid)
                if track_state and track_state.history:
                    nearest = min(
                        track_state.history,
                        key=lambda obs: abs(obs.frame - impact_frame),
                    )
                    involved_bboxes[tid] = tuple(map(float, nearest.bbox))

            sync_text = "synchronized impulses" if synchronized else "contact-coupled impulse"
            return CrashCandidate(
                score=score,
                reason="collision",
                description=(
                    f"Tracks #{a} and #{b}: {sync_text}, "
                    f"contact={contact_peak:.2f}, risk={risk_peak:.2f}"
                ),
                trigger_frame=impact_frame,
                detected_frame=fc,
                involved_tracks=[a, b],
                evidence=evidence,
                involved_bboxes=involved_bboxes,
            )

        # State cleanup / near-miss classification.  Near misses are internal;
        # they are never emitted as crash events.
        if ps.event == PairEvent.CONVERGING:
            if fc - ps.event_frame > self._frames(self.config.converging_timeout_seconds) and risk_peak < 0.25:
                ps.event = PairEvent.NEAR_MISS
                ps.event_frame = fc
        elif ps.event == PairEvent.IMPACT_CANDIDATE:
            if fc - ps.event_frame > self._frames(self.config.candidate_timeout_seconds) and score < self.impact_threshold:
                ps.event = PairEvent.NEAR_MISS
                ps.event_frame = fc
        elif ps.event == PairEvent.CRASH_CONFIRMED:
            if fc - ps.event_frame > self._frames(self.config.confirmed_to_aftermath_seconds):
                ps.event = PairEvent.AFTERMATH
                ps.event_frame = fc
        elif ps.event == PairEvent.NEAR_MISS:
            if fc - ps.event_frame > self._frames(self.config.near_miss_timeout_seconds):
                ps.event = PairEvent.NORMAL
                ps.event_frame = fc

        return None

    def _aftermath_score(self, key: tuple[int, int], impact_frame: int) -> float:
        """Supporting evidence only; never creates an event by itself."""
        vals = []
        for tid in key:
            ts = self._tracks.get(tid)
            if ts is None:
                continue
            hist = list(ts.history)
            pre_start = self._frames(self.config.aftermath_pre_start_seconds)
            pre_end = self._frames(self.config.aftermath_pre_end_seconds)
            post_window = self._frames(self.config.aftermath_post_seconds)
            pre = [o for o in hist if impact_frame - pre_start <= o.frame <= impact_frame - pre_end]
            post = [o for o in hist if self._frame_count - post_window <= o.frame <= self._frame_count]
            if len(pre) < 3 or len(post) < 3:
                continue
            vb = self._velocity(pre)
            va = self._velocity(post)
            h = max(18.0, float(np.median([o.h for o in hist[-12:]])))
            sb = math.hypot(*vb) / h
            sa = math.hypot(*va) / h
            if sb > 0.025 and sa < 0.010:
                vals.append(1.0)
            elif sb > 0.020 and sa < 0.016:
                vals.append(0.55)
            else:
                vals.append(0.0)
        return float(np.mean(vals)) if vals else 0.0

    @staticmethod
    def _physical_impulse(ts: _TrackState) -> bool:
        """Reject box-jitter residuals that do not resemble a physical impulse.

        Prediction error alone is useful as supporting evidence but is too easy
        to create with detector box jitter.  A primary impact impulse therefore
        needs either a meaningful direction change or an exceptionally large
        normalized delta-v.
        """
        # A large delta-v by itself is hard braking, not proof of impact.
        # Single-track evidence therefore requires an abrupt directional change.
        # Pure speed impulses are admitted only when *both* interacting tracks
        # show synchronized strong delta-v (handled in _advance_pair_state).
        directional = ts.last_disc_heading >= 40.0 and ts.last_disc_dv >= 0.040
        return bool(directional)

    # ------------------------------------------------------------------
    # Candidate-only local optical flow
    # ------------------------------------------------------------------

    @staticmethod
    def _local_flow_score(
        prev_gray: np.ndarray,
        gray: np.ndarray,
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        h, w = gray.shape[:2]
        x1 = max(0, int(min(box_a[0], box_b[0])))
        y1 = max(0, int(min(box_a[1], box_b[1])))
        x2 = min(w, int(max(box_a[2], box_b[2])))
        y2 = min(h, int(max(box_a[3], box_b[3])))
        bw, bh = x2 - x1, y2 - y1
        margin_x, margin_y = int(0.22 * bw), int(0.22 * bh)
        x1, y1 = max(0, x1 - margin_x), max(0, y1 - margin_y)
        x2, y2 = min(w, x2 + margin_x), min(h, y2 + margin_y)
        if x2 - x1 < 24 or y2 - y1 < 24:
            return 0.0

        p = prev_gray[y1:y2, x1:x2]
        q = gray[y1:y2, x1:x2]
        max_dim = max(p.shape)
        scale = min(1.0, 220.0 / max(max_dim, 1))
        if scale < 1.0:
            size = (max(24, int(p.shape[1] * scale)), max(24, int(p.shape[0] * scale)))
            p = cv2.resize(p, size, interpolation=cv2.INTER_AREA)
            q = cv2.resize(q, size, interpolation=cv2.INTER_AREA)

        flow = cv2.calcOpticalFlowFarneback(
            p, q, None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
        )
        fx, fy = flow[..., 0], flow[..., 1]
        medx, medy = float(np.median(fx)), float(np.median(fy))
        residual = np.hypot(fx - medx, fy - medy)
        # Mixed local motion is more useful than raw magnitude, which is high
        # during perfectly normal fast driving too.
        p90 = float(np.percentile(residual, 90))
        p50 = float(np.percentile(residual, 50))
        incoherence = max(0.0, p90 - p50)
        return float(np.clip((incoherence - 0.6) / 4.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Helpers / cleanup
    # ------------------------------------------------------------------

    def _forget_stale(self) -> None:
        track_stale = self._frames(self.config.pair_stale_seconds)
        for tid in [
            tid for tid, ts in self._tracks.items()
            if self._frame_count - ts.last_seen > track_stale
        ]:
            self._tracks.pop(tid, None)

        pair_stale = self._frames(self.config.pair_stale_seconds)
        for key in [
            key for key, ps in self._pairs.items()
            if self._frame_count - ps.last_seen > pair_stale
        ]:
            self._pairs.pop(key, None)

        for tid in [tid for tid, until in self._claimed_until.items() if until < self._frame_count]:
            self._claimed_until.pop(tid, None)

    @staticmethod
    def _bbox_gap(a, b) -> float:
        dx = max(a[0] - b[2], b[0] - a[2], 0.0)
        dy = max(a[1] - b[3], b[1] - a[3], 0.0)
        return math.hypot(dx, dy)

    @staticmethod
    def _iou(a, b) -> float:
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        return inter / max(aa + bb - inter, 1e-9)

    @staticmethod
    def _angle_delta(a: float, b: float) -> float:
        return (a - b + 180.0) % 360.0 - 180.0

# Backward-compatible re-exports; visualization state lives in its own module.
from .crash_visuals import (
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

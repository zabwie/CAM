"""Pair crash state machine — internal to the crash detector.

The state machine owns per-track kinematics, pair interaction geometry, the
temporal event state machine, and all cleanup.  It is intentionally isolated
from frame-level orchestration so the public ``CrashDetector`` facade can focus
on input filtering and candidate collection.
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
# Event types
# ---------------------------------------------------------------------------

class PairEvent(Enum):
    NORMAL = "NORMAL"
    CONVERGING = "CONVERGING"
    IMPACT_CANDIDATE = "IMPACT_CANDIDATE"
    CRASH_CONFIRMED = "CRASH_CONFIRMED"
    AFTERMATH = "AFTERMATH"
    NEAR_MISS = "NEAR_MISS"


# ---------------------------------------------------------------------------
# Internal records
# ---------------------------------------------------------------------------

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
    last_disc_speed_before_norm: float = 0.0
    last_disc_speed_after_norm: float = 0.0


@dataclass
class _PairMetric:
    frame: int
    gap_norm: float
    iou: float
    ttc_frames: float
    dmin_norm: float
    risk: float
    contact: float
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


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class PairCrashFSM:
    """Per-track kinematics, pair interaction, and temporal event machine."""

    def __init__(self, fps: float, config, max_history: int, pair_history: int) -> None:
        self.fps = fps
        self.config = config
        self._max_history = max_history
        self._pair_history = pair_history
        self._frame_count = 0
        self._tracks: dict[int, _TrackState] = {}
        self._pairs: dict[tuple[int, int], _PairState] = {}
        self._claimed_until: dict[int, int] = {}

    def reset(self) -> None:
        self._tracks.clear()
        self._pairs.clear()
        self._claimed_until.clear()
        self._frame_count = 0

    def set_fps(self, fps: float) -> None:
        self.fps = float(fps or 30.0)
        self._max_history = max(20, self._frames(self.config.history_seconds))
        self._pair_history = max(15, self._frames(self.config.pair_history_seconds))

    def _frames(self, seconds: float, minimum: int = 1) -> int:
        return max(minimum, int(round(float(seconds) * self.fps)))

    # ------------------------------------------------------------------
    # Track management
    # ------------------------------------------------------------------

    def append_track(self, tid: int, obs: _Observation) -> None:
        ts = self._tracks.get(tid)
        if ts is None:
            ts = _TrackState(history=deque(maxlen=self._max_history))
            self._tracks[tid] = ts
        if ts.history and ts.history[-1].frame == obs.frame:
            ts.history[-1] = obs
        else:
            ts.history.append(obs)
        ts.last_seen = obs.frame

    def velocity(self, entries: list[_Observation]) -> tuple[float, float]:
        """Median velocity in 30-FPS-equivalent pixels per frame."""
        if len(entries) < 2:
            return 0.0, 0.0
        rate_scale = self.fps / 30.0
        vxs, vys = [], []
        for p, q in zip(entries[:-1], entries[1:]):
            df = max(1, q.frame - p.frame)
            vxs.append(((q.anchor_x - p.anchor_x) / df) * rate_scale)
            vys.append(((q.anchor_y - p.anchor_y) / df) * rate_scale)
        return float(np.median(vxs)), float(np.median(vys))

    def recent_velocity(self, tid: int, seconds: float = 0.20) -> tuple[float, float]:
        ts = self._tracks.get(tid)
        if ts is None:
            return 0.0, 0.0
        cutoff = self._frame_count - self._frames(seconds)
        entries = [o for o in ts.history if o.frame >= cutoff]
        return self.velocity(entries)

    # ------------------------------------------------------------------
    # Discontinuity detection
    # ------------------------------------------------------------------

    def update_discontinuity(self, tid: int) -> None:
        ts = self._tracks[tid]
        hist = list(ts.history)
        min_history = max(5, self._frames(0.30))
        if len(hist) < min_history:
            return

        fc = self._frame_count
        pre_start = self._frames(self.config.discontinuity_pre_start_seconds)
        pre_end = self._frames(self.config.discontinuity_pre_end_seconds)
        post_window = self._frames(self.config.discontinuity_post_seconds)
        pre = [o for o in hist if fc - pre_start <= o.frame <= fc - pre_end]
        post = [o for o in hist if fc - post_window <= o.frame <= fc]
        # Low-rate camera feeds may only contribute two samples to a physically
        # meaningful time window.  Fall back to the nearest clean samples rather
        # than making 5-8 FPS feeds structurally impossible to analyze.
        if len(pre) < 2:
            pre = [o for o in hist if o.frame <= fc - max(1, pre_end)][-3:]
        if len(post) < 2:
            post = hist[-max(2, post_window + 1):]
        if len(pre) < 2 or len(post) < 2:
            return

        vb = self.velocity(pre)
        va = self.velocity(post)
        speed_b = math.hypot(*vb)
        speed_a = math.hypot(*va)
        scale = max(18.0, float(np.median([o.h for o in hist[-12:]])))
        dv_norm = math.hypot(va[0] - vb[0], va[1] - vb[1]) / scale

        heading_change = 0.0
        if speed_b / scale > 0.012 and speed_a / scale > 0.012:
            hb = math.degrees(math.atan2(vb[1], vb[0]))
            ha = math.degrees(math.atan2(va[1], va[0]))
            heading_change = abs(self._angle_delta(ha, hb))

        prior = hist[-7:-1]
        pred_error = 0.0
        if len(prior) >= 3:
            pv = self.velocity(prior)
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

        if score >= 0.34:
            replace_frames = self._frames(self.config.discontinuity_replace_seconds)
            if score >= ts.last_disc_score or fc - ts.last_disc_frame > replace_frames:
                ts.last_disc_frame = fc
                ts.last_disc_score = float(score)
                ts.last_disc_dv = float(dv_norm)
                ts.last_disc_heading = float(heading_change)
                ts.last_disc_pred_error = float(pred_error)
                ts.last_disc_history_span = int(hist[-1].frame - hist[0].frame)
                ts.last_disc_speed_before_norm = float(speed_b / scale)
                ts.last_disc_speed_after_norm = float(speed_a / scale)

    def _track_motion_change(
        self, ts: _TrackState
    ) -> tuple[tuple[float, float], tuple[float, float], float] | None:
        """Return robust pre/post velocities and a representative box scale."""
        hist = list(ts.history)
        if len(hist) < 4:
            return None
        fc = self._frame_count
        pre_start = self._frames(self.config.discontinuity_pre_start_seconds)
        pre_end = self._frames(self.config.discontinuity_pre_end_seconds)
        post_window = self._frames(self.config.discontinuity_post_seconds)
        pre = [o for o in hist if fc - pre_start <= o.frame <= fc - pre_end]
        post = [o for o in hist if fc - post_window <= o.frame <= fc]
        if len(pre) < 2:
            pre = [o for o in hist if o.frame <= fc - max(1, pre_end)][-3:]
        if len(post) < 2:
            post = hist[-max(2, post_window + 1):]
        if len(pre) < 2 or len(post) < 2:
            return None
        scale = max(18.0, float(np.median([o.h for o in hist[-12:]])))
        return self.velocity(pre), self.velocity(post), scale

    def _pair_motion_change(
        self, ta: _TrackState, tb: _TrackState
    ) -> tuple[float, float, float, float, float]:
        """Measure whether the pair's relative motion changed at impact time.

        Returns ``(relative_dv_norm, delta_v_cosine, pre_heading_cosine,
        speed_drop_a, speed_drop_b)``.  Common-mode braking has a high
        delta-v cosine but very small relative-dv change; impacts generally
        alter the relative motion of the interacting pair.
        """
        ma = self._track_motion_change(ta)
        mb = self._track_motion_change(tb)
        if ma is None or mb is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        va0, va1, sa = ma
        vb0, vb1, sb = mb
        scale = max(18.0, 0.5 * (sa + sb))

        rel0 = (vb0[0] - va0[0], vb0[1] - va0[1])
        rel1 = (vb1[0] - va1[0], vb1[1] - va1[1])
        relative_dv_norm = math.hypot(rel1[0] - rel0[0], rel1[1] - rel0[1]) / scale

        dva = (va1[0] - va0[0], va1[1] - va0[1])
        dvb = (vb1[0] - vb0[0], vb1[1] - vb0[1])

        def cosine(u, v) -> float:
            denom = math.hypot(*u) * math.hypot(*v)
            return 0.0 if denom <= 1e-9 else float(np.clip((u[0] * v[0] + u[1] * v[1]) / denom, -1.0, 1.0))

        delta_v_cosine = cosine(dva, dvb)
        pre_heading_cosine = cosine(va0, vb0)
        speed_a0, speed_a1 = math.hypot(*va0), math.hypot(*va1)
        speed_b0, speed_b1 = math.hypot(*vb0), math.hypot(*vb1)
        speed_drop_a = max(0.0, (speed_a0 - speed_a1) / max(speed_a0, 1e-6))
        speed_drop_b = max(0.0, (speed_b0 - speed_b1) / max(speed_b0, 1e-6))
        return (
            float(relative_dv_norm),
            float(delta_v_cosine),
            float(pre_heading_cosine),
            float(speed_drop_a),
            float(speed_drop_b),
        )

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------

    def advance_pair_state(
        self, key: tuple[int, int], ps: _PairState
    ):
        """Advance one pair's state machine.

        Returns a ``CrashCandidate`` dict on CRASH_CONFIRMED, otherwise None.
        The caller ``CrashDetector`` converts the dict to a ``CrashCandidate``.
        """
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
        # Keep the time-based maturity requirement while allowing low-rate
        # feeds to become mature with a realistic number of observations.
        maturity_frames = max(4, self._frames(self.config.maturity_seconds))
        mature_a = ta.last_disc_history_span >= maturity_frames
        mature_b = tb.last_disc_history_span >= maturity_frames

        physical_a = mature_a and self._physical_impulse(ta) if disc_a > 0 else False
        physical_b = mature_b and self._physical_impulse(tb) if disc_b > 0 else False
        speed_impulse_a = mature_a and disc_a >= 0.34 and ta.last_disc_dv >= 0.090
        speed_impulse_b = mature_b and disc_b >= 0.34 and tb.last_disc_dv >= 0.090
        (
            pair_relative_dv,
            delta_v_cosine,
            pre_heading_cosine,
            speed_drop_a,
            speed_drop_b,
        ) = self._pair_motion_change(ta, tb)
        common_braking_motion = (
            pre_heading_cosine >= self.config.common_heading_cosine
            and delta_v_cosine >= self.config.common_braking_cosine
            and speed_drop_a >= self.config.common_braking_min_drop
            and speed_drop_b >= self.config.common_braking_min_drop
            and pair_relative_dv < self.config.common_braking_max_relative_dv_norm
        )
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
            and pair_relative_dv >= self.config.speed_only_relative_dv_norm
        )
        synchronized = directional_sync or speed_only_sync

        if synchronized:
            kinematic = 0.62 * max(disc_a, disc_b) + 0.38 * min(disc_a, disc_b)
        else:
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

        contact_frames = [m.frame for m in pre_window if m.contact >= 0.50]
        if contact_frames:
            contact_onset = min(contact_frames)
            contact_age = impact_frame_hint - contact_onset
        else:
            contact_age = 9999
        recent_contact_onset = -2 <= contact_age <= max(3, self._frames(self.config.recent_contact_onset_seconds))

        pair_impact_evidence = (
            pair_relative_dv >= self.config.min_pair_relative_dv_norm
            or recent_contact_onset
            or contact_dropout
            or flow >= 0.25
        )
        stale_contact = contact_age > max(
            3, self._frames(self.config.recent_contact_onset_seconds)
        )
        common_braking = common_braking_motion and stale_contact
        if stale_contact and not contact_dropout and flow < 0.25:
            # Long-standing 2-D box overlap is common in traffic queues and
            # perspective compression.  It is not fresh collision evidence.
            pair_impact_evidence = (
                pair_relative_dv >= self.config.stale_contact_relative_dv_norm
            )

        geometry_ok = (
            near_contact
            and contact_aligned
            and depth_ok
            and (contact_peak >= 0.28 or (risk_peak >= 0.62 and min_gap_metric.gap_norm <= 0.80))
            and (risk_peak >= 0.35 or contact_peak >= 0.70)
        )
        kinematics_ok = (
            (
                physical_a
                or physical_b
                or directional_sync
                or speed_only_sync
            )
            and pair_impact_evidence
            and not common_braking
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

        if geometry_ok and kinematics_ok and can_trigger and score >= getattr(self.config, 'impact_threshold', 0.64):
            ps.event = PairEvent.CRASH_CONFIRMED
            ps.event_frame = fc
            ps.last_trigger_frame = fc
            lock = self._frames(self.config.shared_track_lock_seconds)
            self._claimed_until[a] = fc + lock
            self._claimed_until[b] = fc + lock

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
                "pair_relative_dv": round(pair_relative_dv, 4),
                "delta_v_cosine": round(delta_v_cosine, 4),
                "pre_heading_cosine": round(pre_heading_cosine, 4),
                "common_braking": float(common_braking),
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
            return {
                "type": "crash",
                "score": float(score),
                "reason": "collision",
                "description": (
                    f"Tracks #{a} and #{b}: {sync_text}, "
                    f"contact={contact_peak:.2f}, risk={risk_peak:.2f}"
                ),
                "trigger_frame": int(impact_frame),
                "detected_frame": int(fc),
                "involved_tracks": [a, b],
                "evidence": evidence,
                "involved_bboxes": involved_bboxes,
            }

        # State transition cleanup
        if ps.event == PairEvent.CONVERGING:
            if fc - ps.event_frame > self._frames(self.config.converging_timeout_seconds) and risk_peak < 0.25:
                ps.event = PairEvent.NEAR_MISS
                ps.event_frame = fc
        elif ps.event == PairEvent.IMPACT_CANDIDATE:
            if fc - ps.event_frame > self._frames(self.config.candidate_timeout_seconds) and score < getattr(self.config, 'impact_threshold', 0.64):
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
            vb = self.velocity(pre)
            va = self.velocity(post)
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
        """Directional discontinuity with meaningful delta-v."""
        # Heading is unstable when a vehicle has nearly stopped; detector jitter
        # can otherwise turn ordinary braking into a fictitious 90-degree turn.
        directional = (
            ts.last_disc_heading >= 40.0
            and ts.last_disc_dv >= 0.040
            and ts.last_disc_speed_after_norm >= 0.020
        )
        return bool(directional)

    # ------------------------------------------------------------------
    # Optical flow
    # ------------------------------------------------------------------

    @staticmethod
    def local_flow_score(
        prev_gray: np.ndarray,
        gray: np.ndarray,
        box_a: tuple[float, float, float, float],
        box_b: tuple[float, float, float, float],
    ) -> float:
        """Candidate-only local optical flow incoherence score."""
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
        p90 = float(np.percentile(residual, 90))
        p50 = float(np.percentile(residual, 50))
        incoherence = max(0.0, p90 - p50)
        return float(np.clip((incoherence - 0.6) / 4.0, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def forget_stale(self) -> None:
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

    # ------------------------------------------------------------------
    # Accessors for the facade
    # ------------------------------------------------------------------

    @property
    def tracks(self) -> dict:
        return self._tracks

    @property
    def pairs(self) -> dict:
        return self._pairs

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @frame_count.setter
    def frame_count(self, value: int) -> None:
        self._frame_count = int(value)

    @property
    def claimed_until(self) -> dict:
        return self._claimed_until

    @property
    def history_seconds(self) -> float:
        return self.config.history_seconds

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

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

"""Robust world-space vehicle speed estimation.

The estimator deliberately waits for a useful trajectory window instead of
turning frame-to-frame detector jitter into an MPH reading.  It also rejects
impossible point jumps, fits velocity across the whole recent trajectory, and
limits displayed acceleration so a noisy detection cannot make the readout
teleport between speeds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

MPS_TO_MPH = 2.2369362920544


@dataclass(frozen=True)
class SpeedEstimatorConfig:
    min_window_seconds: float = 0.55
    max_window_seconds: float = 1.35
    max_track_gap_seconds: float = 0.50
    min_samples: int = 8
    max_instant_speed_mph: float = 160.0
    max_output_speed_mph: float = 130.0
    max_accel_mph_per_second: float = 24.0
    max_brake_mph_per_second: float = 36.0
    smoothing_seconds: float = 0.45
    residual_floor_meters: float = 0.18
    residual_sigma: float = 3.5


@dataclass
class _TrackState:
    history: Deque[tuple[int, float, float]]
    displayed_mph: Optional[float] = None
    last_display_frame: Optional[int] = None
    rejected_jump_streak: int = 0
    last_reason: str = "VALID"
    last_fit_quality: float = 0.0


class RobustSpeedEstimator:
    """Estimate stable MPH values from frame-indexed world coordinates."""

    def __init__(self, fps: float, config: Optional[SpeedEstimatorConfig] = None):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)
        self.config = config or SpeedEstimatorConfig()
        self._tracks: dict[int, _TrackState] = {}

    def set_fps(self, fps: float) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.fps = float(fps)

    def reset(self) -> None:
        self._tracks.clear()

    def forget_stale(self, current_frame: int) -> None:
        """Drop tracks that have not been seen for twice the allowed gap."""
        stale_frames = max(1, int(round(self.config.max_track_gap_seconds * self.fps * 2.0)))
        stale_ids = []
        for tid, state in self._tracks.items():
            if not state.history or current_frame - state.history[-1][0] > stale_frames:
                stale_ids.append(tid)
        for tid in stale_ids:
            del self._tracks[tid]

    def update(self, track_id: int, frame: int, x_m: float, y_m: float) -> Optional[float]:
        """Add one world-space sample and return stable MPH when reliable.

        ``None`` means there is not yet enough trustworthy motion history to
        show a speed.  This is intentionally different from a valid 0 mph.
        """
        if not np.isfinite([x_m, y_m]).all():
            state = self._tracks.get(track_id)
            if state:
                state.last_reason = "NON_FINITE_POSITION"
            return None

        state = self._tracks.get(track_id)
        if state is None:
            state = _TrackState(history=deque())
            self._tracks[track_id] = state

        state.last_reason = "VALID"

        if state.history:
            prev_frame, prev_x, prev_y = state.history[-1]
            df = frame - prev_frame
            if df <= 0:
                return state.displayed_mph

            gap_seconds = df / self.fps
            if gap_seconds > self.config.max_track_gap_seconds:
                state.last_reason = "TRACK_GAP"
                state.history.clear()
                state.displayed_mph = None
                state.last_display_frame = None
            else:
                dist_m = float(np.hypot(x_m - prev_x, y_m - prev_y))
                instant_mph = (dist_m / gap_seconds) * MPS_TO_MPH
                if instant_mph > self.config.max_instant_speed_mph:
                    state.rejected_jump_streak += 1
                    if state.rejected_jump_streak == 1:
                        state.last_reason = "JUMP_REJECTED"
                        return state.displayed_mph
                    state.last_reason = "TRACK_RESET"
                    state.history.clear()
                    state.displayed_mph = None
                    state.last_display_frame = None
                    state.rejected_jump_streak = 0
                else:
                    state.rejected_jump_streak = 0

        state.history.append((int(frame), float(x_m), float(y_m)))
        max_age_frames = max(1, int(round(self.config.max_window_seconds * self.fps)))
        while state.history and frame - state.history[0][0] > max_age_frames:
            state.history.popleft()

        raw_mph, fit_reason, fit_quality = self._fit_speed(state.history)
        state.last_reason = fit_reason
        state.last_fit_quality = fit_quality
        if raw_mph is None:
            return None
        if raw_mph > self.config.max_output_speed_mph:
            state.last_reason = "EXCESSIVE_SPEED"
            return state.displayed_mph

        state.last_reason = "VALID"
        if state.displayed_mph is None or state.last_display_frame is None:
            state.displayed_mph = raw_mph
            state.last_display_frame = frame
            return state.displayed_mph

        dt = max((frame - state.last_display_frame) / self.fps, 1.0 / self.fps)
        delta = raw_mph - state.displayed_mph
        max_up = self.config.max_accel_mph_per_second * dt
        max_down = self.config.max_brake_mph_per_second * dt
        limited = state.displayed_mph + float(np.clip(delta, -max_down, max_up))

        tau = max(self.config.smoothing_seconds, 1e-6)
        alpha = 1.0 - float(np.exp(-dt / tau))
        state.displayed_mph += alpha * (limited - state.displayed_mph)
        state.last_display_frame = frame
        return max(0.0, state.displayed_mph)

    def _fit_speed(
        self, history: Deque[tuple[int, float, float]]
    ) -> tuple[Optional[float], str, float]:
        """Return ``(speed_mph, reason, fit_quality)``.

        ``fit_quality`` is based on inlier ratio, residual magnitude, and time
        coverage.  It is kept separate from the speed value so downstream code
        can distinguish a valid-but-weak estimate from a stable trajectory.
        """
        cfg = self.config
        if len(history) < cfg.min_samples:
            return (None, "INSUFFICIENT_SAMPLES", 0.0)

        arr = np.asarray(history, dtype=np.float64)
        frames = arr[:, 0]
        duration = (frames[-1] - frames[0]) / self.fps
        if duration < cfg.min_window_seconds:
            return (None, "SHORT_DURATION", 0.0)

        t = (frames - frames[0]) / self.fps
        xy = arr[:, 1:3]
        A = np.column_stack((t, np.ones_like(t)))

        coeff, _, _, _ = np.linalg.lstsq(A, xy, rcond=None)
        predicted = A @ coeff
        residual = np.linalg.norm(xy - predicted, axis=1)

        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        robust_sigma = 1.4826 * mad
        threshold = max(cfg.residual_floor_meters, med + cfg.residual_sigma * robust_sigma)
        inliers = residual <= threshold

        min_inliers = max(6, int(np.ceil(cfg.min_samples * 0.75)))
        if int(inliers.sum()) < min_inliers:
            return (None, "TRAJECTORY_OUTLIER", 0.0)

        t_in = t[inliers]
        xy_in = xy[inliers]
        if (t_in[-1] - t_in[0]) < cfg.min_window_seconds * 0.8:
            return (None, "SHORT_POST_OUTLIER", 0.0)

        A_in = np.column_stack((t_in, np.ones_like(t_in)))
        coeff_in, _, _, _ = np.linalg.lstsq(A_in, xy_in, rcond=None)
        vx, vy = coeff_in[0]
        speed_mph = float(np.hypot(vx, vy) * MPS_TO_MPH)
        if not np.isfinite(speed_mph):
            return (None, "NON_FINITE_FIT", 0.0)
        inlier_ratio = float(np.mean(inliers))
        median_residual = float(np.median(residual[inliers])) if np.any(inliers) else 999.0
        residual_score = float(np.exp(-median_residual / max(cfg.residual_floor_meters * 2.0, 1e-6)))
        duration_score = float(np.clip(duration / cfg.max_window_seconds, 0.0, 1.0))
        fit_quality = float(np.clip(
            0.45 * inlier_ratio + 0.35 * residual_score + 0.20 * duration_score,
            0.0,
            1.0,
        ))
        return (speed_mph, "VALID", fit_quality)

    def last_reason(self, track_id: int) -> str:
        """Return the reason code for the most recent update() call on a track.

        Returns one of:
          VALID, NON_FINITE_POSITION, TRACK_GAP, JUMP_REJECTED, TRACK_RESET,
          INSUFFICIENT_SAMPLES, SHORT_DURATION, TRAJECTORY_OUTLIER,
          SHORT_POST_OUTLIER, NON_FINITE_FIT, EXCESSIVE_SPEED
        """
        state = self._tracks.get(track_id)
        if state is None:
            return "NO_TRACK"
        return state.last_reason

    def trajectory_confidence(self, track_id: int, current_frame: int) -> float:
        """Score the trajectory quality of a tracked vehicle (0-1).

        Measures how much trustworthy motion history exists for this track.
        A high score means long, dense, stable tracking; low means the track
        just appeared or has suffered rejected jumps.
        """
        state = self._tracks.get(track_id)
        if not state or not state.history or len(state.history) < self.config.min_samples:
            return 0.0

        arr = np.asarray(state.history, dtype=np.float64)
        frames = arr[:, 0]
        duration = (frames[-1] - frames[0]) / self.fps

        # Normalised duration — longer observation → higher confidence.
        target = self.config.max_window_seconds * 0.7
        dur_score = min(duration / target, 1.0) if target > 0 else 0.0

        # Sample-density relative to the ideal FPS rate.
        expected = max(1, duration * self.fps)
        density = len(state.history) / expected
        density_score = min(density, 1.0)

        # Deduction for rejected jumps (consecutive implausible positions).
        reject_penalty = max(0.0, 1.0 - state.rejected_jump_streak * 0.35)

        return float(np.clip(
            dur_score * 0.30
            + density_score * 0.20
            + reject_penalty * 0.15
            + state.last_fit_quality * 0.35,
            0.0,
            1.0,
        ))

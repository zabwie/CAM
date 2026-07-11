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
            return None

        state = self._tracks.get(track_id)
        if state is None:
            state = _TrackState(history=deque())
            self._tracks[track_id] = state

        if state.history:
            prev_frame, prev_x, prev_y = state.history[-1]
            df = frame - prev_frame
            if df <= 0:
                return state.displayed_mph

            gap_seconds = df / self.fps
            if gap_seconds > self.config.max_track_gap_seconds:
                state.history.clear()
                state.displayed_mph = None
                state.last_display_frame = None
            else:
                dist_m = float(np.hypot(x_m - prev_x, y_m - prev_y))
                instant_mph = (dist_m / gap_seconds) * MPS_TO_MPH
                # A single detector/tracker jump is ignored rather than fed
                # into the trajectory fit. The track remains alive.
                if instant_mph > self.config.max_instant_speed_mph:
                    state.rejected_jump_streak += 1
                    # One bad detector box should not erase a good speed lock.
                    # Repeated incompatible positions are more likely an ID switch
                    # or track teleport, so force a clean re-lock on the new path.
                    if state.rejected_jump_streak == 1:
                        return state.displayed_mph
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

        raw_mph = self._fit_speed(state.history)
        if raw_mph is None:
            return None
        if raw_mph > self.config.max_output_speed_mph:
            return state.displayed_mph

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

    def _fit_speed(self, history: Deque[tuple[int, float, float]]) -> Optional[float]:
        cfg = self.config
        if len(history) < cfg.min_samples:
            return None

        arr = np.asarray(history, dtype=np.float64)
        frames = arr[:, 0]
        duration = (frames[-1] - frames[0]) / self.fps
        if duration < cfg.min_window_seconds:
            return None

        t = (frames - frames[0]) / self.fps
        xy = arr[:, 1:3]
        A = np.column_stack((t, np.ones_like(t)))

        # Initial 2D linear-motion fit.
        coeff, _, _, _ = np.linalg.lstsq(A, xy, rcond=None)
        predicted = A @ coeff
        residual = np.linalg.norm(xy - predicted, axis=1)

        # Robust residual gate using median absolute deviation.
        med = float(np.median(residual))
        mad = float(np.median(np.abs(residual - med)))
        robust_sigma = 1.4826 * mad
        threshold = max(cfg.residual_floor_meters, med + cfg.residual_sigma * robust_sigma)
        inliers = residual <= threshold

        min_inliers = max(6, int(np.ceil(cfg.min_samples * 0.75)))
        if int(inliers.sum()) < min_inliers:
            return None

        t_in = t[inliers]
        xy_in = xy[inliers]
        if (t_in[-1] - t_in[0]) < cfg.min_window_seconds * 0.8:
            return None

        A_in = np.column_stack((t_in, np.ones_like(t_in)))
        coeff_in, _, _, _ = np.linalg.lstsq(A_in, xy_in, rcond=None)
        vx, vy = coeff_in[0]
        speed_mph = float(np.hypot(vx, vy) * MPS_TO_MPH)
        return speed_mph if np.isfinite(speed_mph) else None

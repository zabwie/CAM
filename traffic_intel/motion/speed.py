"""Robust world-space vehicle speed estimation.

The estimator waits for a useful trajectory window instead of turning
frame-to-frame detector jitter into an MPH reading. Live callers may provide a
monotonic observation timestamp so dropped/skipped frames do not bias speed.
The original frame-indexed API remains supported for deterministic replay.
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
    # (monotonic timestamp seconds, x metres, y metres)
    history: Deque[tuple[float, float, float]]
    last_seen_frame: int = 0
    last_seen_timestamp: float = 0.0
    displayed_mph: Optional[float] = None
    last_display_timestamp: Optional[float] = None
    rejected_jump_streak: int = 0
    last_reason: str = "VALID"
    last_fit_quality: float = 0.0


class RobustSpeedEstimator:
    """Estimate stable MPH values from timestamped world coordinates.

    ``frame`` is retained for compatibility and stale-track bookkeeping.
    ``timestamp_s`` should be a monotonic capture/receipt timestamp for live
    streams. When omitted, ``frame / fps`` is used exactly as before.
    """

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

    def _resolve_timestamp(self, frame: int, timestamp_s: float | None) -> float:
        if timestamp_s is None:
            return float(frame) / self.fps
        value = float(timestamp_s)
        if not np.isfinite(value):
            raise ValueError("timestamp_s must be finite")
        return value

    def forget_stale(
        self,
        current_frame: int,
        current_timestamp_s: float | None = None,
    ) -> None:
        """Drop tracks not seen for twice the configured gap.

        Passing a timestamp makes this independent of inference cadence. The
        frame-only behavior remains available for replay/legacy callers.
        """
        now = self._resolve_timestamp(current_frame, current_timestamp_s)
        stale_seconds = self.config.max_track_gap_seconds * 2.0
        stale_ids = [
            tid
            for tid, state in self._tracks.items()
            if not state.history or now - state.last_seen_timestamp > stale_seconds
        ]
        for tid in stale_ids:
            del self._tracks[tid]

    def update(
        self,
        track_id: int,
        frame: int,
        x_m: float,
        y_m: float,
        *,
        timestamp_s: float | None = None,
    ) -> Optional[float]:
        """Add one world-space sample and return stable MPH when reliable.

        ``None`` means there is not yet enough trustworthy motion history to
        show a speed. This is intentionally different from a valid 0 mph.
        """
        timestamp = self._resolve_timestamp(frame, timestamp_s)
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
            prev_timestamp, prev_x, prev_y = state.history[-1]
            gap_seconds = timestamp - prev_timestamp
            if gap_seconds <= 0:
                state.last_reason = "NON_MONOTONIC_TIMESTAMP"
                return state.displayed_mph

            if gap_seconds > self.config.max_track_gap_seconds:
                state.last_reason = "TRACK_GAP"
                state.history.clear()
                state.displayed_mph = None
                state.last_display_timestamp = None
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
                    state.last_display_timestamp = None
                    state.rejected_jump_streak = 0
                else:
                    state.rejected_jump_streak = 0

        state.history.append((timestamp, float(x_m), float(y_m)))
        state.last_seen_frame = int(frame)
        state.last_seen_timestamp = timestamp
        while state.history and timestamp - state.history[0][0] > self.config.max_window_seconds:
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
        if state.displayed_mph is None or state.last_display_timestamp is None:
            state.displayed_mph = raw_mph
            state.last_display_timestamp = timestamp
            return state.displayed_mph

        dt = max(timestamp - state.last_display_timestamp, 1.0 / self.fps)
        delta = raw_mph - state.displayed_mph
        max_up = self.config.max_accel_mph_per_second * dt
        max_down = self.config.max_brake_mph_per_second * dt
        limited = state.displayed_mph + float(np.clip(delta, -max_down, max_up))

        tau = max(self.config.smoothing_seconds, 1e-6)
        alpha = 1.0 - float(np.exp(-dt / tau))
        state.displayed_mph += alpha * (limited - state.displayed_mph)
        state.last_display_timestamp = timestamp
        return max(0.0, state.displayed_mph)

    def _fit_speed(
        self, history: Deque[tuple[float, float, float]]
    ) -> tuple[Optional[float], str, float]:
        """Return ``(speed_mph, reason, fit_quality)``."""
        cfg = self.config
        if len(history) < cfg.min_samples:
            return (None, "INSUFFICIENT_SAMPLES", 0.0)

        arr = np.asarray(history, dtype=np.float64)
        timestamps = arr[:, 0]
        duration = float(timestamps[-1] - timestamps[0])
        if duration < cfg.min_window_seconds:
            return (None, "SHORT_DURATION", 0.0)

        t = timestamps - timestamps[0]
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
        residual_score = float(
            np.exp(-median_residual / max(cfg.residual_floor_meters * 2.0, 1e-6))
        )
        duration_score = float(np.clip(duration / cfg.max_window_seconds, 0.0, 1.0))
        fit_quality = float(
            np.clip(
                0.45 * inlier_ratio + 0.35 * residual_score + 0.20 * duration_score,
                0.0,
                1.0,
            )
        )
        return (speed_mph, "VALID", fit_quality)

    def last_reason(self, track_id: int) -> str:
        state = self._tracks.get(track_id)
        if state is None:
            return "NO_TRACK"
        return state.last_reason

    def trajectory_confidence(
        self,
        track_id: int,
        current_frame: int,
        current_timestamp_s: float | None = None,
    ) -> float:
        """Score trajectory quality (0-1), using real elapsed time when given."""
        state = self._tracks.get(track_id)
        if not state or not state.history or len(state.history) < self.config.min_samples:
            return 0.0

        arr = np.asarray(state.history, dtype=np.float64)
        timestamps = arr[:, 0]
        duration = float(timestamps[-1] - timestamps[0])

        target = self.config.max_window_seconds * 0.7
        dur_score = min(duration / target, 1.0) if target > 0 else 0.0

        # Expected sample count is based on configured source FPS only as a
        # density reference; elapsed time itself is timestamp-derived.
        expected = max(1.0, duration * self.fps)
        density_score = min(len(state.history) / expected, 1.0)
        reject_penalty = max(0.0, 1.0 - state.rejected_jump_streak * 0.35)

        return float(
            np.clip(
                dur_score * 0.30
                + density_score * 0.20
                + reject_penalty * 0.15
                + state.last_fit_quality * 0.35,
                0.0,
                1.0,
            )
        )

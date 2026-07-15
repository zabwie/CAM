from __future__ import annotations

import pytest

from traffic_intel.motion.speed import RobustSpeedEstimator, SpeedEstimatorConfig


def _config() -> SpeedEstimatorConfig:
    return SpeedEstimatorConfig(
        min_window_seconds=0.40,
        max_window_seconds=1.50,
        max_track_gap_seconds=0.60,
        min_samples=6,
        smoothing_seconds=0.01,
    )


def test_irregular_live_timestamps_do_not_inflate_speed() -> None:
    estimator = RobustSpeedEstimator(30, _config())
    timestamps = [0.00, 0.04, 0.11, 0.18, 0.26, 0.35, 0.45, 0.56, 0.68, 0.81]
    result = None
    for frame, timestamp in enumerate(timestamps, start=1):
        result = estimator.update(
            7,
            frame,
            10.0 * timestamp,
            0.0,
            timestamp_s=timestamp,
        )
    assert result is not None
    assert result == pytest.approx(22.369, rel=0.03)


def test_frame_index_api_remains_backward_compatible() -> None:
    estimator = RobustSpeedEstimator(30, _config())
    result = None
    for frame in range(1, 31):
        timestamp = frame / 30.0
        result = estimator.update(1, frame, 10.0 * timestamp, 0.0)
    assert result is not None
    assert result == pytest.approx(22.369, rel=0.03)


def test_non_monotonic_timestamp_is_rejected_without_corrupting_track() -> None:
    estimator = RobustSpeedEstimator(30, _config())
    estimator.update(1, 1, 0.0, 0.0, timestamp_s=1.0)
    assert estimator.update(1, 2, 1.0, 0.0, timestamp_s=0.9) is None
    assert estimator.last_reason(1) == "NON_MONOTONIC_TIMESTAMP"

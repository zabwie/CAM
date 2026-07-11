import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "traffic_intel"))
from speed import MPS_TO_MPH, RobustSpeedEstimator, SpeedEstimatorConfig


def feed_constant_speed(estimator, mph, fps=30, seconds=3.0, jitter=0.0, tid=1):
    rng = np.random.default_rng(7)
    mps = mph / MPS_TO_MPH
    outputs = []
    for frame in range(int(fps * seconds)):
        t = frame / fps
        x = mps * t + (rng.normal(0, jitter) if jitter else 0.0)
        y = rng.normal(0, jitter) if jitter else 0.0
        outputs.append(estimator.update(tid, frame, x, y))
    return [x for x in outputs if x is not None]


def test_constant_65_mph_is_accurate():
    est = RobustSpeedEstimator(30)
    values = feed_constant_speed(est, 65.0)
    assert values
    assert abs(values[-1] - 65.0) < 0.5


def test_jitter_does_not_create_wild_mph():
    est = RobustSpeedEstimator(30)
    values = feed_constant_speed(est, 65.0, jitter=0.06)
    assert values
    assert 60.0 <= values[-1] <= 70.0
    assert max(values) < 75.0


def test_one_impossible_position_jump_is_rejected():
    est = RobustSpeedEstimator(30)
    mps = 55.0 / MPS_TO_MPH
    values = []
    for frame in range(100):
        x = mps * (frame / 30)
        if frame == 55:
            x += 100.0
        value = est.update(3, frame, x, 0.0)
        if value is not None:
            values.append(value)
    assert values
    assert 50.0 <= values[-1] <= 60.0
    assert max(values) < 65.0


def test_track_gap_resets_speed_lock():
    est = RobustSpeedEstimator(30)
    values = feed_constant_speed(est, 45.0, seconds=1.5, tid=9)
    assert values
    # A long absence followed by a far-away reacquisition must not bridge distance.
    assert est.update(9, 100, 500.0, 500.0) is None


def test_no_reading_before_minimum_window():
    est = RobustSpeedEstimator(30)
    mps = 70.0 / MPS_TO_MPH
    for frame in range(12):
        assert est.update(4, frame, mps * frame / 30, 0.0) is None


def test_stationary_jitter_stays_near_zero():
    est = RobustSpeedEstimator(30)
    rng = np.random.default_rng(11)
    values = []
    for frame in range(120):
        value = est.update(12, frame, rng.normal(0, 0.025), rng.normal(0, 0.025))
        if value is not None:
            values.append(value)
    assert values
    assert values[-1] < 1.0
    assert max(values) < 1.5


def test_persistent_track_teleport_forces_relock():
    est = RobustSpeedEstimator(30)
    mps = 50.0 / MPS_TO_MPH
    locked = []
    for frame in range(60):
        value = est.update(21, frame, mps * frame / 30, 0.0)
        if value is not None:
            locked.append(value)
    assert locked

    # First impossible point is treated as one bad box; the second confirms
    # that the track has moved to a different trajectory and clears the lock.
    first = est.update(21, 60, 300.0, 0.0)
    second = est.update(21, 61, 301.0, 0.0)
    assert first is not None
    assert second is None

    relocked = []
    for frame in range(62, 100):
        x = 301.0 + mps * ((frame - 61) / 30)
        value = est.update(21, frame, x, 0.0)
        if value is not None:
            relocked.append(value)
    assert relocked
    assert 45.0 <= relocked[-1] <= 55.0

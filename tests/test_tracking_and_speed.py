from __future__ import annotations

import numpy as np

from traffic_intel.config import TrackingConfig
from traffic_intel.core.tracking import TrackQualityGate
from traffic_intel.motion.speed import RobustSpeedEstimator


def test_track_quality_rejects_large_id_jump_then_reacquires() -> None:
    gate = TrackQualityGate(
        fps=30,
        config=TrackingConfig(min_hits=3, min_confidence=0.20, reacquire_hits=2),
    )
    for frame in range(1, 6):
        assessment = gate.update(
            frame=frame,
            track_id=7,
            class_id=2,
            confidence=0.90,
            bbox=np.array([100 + frame, 100, 160 + frame, 140], dtype=np.float32),
        )
    assert assessment.confirmed

    jumped = gate.update(
        frame=6,
        track_id=7,
        class_id=2,
        confidence=0.90,
        bbox=np.array([700, 100, 760, 140], dtype=np.float32),
    )
    assert not jumped.confirmed
    assert jumped.instability > 0

    confirmations = []
    for frame in range(7, 14):
        assessment = gate.update(
            frame=frame,
            track_id=7,
            class_id=2,
            confidence=0.90,
            bbox=np.array([700 + frame, 100, 760 + frame, 140], dtype=np.float32),
        )
        confirmations.append(assessment.confirmed)
    assert confirmations[-1]


def test_speed_estimator_is_stable_for_constant_world_velocity() -> None:
    estimator = RobustSpeedEstimator(fps=30)
    values = []
    # 10 m/s = 22.369 mph.
    for frame in range(1, 70):
        speed = estimator.update(1, frame, x_m=(frame / 30.0) * 10.0, y_m=0.0)
        if speed is not None:
            values.append(speed)
    assert values
    assert abs(values[-1] - 22.369) < 0.75
    assert estimator.trajectory_confidence(1, 69) > 0.70


def test_speed_estimator_rejects_single_extreme_position_jump() -> None:
    estimator = RobustSpeedEstimator(fps=30)
    for frame in range(1, 30):
        estimator.update(1, frame, x_m=(frame / 30.0) * 8.0, y_m=0.0)
    before = estimator.update(1, 30, x_m=8.0, y_m=0.0)
    after_jump = estimator.update(1, 31, x_m=1000.0, y_m=0.0)
    assert after_jump == before
    assert estimator.last_reason(1) == "JUMP_REJECTED"

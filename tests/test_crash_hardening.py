from __future__ import annotations

import random

import cv2
import numpy as np
import pytest

from traffic_intel.domain import Detection
from traffic_intel.incident.crash_detector import CrashDetector


def _det(
    frame: int,
    tid: int,
    cx: float,
    cy: float,
    *,
    jitter_x: float = 0.0,
    jitter_y: float = 0.0,
    w: float = 60.0,
    h: float = 40.0,
    quality: float = 0.85,
) -> Detection:
    return Detection(
        frame=frame,
        track_id=tid,
        class_name="car",
        confidence=0.85,
        bbox=(
            cx - w / 2 + jitter_x,
            cy - h / 2 + jitter_y,
            cx + w / 2 + jitter_x,
            cy + h / 2 + jitter_y,
        ),
        track_quality=quality,
        track_confirmed=True,
    )


def _hard_braking_trial(seed: int, jitter: float, *, optical_flow: bool = False):
    rng = random.Random(seed)
    detector = CrashDetector(fps=30)
    events = []
    lead_x = 300.0
    rear_x = 225.0

    texture = np.random.default_rng(seed).integers(0, 32, size=(480, 900), dtype=np.uint8)
    texture = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

    for frame in range(1, 91):
        lead_x += 4.0 if frame < 48 else 0.3
        rear_x += 4.8 if frame < 48 else 0.4
        lead_jx, lead_jy = rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)
        rear_jx, rear_jy = rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)
        detections = [
            _det(frame, 1, lead_x, 300, jitter_x=lead_jx, jitter_y=lead_jy),
            _det(frame, 2, rear_x, 300, jitter_x=rear_jx, jitter_y=rear_jy),
        ]

        image = None
        if optical_flow:
            image = texture.copy()
            for det in detections:
                x1, y1, x2, y2 = map(int, det.bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (180, 180, 180), -1)
        events.extend(detector.update(frame, detections, frame=image))
    return events


@pytest.mark.parametrize("jitter", [0.5, 1.0, 1.5])
def test_common_mode_hard_braking_with_detection_jitter_is_not_a_crash(jitter: float) -> None:
    for seed in range(40):
        assert _hard_braking_trial(seed, jitter) == []


def test_common_mode_hard_braking_with_optical_flow_is_not_a_crash() -> None:
    for seed in range(5):
        assert _hard_braking_trial(seed, 1.0, optical_flow=True) == []


@pytest.mark.parametrize("fps", [5, 6, 7, 8])
def test_low_fps_collision_remains_detectable(fps: int) -> None:
    detector = CrashDetector(fps=fps)
    events = []
    impact_t = 42 / 30
    duration = 81 / 30

    for frame in range(1, int(duration * fps) + 1):
        t = frame / fps
        ax, ay = 200 + 120 * t, 300
        if t < impact_t:
            bx, by = 540 - 90 * t, 300
        else:
            bx = (540 - 90 * impact_t) + 30 * (t - impact_t)
            by = 300 + 150 * (t - impact_t)
        events.extend(detector.update(frame, [_det(frame, 1, ax, ay), _det(frame, 2, bx, by)]))

    assert len(events) == 1
    assert events[0].involved_tracks == [1, 2]
    assert abs(events[0].trigger_frame / fps - impact_t) <= 0.25


def test_resolution_change_resets_optical_flow_reference_without_crashing() -> None:
    detector = CrashDetector(fps=30)
    for frame_no, shape in [(1, (100, 100, 3)), (2, (120, 120, 3)), (3, (120, 120, 3))]:
        image = np.zeros(shape, dtype=np.uint8)
        events = detector.update(
            frame_no,
            [_det(frame_no, 1, 35, 50, w=30, h=30), _det(frame_no, 2, 70, 50, w=30, h=30)],
            frame=image,
        )
        assert events == []


@pytest.mark.parametrize("fps", [-30, 0, 4.99, float("nan"), float("inf")])
def test_invalid_fps_is_rejected(fps: float) -> None:
    with pytest.raises(ValueError):
        CrashDetector(fps=fps)


def test_non_monotonic_frame_numbers_are_rejected() -> None:
    detector = CrashDetector(fps=30)
    detector.update(1, [])
    with pytest.raises(ValueError):
        detector.update(1, [])
    with pytest.raises(ValueError):
        detector.update(0, [])


def test_malformed_detections_are_dropped_and_duplicate_ids_choose_best_quality() -> None:
    detector = CrashDetector(fps=30)
    malformed = [
        _det(1, 2, 20, 20, w=-10, h=30),
        Detection(1, 3, "car", 0.9, (float("nan"), 0, 10, 30)),
        Detection(1, 4, "car", float("nan"), (0, 0, 20, 30)),
    ]
    low = _det(1, 1, 30, 50, w=30, h=30, quality=0.40)
    high = _det(1, 1, 70, 50, w=30, h=30, quality=0.95)

    assert detector.update(1, [*malformed, low, high]) == []
    assert set(detector.fsm.tracks) == {1}
    assert detector.fsm.tracks[1].history[-1].cx == pytest.approx(70.0)


def test_crowded_same_direction_scene_does_not_accumulate_irrelevant_pairs() -> None:
    detector = CrashDetector(fps=30)
    for frame in range(1, 61):
        detections = []
        for i in range(80):
            row, col = divmod(i, 10)
            detections.append(
                _det(frame, i + 1, 100 + col * 120 + frame * 1.5, 100 + row * 90, w=40, h=30)
            )
        assert detector.update(frame, detections) == []

    assert len(detector.fsm.pairs) <= 80

from __future__ import annotations

from traffic_intel.domain import Detection
from traffic_intel.incident.crash_detector import CrashDetector


def _det(frame: int, tid: int, cx: float, cy: float, w: float = 60, h: float = 40) -> Detection:
    return Detection(
        frame=frame,
        track_id=tid,
        class_name="car",
        confidence=0.85,
        bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        track_quality=0.85,
        track_confirmed=True,
    )


def test_single_vehicle_hard_stop_never_creates_crash() -> None:
    detector = CrashDetector(fps=30)
    events = []
    x = 100.0
    for frame in range(1, 80):
        x += 5.0 if frame < 42 else 0.0
        events.extend(detector.update(frame, [_det(frame, 1, x, 300)], frame=None))
    assert events == []


def test_close_same_direction_hard_braking_is_not_a_crash() -> None:
    detector = CrashDetector(fps=30)
    events = []
    lead_x = 300.0
    rear_x = 220.0
    for frame in range(1, 90):
        lead_x += 4.0 if frame < 48 else 0.3
        rear_x += 4.8 if frame < 48 else 0.4
        events.extend(
            detector.update(
                frame,
                [_det(frame, 1, lead_x, 300), _det(frame, 2, rear_x, 300)],
                frame=None,
            )
        )
    assert events == []


def test_pair_collision_is_attributed_to_only_the_interacting_pair() -> None:
    detector = CrashDetector(fps=30)
    events = []
    for frame in range(1, 81):
        ax, ay = 200 + 4 * frame, 300
        bx, by = 540 - 3 * frame, 300
        if frame >= 43:
            bx = (540 - 3 * 42) + (frame - 42)
            by = 300 + 5 * (frame - 42)
        # Third vehicle is nearby in the image but uninvolved.
        cx, cy = 720 - 1.5 * frame, 390
        events.extend(
            detector.update(
                frame,
                [
                    _det(frame, 1, ax, ay),
                    _det(frame, 2, bx, by),
                    _det(frame, 3, cx, cy),
                ],
                frame=None,
            )
        )

    assert len(events) == 1
    assert events[0].involved_tracks == [1, 2]
    assert 42 <= events[0].trigger_frame <= 47
    assert 3 not in events[0].involved_tracks


def test_immature_track_turn_does_not_become_valid_retroactively() -> None:
    detector = CrashDetector(fps=30)
    events = []
    for frame in range(1, 75):
        detections = [_det(frame, 1, 200 + 3 * frame, 300)]
        if frame >= 50:
            # New track appears close to track 1, then turns sharply while its
            # detector/tracker geometry is still settling.
            age = frame - 50
            x = 365 + 2 * age
            y = 300 if age < 8 else 300 + 6 * (age - 7)
            detections.append(_det(frame, 2, x, y))
        events.extend(detector.update(frame, detections, frame=None))
    assert events == []

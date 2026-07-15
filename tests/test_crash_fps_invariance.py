from __future__ import annotations

import pytest

from traffic_intel.incident.crash_detector import CrashDetector
from traffic_intel.domain import Detection


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


@pytest.mark.parametrize("fps", [15, 30, 60])
def test_same_physical_collision_detects_at_similar_time_across_fps(fps: int) -> None:
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
        cx, cy = 720 - 45 * t, 390
        events.extend(
            detector.update(
                frame,
                [_det(frame, 1, ax, ay), _det(frame, 2, bx, by), _det(frame, 3, cx, cy)],
                frame=None,
            )
        )

    assert len(events) == 1
    assert events[0].involved_tracks == [1, 2]
    assert abs(events[0].trigger_frame / fps - 1.5) <= 0.08

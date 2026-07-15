from __future__ import annotations

from pathlib import Path

import numpy as np

from traffic_intel.core.engine import TrafficEngine
from traffic_intel.recording.event_recorder import RollingSegmentBuffer, _Segment


def test_scene_cut_ignores_local_motion_but_resets_on_global_change() -> None:
    engine = TrafficEngine.__new__(TrafficEngine)
    engine._prev_scene_gray = None

    base = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert engine._detect_scene_cut(base) is False

    local = base.copy()
    local[250:350, 400:550] = 255
    assert engine._detect_scene_cut(local) is False

    global_change = np.full_like(base, 220)
    assert engine._detect_scene_cut(global_change) is True


def test_event_recorder_keeps_impact_frame_and_ignores_duplicate_trigger(tmp_path: Path) -> None:
    recorder = RollingSegmentBuffer(fps=30, output_dir=str(tmp_path / "events"))
    fake = tmp_path / "seg.mp4"
    fake.write_bytes(b"not-a-real-video")
    recorder._segments.append(_Segment(video_path=fake, frame_count=1))

    class Engine:
        frame_count = 140

    recorder.trigger_auto(
        "collision",
        Engine(),
        trigger_frame=123,
        event_metadata={"involved_tracks": [7, 9], "score": 0.91},
    )
    assert recorder._post_trigger is not None
    assert recorder._post_trigger["trigger_frame"] == 123
    assert recorder._post_trigger["detected_frame"] == 140

    # A second candidate must not overwrite the active event package.
    recorder.trigger_auto("collision", Engine(), trigger_frame=130)
    assert recorder._post_trigger["trigger_frame"] == 123

    recorder._post_trigger = None
    recorder.close()


def test_event_recorder_uses_unique_segments_and_reports_configured_windows(tmp_path: Path) -> None:
    import json

    recorder = RollingSegmentBuffer(
        fps=2,
        output_dir=str(tmp_path / "events"),
        segment_duration=0.5,
        max_seconds=1.0,
        post_event_seconds=0.5,
    )
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    class Engine:
        frame_count = 3

    for frame_no in range(1, 4):
        recorder.write_frame(frame, [], frame_no)
    pre_paths = [segment.video_path.name for segment in recorder._segments]
    recorder.trigger_auto("collision", Engine(), trigger_frame=2)
    assert recorder._post_trigger is not None

    recorder.write_frame(frame, [], 4)
    assert recorder.last_saved_event is not None
    metadata = json.loads((recorder.last_saved_event / "event.json").read_text())
    assert metadata["pre_event_seconds"] == 1.0
    assert metadata["post_event_seconds"] == 0.5
    assert metadata["trigger_frame"] == 2
    assert len(pre_paths) == len(set(pre_paths))
    assert (recorder.last_saved_event / "clip.mp4").exists()
    recorder.close()

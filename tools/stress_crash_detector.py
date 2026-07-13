"""Deterministic adversarial stress suite for the crash detector.

This is intentionally self-contained: it does not require YOLO weights or the
private validation videos.  It targets detector semantics and runtime hardening,
not end-to-end model accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from traffic_intel.domain import Detection
from traffic_intel.incident.crash_detector import CrashDetector


def det(frame: int, tid: int, cx: float, cy: float, *, jx: float = 0.0, jy: float = 0.0,
        w: float = 60.0, h: float = 40.0) -> Detection:
    return Detection(
        frame=frame,
        track_id=tid,
        class_name="car",
        confidence=0.85,
        bbox=(cx - w / 2 + jx, cy - h / 2 + jy, cx + w / 2 + jx, cy + h / 2 + jy),
        track_quality=0.85,
        track_confirmed=True,
    )


def hard_braking_trials(jitter: float, trials: int = 40) -> int:
    false_alerts = 0
    for seed in range(trials):
        rng = random.Random(seed)
        detector = CrashDetector(fps=30)
        lead_x, rear_x = 300.0, 225.0
        events = []
        for frame in range(1, 91):
            lead_x += 4.0 if frame < 48 else 0.3
            rear_x += 4.8 if frame < 48 else 0.4
            events.extend(detector.update(frame, [
                det(frame, 1, lead_x, 300, jx=rng.uniform(-jitter, jitter), jy=rng.uniform(-jitter, jitter)),
                det(frame, 2, rear_x, 300, jx=rng.uniform(-jitter, jitter), jy=rng.uniform(-jitter, jitter)),
            ]))
        false_alerts += int(bool(events))
    return false_alerts


def crossing_near_miss_trials(trials: int = 60) -> int:
    false_alerts = 0
    for seed in range(trials):
        rng = random.Random(seed)
        detector = CrashDetector(fps=30)
        offset = rng.choice([-24, -20, -18, 18, 20, 24])
        events = []
        for frame in range(1, 100):
            events.extend(detector.update(frame, [
                det(frame, 1, 100 + 7 * frame, 300,
                    jx=rng.uniform(-1, 1), jy=rng.uniform(-1, 1)),
                det(frame, 2, 415, 50 + 7 * (frame - offset),
                    jx=rng.uniform(-1, 1), jy=rng.uniform(-1, 1)),
            ]))
        false_alerts += int(bool(events))
    return false_alerts


def collision_trial(seed: int = 0, *, fps: int = 30, jitter: float = 0.0,
                    dropout: float = 0.0) -> list:
    rng = random.Random(seed)
    detector = CrashDetector(fps=fps)
    impact_t = 42 / 30
    duration = 81 / 30
    events = []
    for frame in range(1, int(duration * fps) + 1):
        t = frame / fps
        ax, ay = 200 + 120 * t, 300
        if t < impact_t:
            bx, by = 540 - 90 * t, 300
        else:
            bx = (540 - 90 * impact_t) + 30 * (t - impact_t)
            by = 300 + 150 * (t - impact_t)
        detections = []
        if rng.random() >= dropout:
            detections.append(det(frame, 1, ax, ay,
                                  jx=rng.uniform(-jitter, jitter), jy=rng.uniform(-jitter, jitter)))
        if rng.random() >= dropout:
            detections.append(det(frame, 2, bx, by,
                                  jx=rng.uniform(-jitter, jitter), jy=rng.uniform(-jitter, jitter)))
        events.extend(detector.update(frame, detections))
    return events


def crowded_soak(frames: int = 600, vehicles: int = 16) -> tuple[int, int]:
    detector = CrashDetector(fps=30)
    alerts = 0
    for frame in range(1, frames + 1):
        detections = []
        for i in range(vehicles):
            row, col = divmod(i, 8)
            detections.append(det(
                frame, i + 1,
                80 + col * 115 + frame * (1.0 + row * 0.05),
                180 + row * 100,
                w=44, h=32,
            ))
        alerts += len(detector.update(frame, detections))
    return alerts, len(detector.fsm.pairs)


def throughput_benchmark(counts: tuple[int, ...] = (5, 10, 20, 40, 80), frames: int = 120) -> dict[str, float]:
    result: dict[str, float] = {}
    for n in counts:
        detector = CrashDetector(fps=30)
        start = time.perf_counter()
        for frame in range(1, frames + 1):
            detections = []
            for i in range(n):
                row, col = divmod(i, 10)
                detections.append(det(
                    frame, i + 1,
                    100 + col * 120 + frame * 1.5,
                    100 + row * 90,
                    w=40, h=30,
                ))
            detector.update(frame, detections)
        elapsed = time.perf_counter() - start
        result[str(n)] = round(frames / elapsed, 2)
    return result


def run() -> tuple[dict, list[str]]:
    failures: list[str] = []
    hard = {str(j): hard_braking_trials(j) for j in (0.5, 1.0, 1.5)}
    for jitter, alerts in hard.items():
        if alerts:
            failures.append(f"hard braking jitter={jitter}: {alerts}/40 false alerts")

    crossing = crossing_near_miss_trials()
    if crossing:
        failures.append(f"crossing near miss: {crossing}/60 false alerts")

    low_fps = {}
    for fps in (5, 6, 7, 8, 9, 10, 15, 30, 60, 120):
        events = collision_trial(fps=fps)
        low_fps[str(fps)] = len(events)
        if len(events) != 1:
            failures.append(f"collision at {fps} FPS: expected 1 event, got {len(events)}")

    dropout_hits = 0
    for seed in range(40):
        dropout_hits += int(bool(collision_trial(seed, jitter=1.0, dropout=0.30)))
    if dropout_hits != 40:
        failures.append(f"30% dropout collision recovery: {dropout_hits}/40 detected")

    soak_alerts, soak_pairs = crowded_soak()
    if soak_alerts:
        failures.append(f"crowded same-direction soak: {soak_alerts} false alerts")

    result = {
        "hard_braking_false_alerts_40_trials": hard,
        "crossing_near_miss_false_alerts_60_trials": crossing,
        "collision_event_count_by_fps": low_fps,
        "collision_detection_with_30pct_dropout_40_trials": dropout_hits,
        "crowded_16_vehicle_600_frame_soak": {
            "false_alerts": soak_alerts,
            "remaining_pair_states": soak_pairs,
        },
        "crash_analysis_throughput_fps_by_track_count": throughput_benchmark(),
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    return result, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON result path")
    args = parser.parse_args()

    result, failures = run()
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

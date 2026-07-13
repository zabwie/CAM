# Validation fixtures

`cached/` contains YOLO detector output cached at 640 px.

The full cached tracker/identity/crash replay also requires the original source pixels in:

- `validation/videos/crash.mp4`
- `validation/videos/crash2.mp4`

Those large clips are intentionally omitted from the slim package. When they are present, run:

```bash
python tools/replay_cached_crash_regression.py
```

For a self-contained regression that does not require model weights or source videos, run:

```bash
python tools/stress_crash_detector.py --output validation/latest_stress_results.json
```

The production default remains 1280 px. Cached replay and synthetic stress tests are downstream algorithm tests, not substitutes for end-to-end detector validation on representative deployment footage.

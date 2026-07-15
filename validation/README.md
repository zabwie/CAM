# Validation fixtures

Source video clips are in the repository `videos/` directory.

`cached/` contains YOLO detector output cached at 640px. These files make tracker and crash-logic iteration fast and deterministic without requiring model inference on every test run.

Run:

```bash
python tools/replay_cached_crash_regression.py
```

The production default remains 1280px. Cached regression is a downstream algorithm test, not an end-to-end detector benchmark.

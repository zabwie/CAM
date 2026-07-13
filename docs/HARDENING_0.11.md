# Crash-detector hardening — v0.11.0

This release addresses adversarial failures found during pre-sale stress testing.
It does not claim zero defects or replace validation on representative customer cameras.

## Fixed

- **Common-mode hard-braking false positives:** crash confirmation now requires pair-level relative-motion change. Long-standing 2-D overlap plus synchronized braking is suppressed instead of being treated as impact evidence.
- **Low-frame-rate blind spot:** time windows now adapt to sparse sampling; the synthetic physical collision regression detects from **5 through 120 FPS**.
- **Crowded-scene scaling:** recent velocities are cached once per track, a cheap broad phase rejects impossible interactions, and low-risk distant/parallel pairs are not persisted in the FSM.
- **Resolution changes:** optical-flow state resets safely when frame dimensions change.
- **Malformed input:** non-finite/reversed boxes and invalid confidence/quality rows are dropped; duplicate track IDs choose the strongest observation deterministically.
- **Temporal integrity:** non-monotonic frame numbers and unsupported FPS values are rejected explicitly.
- **Package import isolation:** importing lightweight modules no longer eagerly imports the full YOLO/ByteTrack runtime.
- **Upload path hardening:** dashboard uploads are written under generated temporary names rather than client-provided paths.
- **Slim-package diagnostics:** cached video replay now explains when source clips are missing instead of failing deep inside OpenCV.

## Self-contained stress result

Run:

```bash
python tools/stress_crash_detector.py --output validation/latest_stress_results.json
```

Latest result in this package:

- 0/40 false alerts at 0.5 px hard-braking jitter.
- 0/40 false alerts at 1.0 px hard-braking jitter.
- 0/40 false alerts at 1.5 px hard-braking jitter.
- 0/60 crossing near-miss false alerts.
- 1/1 collision event at each tested source rate: 5, 6, 7, 8, 9, 10, 15, 30, 60, and 120 FPS.
- 40/40 collision trials detected with 30% random observation dropout plus 1 px localization jitter.
- 0 alerts in a 600-frame, 16-vehicle same-direction soak.
- Crash-analysis-only throughput on the audit machine: 1617, 795, 181, 68, and 29 FPS at 5, 10, 20, 40, and 80 simultaneous tracks respectively. Throughput is hardware-dependent.

## Important boundary

These are deterministic synthetic/adversarial regressions. They prove that the specific failures above are covered; they do **not** establish a real-world false-positive or false-negative rate. A larger labeled video corpus remains required before making reliability guarantees.

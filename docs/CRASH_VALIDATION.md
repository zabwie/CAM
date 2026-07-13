# Crash detector regression notes

The crash detector was iterated against the two supplied source clips and prior annotated outputs that exposed late triggers, wrong participant attribution, phantom tracks, and visual-state leakage.

## Main failure modes corrected

1. Duplicate cross-class detections could become separate tracks for one physical vehicle.
2. Tentative/new tracks were trusted before their geometry stabilized.
3. A single unrelated track discontinuity could emit a crash.
4. Pair state and impact evidence were not the true event-emission path.
5. Image-space overlap could confuse vehicles at different apparent road depths.
6. Post-impact stopping caused late timestamps.
7. Persistent crash graphics and temporal state could survive a source transition.
8. Crash kinematic thresholds were implicitly frame-rate dependent.
9. Event-buffer segment names could be reused after a trigger and overwrite captured pre-event footage.

## Selected behavior

- class-agnostic vehicle NMS;
- explicit track trust gate;
- pair-only crash emission;
- trajectory interaction and apparent-depth consistency;
- impact-time motion discontinuity;
- synchronized pair evidence or strong contact-coupled directional impulse;
- post-impact behavior as supporting evidence only;
- candidate-only optical flow;
- source-change reset;
- 30-FPS-equivalent kinematic normalization across source frame rates.

## Current deterministic replay

```bash
python tools/replay_cached_crash_regression.py
```

| Source | Impact | Confirmed | Score* | Extra events |
|---|---:|---:|---:|---:|
| `crash.mp4` | 123 | 129 | 0.817 | 0 |
| `crash2.mp4` | 238 | 238 | 0.658 | 0 |

\* Scores shown here are from the cached 640px detector replay with optical flow disabled. Earlier full-video review with candidate-only optical-flow support strengthened the second event; optical flow is supporting evidence and is not allowed to create a crash independently.

These are regression results for the supplied clips, not a general accuracy percentage.

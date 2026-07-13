# Validation status

## Automated suite

```bash
pytest -q
```

Current result: **19 passed**.

Coverage includes:

- isolated hard stop does not become a crash;
- coordinated braking does not become a crash;
- crash is attributed to the interacting pair only;
- nearby uninvolved third vehicle is excluded;
- immature track discontinuity cannot become valid retroactively;
- global scene change resets temporal assumptions while local motion does not;
- event recorder preserves the impact frame and rejects duplicate active triggers;
- event recorder uses unique segment paths and reports configured pre/post windows;
- track-quality gate rejects a large ID jump and requires reacquisition;
- constant world velocity produces a stable speed estimate;
- a single extreme world-position jump is rejected;
- calibration projection and calibrated-zone bounds;
- the same synthetic physical collision is detected at similar time at 15, 30, and 60 FPS;
- a raw tracker ID change (for example 40 → 63) can preserve the same canonical physical-vehicle ID;
- ambiguous re-entry is not force-stitched;
- a raw-ID hijack cannot inherit another vehicle's analytics history;
- a strong canonical stitch can preserve a mature trusted track;
- a near-static foreground false track cannot absorb a relocated moving vehicle without strong supporting evidence.

## Supplied crash-video regression

Two source videos are included under `validation/videos/`.

For fast repeatable downstream regression, cached YOLO outputs are stored under `validation/cached/`. Run:

```bash
python tools/replay_cached_crash_regression.py
```

Current result:

| Source | Impact frame | Confirmation frame | Events | Extra events |
|---|---:|---:|---:|---:|
| `crash.mp4` | 123 | 127 | 1 | 0 |
| `crash2.mp4` | 238 | 238 | 1 | 0 |

The first result is attributed to canonical vehicle IDs 11 and 19; the second to canonical IDs 13 and 14. Raw ByteTrack IDs are run-local association handles and may change underneath a stable canonical identity.

## Full-model validation

Use:

```bash
python -m traffic_intel.validate_crashes validation/videos/crash.mp4 \
  --model yolo11n.pt --imgsz 1280 \
  --output crash_validated.mp4 \
  --events-json crash_events.json
```

The full production-resolution path should be run on the target acceleration hardware. The provided cached replay is specifically for deterministic tracker/crash-logic iteration and does not claim end-to-end detector accuracy.

## Current limitation

Two positive crash clips are not enough to estimate general crash precision or recall. A serious pilot should add labeled hard negatives and varied positives, then report:

- event precision and recall;
- false alarms per camera-hour;
- impact timestamp error;
- participant precision/recall;
- performance by weather, lighting, camera geometry, occlusion, and crash type.

## Canonical identity continuity regression

The final annotated full-clip replay produced the following on both supplied clips:

- adjacent high-overlap canonical ID switches: **0**;
- remaining canonical fragmentation candidates: **0**;
- duplicate canonical assignment frames: **0**.

Raw ByteTrack fragmentation still occurs underneath the canonical layer. That is expected and is visible in the validation overlay as a smaller `raw` ID changing while the larger canonical `ID` remains stable. Metrics are regression heuristics on the supplied clips, not a general IDF1/HOTA benchmark.

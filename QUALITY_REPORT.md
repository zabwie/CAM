# Engineering quality report — v0.10.0

## Executive assessment

The project is now organized as a sellable engineering demo/pilot codebase rather than a single-script proof of concept.

The strongest parts are the perception pipeline, canonical identity layer, track trust gate, robust speed estimator, pair-attributed crash detector, deterministic regression workflow, and incident recorder. The dashboard remains intentionally basic, and legal/forensic proof controls are deferred to a later layer as requested.

## Before this cleanup

- `engine.py`: 729 lines and owned calibration, scene-change detection, track trust, model orchestration, speed integration, drawing, and export.
- `crash_detector.py`: incident semantics and persistent UI state in the same module.
- Track trust used hit count and confidence but had no explicit reacquisition cooldown or size-change instability penalty.
- Crash kinematics were implicitly tied to source FPS.
- Event recorder could reuse segment filenames after a trigger and overwrite captured pre-event footage.
- Recorder metadata could report default pre/post durations instead of configured durations.
- Live speed alerts inspected only the last current vehicle.
- Calibration tool hard-coded a macOS Matplotlib backend.
- Plain `pytest -q` was not reliable in the repository environment.
- 6 regression tests.
- Documentation contained stale constructor arguments and capabilities that no longer matched the code.

## After this cleanup

### Architecture

- `calibration.py` — projection, smoothing, calibration quality.
- `config.py` — typed runtime settings.
- `domain.py` — trusted observation record.
- `identity.py` — canonical physical-vehicle identity, conservative track stitching, and raw-ID hijack protection.
- `tracking.py` — track-quality and reacquisition gate.
- `scene.py` — source-discontinuity detection.
- `engine.py` — perception orchestrator only.
- `pipeline.py` — canonical perception + incident path.
- `crash_visuals.py` — UI persistence separated from incident semantics.
- `event_recorder.py` — bounded incident capture with corrected segment lifecycle.

`engine.py` dropped from 729 lines to approximately 511 lines while adding clearer module boundaries.

### Tracking and identity

ByteTrack IDs are now treated as ephemeral association handles rather than durable vehicle identity. `CanonicalIdentityManager` sits before analytics and provides:

- stable per-camera canonical IDs;
- conservative short-gap stitching after tracker fragmentation;
- motion-predicted re-entry matching;
- scale, class, and appearance consistency checks;
- raw-ID hijack detection;
- provisional new-ID suppression so a one-frame temporary identity is never exposed to analytics;
- filtered display geometry while retaining raw geometry for impact evidence.

Track trust then considers:

- hit maturity;
- confidence EMA;
- class consistency;
- normalized center jump;
- abrupt bounding-box area change;
- explicit reacquisition cooldown;
- decaying instability score.

ByteTrack remains responsible for association; the quality gate decides whether downstream analytics may trust the track.

### Speed estimation

- Bottom-center calibration smoothing now uses both image coordinates, not only vertical position.
- Existing robust trajectory fit, outlier rejection, gap reset, speed caps, and acceleration limiting are preserved.
- Trajectory confidence now incorporates robust fit quality and residual behavior.

### Crash detection

- Existing pair-only attribution and supplied-video timing are preserved.
- Temporal windows are expressed in seconds.
- Kinematics are normalized to a 30-FPS-equivalent motion rate.
- Synthetic regression verifies comparable physical collision timing at 15, 30, and 60 FPS.

### Recording

- Monotonic segment IDs eliminate pre-event overwrite risk.
- Configured pre/post windows are written accurately to event metadata.
- Event directories include a unique event-ID prefix.
- Duplicate triggers cannot replace an active event.

## Verification performed

- `pytest -q` → **19 passed**.
- Deterministic cached crash replay:
  - `crash.mp4` → impact 123, detected 127, one event, no extra events.
  - `crash2.mp4` → impact 238, detected 238, one event, no extra events.
- Editable package installation succeeded.
- Runtime doctor passed for Python, OpenCV, NumPy, and bundled model presence.
- CLI help paths verified for live, calibration, and crash validation.
- Real YOLO + ByteTrack smoke test on supplied footage completed without pipeline errors; trusted detections began emitting after the expected warm-up window.

### Identity continuity regression

Full supplied-clip replay using cached detections plus source pixels for appearance/scene handling produced:

- 0 adjacent high-overlap canonical ID switches;
- 0 remaining canonical fragmentation candidates under the regression heuristic;
- 0 frames where one canonical ID was assigned to two simultaneous tracks;
- repaired raw ByteTrack fragmentation while keeping crash participants distinct through impact.

The annotated validation videos display the large canonical ID and smaller raw ByteTrack ID so repairs are directly inspectable.

## Current engineering score

| Area | Assessment |
|---|---:|
| Perception/tracking architecture | 9.0 / 10 |
| Speed-estimation foundation | 8.5 / 10 |
| Crash-detection architecture | 8.5 / 10 |
| Code organization | 8.0 / 10 |
| Regression/testability | 8.5 / 10 |
| Runtime packaging | 8.0 / 10 |
| Dashboard/product UI | 4.0 / 10 |
| Forensic proof / custody layer | intentionally deferred |

## Remaining technical priorities

1. Build a larger labeled positive and hard-negative video corpus.
2. Add standalone incident analyzers behind the same trusted detection stream.
3. Expand canonical ReID from conservative short gaps to longer occlusions only after a larger labeled identity corpus exists.
4. Validate calibrated speed against independent ground truth.
5. Build the operator workflow and persistent incident store after the event schema stabilizes.
6. Add the signed evidence/chain-of-custody layer last, around stable event outputs.

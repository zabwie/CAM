# Engineering quality report — v0.12.0

## Current verification

- `pytest -q`: **45 passed**.
- Streamlit application test runner: **0 exceptions** on both empty and populated operational data.
- End-to-end review interaction test: an actual dashboard **Approve** action persisted the incident state change successfully.
- `python tools/stress_crash_detector.py`: **PASS**.
- Synthetic hard-braking false alerts: **0/40** at each tested jitter level (0.5, 1.0, and 1.5 px).
- Crossing near-miss false alerts: **0/60**.
- Synthetic collision regression: one event at every tested source rate from **5 through 120 FPS**.
- Collision detection with 30% random observation dropout: **40/40**.
- 600-frame, 16-vehicle same-direction soak: **0 false alerts**.
- Final wheel target: `traffic_intel-0.12.0-py3-none-any.whl`.

## Product hardening added in v0.12.0

The codebase now includes an operations layer around the perception engine:

- durable SQLite incident/review/camera/speed/notification storage;
- searchable human review workflow;
- approve, dismiss, and needs-information decisions;
- reviewer identity, notes, corrected classification, and review history;
- gated human-feedback score calibration;
- evidence ZIP export;
- speed/day/hour/hotspot analytics;
- camera geolocation and agency-timezone support;
- in-app notification records and optional webhook delivery;
- six-workspace operations dashboard.

The human-feedback model is deliberately conservative. It does not silently retrain the live crash detector from individual clicks; it learns a review-priority calibration only after minimum data and class-balance requirements are met.

## Remaining boundary

This release is materially stronger as a paid-pilot product, but it is still not a claim of zero defects, emergency-grade certification, or a measured real-world false-positive/false-negative rate. The highest-value remaining work is a larger labeled deployment corpus, authenticated multi-user access, role-based permissions, centralized service health, managed deployment/backups, and independent field validation against real agency workflows.

---

# Engineering quality report — v0.11.0

## Current verification

- `pytest -q`: **38 passed**.
- Coverage: **54% overall**, with crash detector **90%** and crash FSM **92%**. Overall coverage remains diluted by CLI/dashboard/full-model paths that are not exercised in the unit suite.
- `python tools/stress_crash_detector.py`: **PASS**.
- Wheel build: `traffic_intel-0.11.0-py3-none-any.whl` built successfully.
- Runtime doctor: Python/OpenCV/NumPy pass; an uncached bare YOLO model name is reported as a warning rather than a false local-file failure.

## Hardening result

The reproduced hard-braking false-positive pathway is now covered by pair-relative motion validation, stale-contact gating, and common-mode braking suppression. The deterministic stress suite records zero false alerts in 40 trials each at 0.5, 1.0, and 1.5 px localization jitter, zero in 60 crossing near misses, full detection across 5–120 FPS, 40/40 detection with 30% random observation dropout, and zero alerts in a 600-frame 16-vehicle soak. See `HARDENING_0.11.md`.

## Remaining boundary

This release is materially safer and more testable, but synthetic stress tests do not establish a real-world error rate. The most important remaining work is a larger labeled deployment corpus, end-to-end engine integration coverage, operational authentication/authorization for remote deployments, and independent licensing/deployment review for the chosen model stack.

---

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

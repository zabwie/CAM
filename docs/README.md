# Traffic Intelligence

Camera-based vehicle perception, calibrated speed estimation, pair-attributed crash detection, and rolling incident capture.

This repository is a working engineering package, not a dashboard-first demo. The core runtime is split into independent perception, motion, incident, and recording layers so new forensic incident analyzers can be added without turning the tracker into a monolith.

## Current pipeline

```text
Camera / video
    ↓
YOLO11 @ configurable inference size
    ↓
class-agnostic vehicle NMS
    ↓
ByteTrack (raw association IDs)
    ↓
CanonicalIdentityManager
  • stable per-camera physical-vehicle IDs
  • conservative short-gap track stitching
  • appearance / motion / scale consistency
  • raw-ID hijack protection
  • provisional-ID suppression
    ↓
TrackQualityGate
  • maturity
  • confidence EMA
  • class consistency
  • geometry-instability rejection
  • reacquisition cooldown
    ↓
Trusted Detection stream
    ├──────────────→ RobustSpeedEstimator + Calibration
    │
    └──────────────→ CrashDetector
                       • pair geometry / TTC
                       • apparent-depth consistency
                       • impact-time motion discontinuity
                       • synchronized pair evidence
                       • candidate-only optical flow
                       • post-impact support
                             ↓
                    TrafficIncidentPipeline
                             ↓
                    RollingSegmentBuffer
```

## What works now

- Vehicle detection for COCO vehicle classes: car, motorcycle, bus, truck.
- Class-agnostic NMS before tracking to prevent duplicate `car`/`truck` tracks for one physical vehicle.
- ByteTrack association plus a canonical identity layer that repairs conservative short-gap fragmentation before analytics see the track.
- Raw tracker IDs remain diagnostic only; speed/crash history is keyed by canonical vehicle identity.
- Scene/source-change reset so tracking, identity, speed, incident state, and crash overlays do not leak between sources.
- Homography-based road-plane projection with calibration-fit diagnostics.
- Robust trajectory speed estimation with outlier rejection, track-gap handling, acceleration limiting, and fit-quality confidence.
- Pair-attributed crash detection. A standalone hard stop, noisy track, or unrelated nearby vehicle cannot emit a crash by itself.
- FPS-normalized crash kinematics validated from 5 through 120 FPS in synthetic regression tests.
- Rolling pre/post incident recorder with synchronized telemetry and accurate configured event-window metadata.
- Headless crash validation and deterministic cached-detector regression replay.
- Live-first Streamlit operations panel in Spanish, with active vehicles, speed, crash alerts, incident capture, and evidence review.

## Quick start

### 1. Install

```bash
python -m pip install -e .
```

For dashboard dependencies:

```bash
python -m pip install -e '.[dashboard]'
```

### 2. Check the environment

```bash
traffic-intel-doctor --model yolo11n.pt
```

### 3. Run the test suite

```bash
pytest -q
```

Current repository suite: **38 passing tests**.

### 4. Run deterministic crash regression

When the original source videos are present, cached YOLO outputs can be replayed through tracker/identity/crash logic:

```bash
python tools/replay_cached_crash_regression.py
```

Expected current result:

```text
crash:  impact=123  detected=127  one event
crash2: impact=238  detected=238  one event
```

The slim package intentionally omits source videos. In that package, run `python tools/stress_crash_detector.py` for the self-contained adversarial suite. Cached replay requires restoring `validation/videos/crash.mp4` and `crash2.mp4`. It does **not** replace full production-resolution inference validation.


### 4a. Run the self-contained adversarial stress suite

```bash
python tools/stress_crash_detector.py --output validation/latest_stress_results.json
```

This covers hard-braking false positives, crossing near misses, 5–120 FPS collision invariance, 30% observation dropout, crowded-scene soaking, and crash-analysis throughput. See `docs/HARDENING_0.11.md`.

### 5. Validate a video through the full model pipeline

```bash
python -m traffic_intel.validate_crashes validation/videos/crash.mp4 \
  --model yolo11n.pt \
  --imgsz 1280 \
  --output crash_validated.mp4 \
  --events-json crash_events.json
```

On Apple Silicon, place an exported `yolo11n.mlpackage` beside the `.pt` file and the engine will prefer it automatically when `yolo11n.pt` is requested.

### 6. Run live

```bash
python -m traffic_intel.live \
  --camera 0 \
  --model yolo11n.pt \
  --imgsz 1280 \
  --event-dir events
```

With calibration and a speed threshold:

```bash
python -m traffic_intel.live \
  --camera rtsp://camera/stream \
  --calibration calib.json \
  --speed-limit 50 \
  --pre-event-seconds 20 \
  --post-event-seconds 10
```

Controls: `q` quits, `m` manually captures an incident package.

## Python API

```python
from traffic_intel import Calibration, TrafficEngine, TrafficIncidentPipeline

calibration = Calibration.load("calib.json")
engine = TrafficEngine(
    model_path="yolo11n.pt",
    calibration=calibration,
    imgsz=1280,
    retain_history=False,
)
pipeline = TrafficIncidentPipeline(engine)

# result = pipeline.process_frame(frame)
# result.detections -> trusted vehicle observations
# result.crashes    -> pair-attributed crash events
# result.annotated  -> display/recording frame
```

## Calibration

```bash
python -m traffic_intel.calibrate --image ref_frame.jpg --output calib.json
```

The current calibration model supports:

- a road-plane homography from four or more image/world correspondences;
- an optional road ROI polygon;
- calibration-fit diagnostics based on reprojection residuals.

Speed is intentionally unavailable when a vehicle lies outside the calibrated road-plane region.

## Project layout

```text
traffic_intel/
├── calibration.py      # road-plane projection and calibration quality
├── config.py           # typed runtime configuration
├── domain.py           # shared trusted observation records
├── identity.py         # canonical physical-vehicle identity / track stitching
├── tracking.py         # downstream track trust / reacquisition gate
├── scene.py            # source-discontinuity detection
├── engine.py           # YOLO + NMS + ByteTrack + canonical identity + speed
├── speed.py            # robust world-space trajectory speed
├── crash_detector.py   # pair interaction and impact-time state machine
├── crash_visuals.py    # crash annotation persistence only
├── pipeline.py         # canonical perception + incident coordinator
├── event_recorder.py   # rolling annotated-video and telemetry capture
├── live.py             # live camera adapter
├── validate_crashes.py # full-video headless validation
├── replay.py           # replay exported tracks through speed estimation
├── calibrate.py        # interactive calibration tool
├── doctor.py           # environment readiness checks
└── app.py              # live Spanish operations dashboard

tests/                  # unit and behavioral regression tests
validation/
├── videos/             # supplied crash source clips
├── cached/             # cached detector outputs for deterministic replay
├── expected.json       # accepted event timing windows
└── latest_cached_results.json
```

See `ARCHITECTURE.md` for design boundaries and `VALIDATION.md` for the current regression scope.

## Important scope

This codebase is ready to demonstrate and pilot as a traffic-perception and incident-detection system. It is **not yet a legally certified speed-measurement device or a complete chain-of-custody evidence platform**. Those require independent calibration validation, deployment controls, immutable manifests/signing, access logging, and jurisdiction-specific review.

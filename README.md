# Traffic Intelligence

Traffic Intelligence turns existing roadway camera feeds into a reviewable operational workflow: trusted vehicle tracking, calibrated speed observations, probable incident detection, automatic evidence capture, human review, notifications, searchable history, and roadway analytics.

The product is intentionally **human-in-the-loop**. The detector surfaces and preserves probable incidents; agency staff approve, dismiss, or reclassify them. Those decisions become supervised feedback for future triage calibration without silently retraining or suppressing the live detector from a single click.

## Product workflow

```text
Camera / video
    ↓
YOLO11 + class-agnostic NMS
    ↓
ByteTrack + CanonicalIdentityManager
    ↓
Trusted vehicle observations
    ├──────────────→ RobustSpeedEstimator + road calibration
    │                    ↓
    │              speed observations
    │                    ↓
    │              day / hour / hotspot analytics
    │
    └──────────────→ Pair-attributed CrashDetector
                         ↓
                   probable incident
                         ↓
                  RollingSegmentBuffer
                         ↓
               evidence package + metadata
                         ↓
                Operations database
                  ├─ review queue
                  ├─ approve / dismiss / needs info
                  ├─ search and filters
                  ├─ notifications
                  ├─ evidence export
                  └─ human-feedback calibration
```

## Current capabilities

### Perception and incident detection

- Vehicle detection for car, motorcycle, bus, and truck classes.
- Class-agnostic NMS before tracking to reduce duplicate vehicle tracks.
- ByteTrack association plus canonical per-camera vehicle identity continuity.
- Track maturity, confidence, class consistency, instability, and reacquisition quality gates.
- Scene/source-change reset so state does not leak across source discontinuities.
- Homography road-plane calibration and robust trajectory speed estimation.
- Pair-attributed crash detection with contact geometry, interaction risk, impact-time motion discontinuity, common-braking suppression, and occlusion/merge handling.
- FPS-normalized crash behavior covered by synthetic regression from 5 through 120 FPS.
- Rolling pre/post incident video capture with telemetry and SHA-256 clip digest.

### Agency workflow

- Durable SQLite operations database for a single-node pilot.
- Camera registry with municipality, location, optional latitude/longitude, and posted speed limit.
- Searchable incident review queue.
- Filters by status, time window, location, camera, and free text.
- Incident video playback and detector evidence inspection.
- Approve, dismiss, or request more information.
- Reviewer identity, notes, corrected classification, and review history.
- Portable evidence ZIP export with manifest, review history, clip, telemetry, and checksums when available.
- In-app notification outbox and optional webhook delivery.
- Import of existing recorder event packages into the operations database.

### Analytics

- Incident totals and review status.
- Average and maximum observed speed.
- Speeding rate relative to the configured posted speed limit.
- Most dangerous days by speeding behavior.
- Hour-of-day speeding patterns.
- Per-location and per-camera hotspot ranking.
- Optional hotspot map when camera coordinates are registered.

The hotspot `risk_index` is an operational ranking for prioritization. It is **not** a calibrated crash-probability claim.

### Human-feedback learning

Reviewed incidents create a supervised feedback dataset:

- approved incident → positive label;
- dismissed incident → negative label;
- needs-information → preserved but not used as a binary training label.

A regularized feedback calibration model activates only after minimum dataset-size and class-balance requirements are met. Its score is blended with the base detector score and used to prioritize the review queue. Raw incident evidence is always preserved.

This is safer than unconstrained online retraining and prevents one mistaken operator decision from immediately changing production crash detection behavior.

## Quick start

### 1. Install

Core runtime:

```bash
python -m pip install -e .
```

Dashboard:

```bash
python -m pip install -e '.[dashboard]'
```

### 2. Check the environment

```bash
traffic-intel-doctor --model yolo11n.pt
```

### 3. Run tests

```bash
pytest -q
```

Current repository suite: **45 passing tests**.

### 4. Run the adversarial crash stress suite

```bash
python tools/stress_crash_detector.py --output validation/latest_stress_results.json
```

The current self-contained suite covers:

- hard-braking false positives;
- crossing near misses;
- collision invariance from 5–120 FPS;
- 30% observation dropout;
- crowded-scene soaking;
- crash-analysis throughput.

### 5. Run the operations dashboard

```bash
traffic-intel-dashboard \
  --camera 0 \
  --calibration calib.json \
  --event-dir events \
  --archive-dir archive \
  --db-path data/traffic_intel.db \
  --speed-limit-mph 35
```

Equivalent direct Streamlit command:

```bash
streamlit run traffic_intel/app.py -- \
  --camera 0 \
  --calibration calib.json
```

The dashboard contains:

1. **Overview** — pending reviews, confirmed incidents, speeding metrics, and hotspots.
2. **Review queue** — search, filter, inspect, classify, and export incidents.
3. **Analytics** — dangerous days, dangerous hours, speed trends, and hotspot rankings/maps.
4. **Live monitor** — current camera feed, trusted vehicles, calibration state, and speed state.
5. **Evidence library** — incident clips and continuous recordings.
6. **Settings** — feedback model state, feedback export, notifications, and storage paths.

### 6. Run live without the dashboard

```bash
python -m traffic_intel.live \
  --camera 0 \
  --model yolo11n.pt \
  --imgsz 1280 \
  --event-dir events
```

## Deterministic crash regression

When the original source videos are present, cached YOLO outputs can be replayed through tracking, identity, and crash logic:

```bash
python tools/replay_cached_crash_regression.py
```

The slim package intentionally omits the original source videos. In that package, the self-contained stress runner remains available. Cached source-pixel replay requires restoring the referenced validation videos.

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
# result.crashes    -> pair-attributed probable incidents
# result.annotated  -> review/recording frame
```

Operational repository API:

```python
from traffic_intel.ops import IncidentStore

store = IncidentStore("data/traffic_intel.db")
incidents = store.list_incidents(status="pending", limit=50)
hotspots = store.hotspots(days=90)
model = store.feedback_model()
```

## Project layout

```text
traffic_intel/
├── core/               # engine, tracking, identity, geometry, scene handling
├── motion/             # calibration and speed estimation
├── incident/           # crash detector, FSM, incident visuals
├── recording/          # incident recorder and continuous archive
├── ops/                # operations DB, feedback learning, evidence export, notifications
├── app/                # dashboard runtime helpers and views
├── app.py              # Streamlit operations application
├── dashboard.py        # console launcher for Streamlit
├── live.py             # live camera adapter
├── validate_crashes.py # full-video validation
├── doctor.py           # environment readiness checks
└── domain.py           # shared observation records

tests/                  # unit and behavioral regression tests
validation/             # cached validation and stress-test outputs
docs/                   # architecture, validation, hardening, and operations notes
```

## Deployment scope

Version 0.12.0 is a strong **paid-pilot foundation** for incident discovery, evidence preparation, review workflow, and traffic analytics. It is not a claim of zero defects, emergency-grade certification, legally certified speed enforcement, or a complete enterprise chain-of-custody platform.

A production multi-user deployment should additionally provide authenticated access, role-based permissions, managed secrets, encrypted transport, centralized health monitoring, backups, deployment automation, independent validation, and jurisdiction-specific evidence controls.

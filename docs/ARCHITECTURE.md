# Architecture

## Design rule

The system separates **observation** from **incident interpretation**.

`TrafficEngine` is allowed to answer:

> Which trusted vehicles are visible, where are they, how reliable is the track, and—when calibrated—what is the estimated speed?

Incident analyzers are allowed to answer:

> What does the temporal interaction between those trusted observations mean?

This prevents crash policy, future wrong-way rules, stopped-vehicle logic, and UI behavior from contaminating the perception core.

## Layers

### 1. Perception

`engine.py`, `identity.py`, `tracking.py`, `scene.py`

Responsibilities:

- YOLO inference;
- vehicle-class filtering;
- class-agnostic NMS;
- ByteTrack association using raw, ephemeral tracker IDs;
- canonical physical-vehicle identity across conservative short gaps;
- raw-ID hijack detection and provisional-ID suppression;
- track maturity and trust scoring;
- source-change reset;
- canonical trusted `Detection` output.

A ByteTrack ID is not a physical-vehicle identity. `CanonicalIdentityManager` maps raw tracker handles onto stable per-camera canonical IDs using motion prediction, scale consistency, class history, and appearance when pixels are available. Ambiguous re-entry is deliberately left as a new identity rather than guessed. `TrackQualityGate` then decides whether that canonical track is mature and stable enough for downstream analytics.

### 2. Calibration and motion

`calibration.py`, `speed.py`

Responsibilities:

- project bottom-center image anchors to the measured road plane;
- reject extrapolated positions outside the supported calibration region;
- smooth image anchors before homography amplification;
- fit velocity over a robust recent trajectory;
- reject impossible jumps and long track gaps;
- expose reason codes and trajectory confidence.

Calibration fit quality and speed accuracy are deliberately treated as different concepts.

### 3. Incident semantics

`crash_detector.py`

The crash detector receives only confirmed tracks. It maintains bounded track and pair history, then evaluates:

- relative convergence and predicted closest approach;
- contact/near-contact geometry normalized by apparent vehicle size;
- apparent-depth compatibility using bottom-center geometry;
- impact-time trajectory discontinuity;
- temporal synchronization between participant impulses;
- candidate-only local optical-flow support;
- post-impact behavior as supporting evidence.

No single-track sudden stop or post-stop condition can independently emit a crash.

Time windows are expressed in seconds. Kinematic velocity is normalized to a 30-FPS-equivalent rate so the tuned thresholds retain approximately the same physical meaning at different source frame rates.

### 4. Pipeline coordination

`pipeline.py`

`TrafficIncidentPipeline` is the canonical frame-level path for live and validation adapters. It:

- runs perception;
- resets crash history when the source changes;
- runs crash analysis;
- applies incident annotations;
- returns a structured `PipelineFrame`.

This is where additional incident analyzers should be composed later.

### 5. Recording

`event_recorder.py`

`RollingSegmentBuffer` owns bounded pre-event video/telemetry and post-trigger capture. It does not own crash logic.

Important invariants:

- pre-event segment paths are monotonic and cannot be overwritten after a trigger;
- configured pre/post durations are instance state and are written accurately to `event.json`;
- a second trigger cannot replace an event already collecting its post window.

The recorder is operational incident capture. A future forensic-proof layer should wrap its output rather than be mixed into frame processing.

## Extension path for additional incidents

The next analyzers should consume the same trusted `Detection` stream:

```text
TrafficEngine
    ↓
raw tracker IDs
    ↓
canonical vehicle identities
    ↓
trusted detections
    ├── CrashDetector
    ├── StoppedVehicleDetector
    ├── WrongWayDetector
    ├── NearMissDetector
    └── HardBrakingDetector
          ↓
structured incident events
          ↓
recorder / database / dashboard / alerts
```

Each analyzer should own bounded temporal state and emit explicit participants, occurrence time, detection time, score, and evidence breakdown.

## 6. Operations and human review

`traffic_intel/ops/`

The operations layer is deliberately downstream of perception and incident semantics. It does not own detector state.

```text
TrafficIncidentPipeline
    ↓
CrashCandidate / trusted speed observations
    ↓
IncidentStore (SQLite pilot repository)
    ├── incidents
    ├── reviews
    ├── cameras
    ├── speed observations
    └── notification outbox
          ↓
Streamlit operations dashboard
    ├── overview
    ├── review queue
    ├── analytics
    ├── live monitor
    ├── evidence library
    └── settings
```

### Review and feedback boundary

`feedback.py` receives only persisted reviewed incidents. Approved and dismissed incidents become supervised labels. A regularized calibration model may then produce a learned review score after minimum dataset-size and class-balance gates are satisfied.

Important invariants:

- one operator click cannot directly retrain the live crash detector;
- raw detector score and evidence remain preserved;
- learned scores affect triage priority, not evidence retention;
- feedback can be exported for larger offline training and independent validation.

### Evidence export

`evidence.py` creates a portable ZIP containing a manifest, incident record, review history, clip, telemetry, and checksum files when available. Server-local absolute paths are not exposed in the portable manifest.

### Notifications

`notifications.py` delivers persisted webhook records from the notification outbox. Delivery status is explicit. A failed notification cannot delete or invalidate the underlying incident evidence.

### Analytics time semantics

Incident and speed timestamps are stored as Unix timestamps. Speed observations additionally persist the agency-local weekday and hour using an IANA timezone so day/hour analytics are not accidentally based on the server host timezone.

# Cam Architecture

## Overview

Cam is a three-layer system:

```
Camera → Edge Device (C++ inference) → Local Server (TS) → Cloud (TS)
```

---

## Layer 1: Edge Device

Runs at each camera location. Purpose: turn video into events in real time.

### Hardware Abstraction

Devices vary. The inference engine doesn't know or care which hardware it runs on.

```
Device (abstract capability)
├── GPU: CUDA | ROCm | OpenCL
├── NPU: Hailo | Google Coral | ...
├── CPU arch: x86_64 | ARM
├── Memory: capacity + bandwidth
└── Compute: TFLOPS / TOPS

→ DeviceManager selects optimal runtime + model variant for capabilities
→ Falls back gracefully (GPU → NPU → CPU)
```

Target for v1: NVIDIA Jetson Orin (most mature edge AI ecosystem).
But nothing in the pipeline code assumes Jetson.

### Inference Abstraction

The detection model is swappable behind a uniform interface:

```
Inference Backend (interface)
├── load_model(path, device) → ModelHandle
├── infer(frame) → Detections[]
│   └── Detections = [{bbox, class, confidence, mask?}]
└── set_batch_size(n)

Implementations:
├── TensorRT  (default — Jetson, NVIDIA GPUs)
├── ONNX Runtime (cross-platform fallback)
├── OpenVINO (Intel CPUs, GPUs, NPUs)
└── Future: CoreML, TFLite, custom accelerator SDK
```

The rest of the pipeline calls `infer(frame)` and gets detections back.
No YOLO-specific types leak past this boundary.

### Pipeline

```
Camera Stream
    │
    ▼
┌──────────────────┐
│     Capture      │  FFmpeg/libav, hardware decode (NVENC/VAAPI)
│  (Stream Layer)  │  Configurable framerate, auto-reconnect
└──────┬───────────┘
       │ raw frames
       ▼
┌──────────────────┐
│ Inference Engine │  Backend-agnostic (TensorRT | ONNX | OpenVINO)
│  (Model Layer)   │  Input: frame → Output: detections[]
└──────┬───────────┘
       │ bbox, class, confidence
       ▼
┌──────────────────┐
│  Vision Pipeline │  Tracking + re-identification
│  (Tracking)      │  SORT / BoT-SORT
│                  │  Input: detections → Output: tracked_objects[]
└──────┬───────────┘
       │ {id, trajectory[], class, confidence}
       ▼
┌──────────────────┐
│Feature Extractors│  Parallel processors attached to tracked objects
│  (Parallel)      │
│  ┌───────────┐  │  Speed: pixel displacement → world coords → km/h
│  │ Speed     │  │  Lane: position within lane geometry
│  ├───────────┤  │  Direction: trajectory heading
│  │ Lane      │  │  Size: bbox dimensions → vehicle classification
│  ├───────────┤  │  Each extractor is a self-contained module
│  │ Direction │  │
│  ├───────────┤  │
│  │ Size      │  │
│  └───────────┘  │
└──────┬───────────┘
       │ enriched tracked objects
       ▼
┌──────────────────┐
│   Event Engine   │  Plugin-based, evaluates rules
│  (Plugin System) │  Each plugin receives enriched objects and produces events
│                  │
│  ┌───────────┐  │  SpeedPlugin: speed > threshold → SpeedEvent
│  │ Speed     │  │  WrongWayPlugin: wrong direction → WrongWayEvent
│  ├───────────┤  │  CongestionPlugin: vehicle count/density → CongestionEvent
│  │ WrongWay  │  │  StoppedVehiclePlugin: stationary for N sec → ObstructionEvent
│  ├───────────┤  │  Future plugins: pedestrian, flood, parking, debris...
│  │ Congestion│  │
│  ├───────────┤  │
│  │ Stopped   │  │  Plugins are .so / .wasm loaded at runtime
│  └───────────┘  │  Crash isolation per plugin
└──────┬───────────┘
       │ structured events
       ▼
┌──────────────────┐
│       Sync       │  MQTT for real-time events
│  (Connectivity)  │  HTTP/2 for clips + batches
│                  │  Store-and-forward when offline
│                  │  Bandwidth management + priority queuing
└──────────────────┘
```

### Pipeline Threading

```
Capture ──→ FrameQueue ──→ Inference ──→ Vision ──→ Extractors ──→ EventEngine ──→ Sync
thread        thread        thread       thread      thread pool      thread         async I/O
```

- Separate thread per stage
- Bounded queues between stages (backpressure)
- Feature extractors run in a thread pool (parallel per object)
- All network I/O is async (libuv/asio)

### AI Scheduler

Edge devices have finite compute. When multiple plugins are active, the scheduler allocates inference budget.

```
Frame Budget (e.g., 30 FPS target)
│
├── Critical (guaranteed)
│   ├── Speed Detection       → 15 FPS (hard guarantee)
│   └── Wrong Way Detection   → 15 FPS (hard guarantee)
│
├── Normal (best-effort, shares remaining budget)
│   ├── Congestion            → 5 FPS (proportional allocation)
│   └── Vehicle Counting      → 5 FPS
│
├── Low Priority (opportunistic)
│   ├── Parking Occupancy     → 1 FPS (when budget allows)
│   └── Bicycle Detection     → 1 FPS
│
└── Idle (spare cycles)
    ├── Infrastructure Inspection → every 60s
    └── Pothole Detection         → every 60s
```

Behavior:
- **Critical** plugins get reserved frame slots. Always.
- **Normal** plugins share remaining budget proportionally.
- **Low priority** runs opportunistically when budget allows.
- **Idle** runs only when no other plugin needs the slot.
- **Under load**: lowest-priority frames dropped first.
- **Over-provisioning**: warn at deploy time if committed budget exceeds device capability.

The scheduler is a configuration document, not hardcoded:

```json
{
  "frame_budget": 30,
  "allocation": {
    "speed": {"priority": "critical", "reserved_fps": 15},
    "wrong_way": {"priority": "critical", "reserved_fps": 10},
    "congestion": {"priority": "normal", "weight": 1.0}
  },
  "device_cap": "jetson_orin_nx_16gb"
}
```

### Calibration Service

Speed measurement is only as good as the calibration. This gets its own subsystem.

```
Calibration
├── Camera Registration  — map camera pixel space to real-world coordinates
├── Perspective Transform — homography matrix (pixel ↔ world)
├── Lane Geometry        — lane boundaries, widths, directions, stop lines
├── Measurement Zones    — speed trap regions, detection areas
├── Validation           — reference patterns, sanity checks, confidence score
└── Versioning           — every change tracked, auditable, deployable
```

Calibration flow:

```
Install → rough (click 4 points) → AI-assisted refinement → validate → deploy
                        ↕
            Recalibrate when: camera moved, lens changed, road work
```

Calibration is a JSON document stored on the device and versioned in the cloud.
A camera nudge shouldn't require a site visit — remote recalibration via reference markers.

### Local Storage (on device)

- **SQLite**: Event queue, device config, calibration, plugin state
- **Circular video buffer**: H.264 encoded, configurable duration, event-triggered clip extraction

### Offline Behavior

Device buffers events + clips locally. On reconnect:
1. Upload event metadata first (small, fast)
2. Upload clips in priority order
3. Rate-limited to avoid saturating the link

---

## Layer 2: Local Server

Per-municipality backend. Runs in Docker on-prem or in SaaS.

### Architecture

```
                      ┌─────────────┐
                      │   Traefik   │  TLS termination, routing
                      └──────┬──────┘
                             │
                    ┌────────┴────────┐
                    │   API Gateway   │  REST + WebSocket
                    │   (TS/Node)     │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │ Internal Event  │  In-process dispatcher (v1)
                    │      Bus        │  → NATS / Redis Streams (v2+)
                    └────────┬────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
          ▼                  ▼                  ▼
   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
   │  Analytics   │   │    Alerts    │   │ Persistence  │
   │  Service     │   │    Service   │   │   Service    │
   │              │   │              │   │              │
   │ TimescaleDB  │   │ Email/SMS/   │   │ PostgreSQL   │
   │ continuous   │   │ Webhook/Push │   │ TimescaleDB  │
   │ aggregates   │   │              │   │              │
   └──────────────┘   └──────────────┘   └──────────────┘
          │                                      │
          │                                      │
          ▼                                      ▼
   ┌──────────────┐                      ┌──────────────┐
   │  WebSocket   │                      │  Media Store │
   │  Publisher   │                      │  (MinIO S3)  │
   └──────────────┘                      └──────────────┘
          │
          ▼
   ┌──────────────┐
   │  Dashboard   │
   │  (React SPA) │
   └──────────────┘

External inputs:
   MQTT Broker (EMQX) ──→ Ingestion Service ──→ Internal Event Bus
   Edge Devices (HTTP) ──→ Media Service ──→ MinIO
```

### Internal Event Bus

This is the backbone for decoupling. v1 is an in-process event dispatcher (zero deps, zero overhead). The interface is stable — swap the implementation later.

```
Event Bus Interface:
  publish(event: Event): void
  subscribe(handler: EventHandler): Subscription
  subscribeType(eventType: string, handler: EventHandler): Subscription

Implementations:
  v1: InProcessEventBus  (simple, no deps)
  v2: RedisStreamsBus    (when multi-process needed)
  v3: NATSBus            (when multi-server needed)

Consumers (all subscribe via the bus):
  Analytics Service  → aggregates, continuous queries
  Alert Service      → rule evaluation, notifications
  Persistence Service → write to DB
  WebSocket Publisher → push to dashboard clients
  Cloud Sync Agent   → batch to cloud (opt-in, configurable filters)
```

The bus is synchronous in-process for v1 (no serialization overhead, no new infrastructure). The abstraction means nothing changes when you need Redis or NATS.

### Services

| Service | Role |
|---------|------|
| **API Gateway** | REST endpoints, WebSocket, auth, rate limiting |
| **Ingestion** | Consumes MQTT stream, publishes to internal event bus |
| **Analytics** | Processes events into aggregates (continuous queries on TimescaleDB) |
| **Alerts** | Evaluates rules, sends email/SMS/webhook/push notifications |
| **Persistence** | Writes events, vehicles, clips metadata to DB |
| **Media** | Manages clip upload, transcoding, serve lifecycle |
| **WebSocket Publisher** | Pushes live events to connected dashboards |
| **Sync Agent** | Batches aggregated data to cloud (opt-in per customer) |

### Database Schema (Core Tables)

```
events                                     (hypertable, partitioned by timestamp)
├── id              UUID PK
├── device_id       UUID FK → devices
├── event_type      VARCHAR  (not enum — plugins define types dynamically)
├── plugin_id       VARCHAR  (which plugin generated this)
├── timestamp       TIMESTAMPTZ
├── location        GEOGRAPHY(point)
├── metadata        JSONB   (plugin-specific payload)
├── clip_id         UUID FK → clips (nullable)
└── vehicle_ids     UUID[]  (FK → vehicles)

vehicles                                   (hypertable)
├── id              UUID PK
├── event_id        UUID FK → events
├── track_id        INT     (from tracker)
├── vehicle_class   VARCHAR (car, truck, bus, motorcycle, bicycle, unknown)
├── speed           FLOAT   (km/h, populated by speed extractor)
├── direction       FLOAT   (heading in degrees)
├── lane            INT
├── trajectory      JSONB   (array of {x, y, timestamp})
└── features        JSONB   (extensible — extractor outputs are key-value)

hourly_aggregations                       (continuous aggregate, TimescaleDB)
├── device_id
├── hour             TIMESTAMPTZ
├── vehicle_count    INT
├── vehicle_counts   JSONB  (per-class breakdown)
├── avg_speed        FLOAT
├── p50_speed        FLOAT
├── p95_speed        FLOAT
├── max_speed        FLOAT
└── event_counts     JSONB  (per-type breakdown)

clips
├── id              UUID PK
├── device_id
├── timestamp
├── duration        INT (seconds)
├── size_bytes      BIGINT
├── storage_path    VARCHAR
└── status          ENUM (uploading, ready, archived, deleted)

devices                                    (captures full device management)
├── id              UUID PK
├── name            VARCHAR
├── location        GEOGRAPHY
├── hardware_model  VARCHAR
├── os_version      VARCHAR
├── firmware        VARCHAR
├── runtime_version VARCHAR
├── model_id        VARCHAR (current active model)
├── calibration_id  UUID FK → calibrations
├── status          ENUM (provisioning, online, offline, degraded, error)
├── last_seen       TIMESTAMPTZ
├── public_key      TEXT
├── certificate     TEXT
├── config          JSONB
├── capabilities    JSONB  (device capability report)
└── health          JSONB  (latest telemetry snapshot)

calibrations
├── id              UUID PK
├── device_id       UUID FK → devices
├── version         INT
├── homography      FLOAT[][] (3x3 perspective transform matrix)
├── lanes           JSONB  (lane geometry)
├── zones           JSONB  (measurement zones)
├── confidence      FLOAT
├── created_by      VARCHAR (auto | user:email)
├── created_at      TIMESTAMPTZ
└── active          BOOLEAN

plugins
├── id              VARCHAR PK  (e.g. "speed", "wrong_way")
├── name            VARCHAR
├── version         VARCHAR
├── tier            VARCHAR  (starter, professional, enterprise)
├── enabled         BOOLEAN
├── config          JSONB
└── binary_hash     VARCHAR (for .so/.wasm verification)

device_telemetry                           (hypertable, high-frequency)
├── device_id       UUID
├── timestamp       TIMESTAMPTZ
├── cpu_usage       FLOAT
├── gpu_usage       FLOAT
├── memory_usage    FLOAT
├── temperature     FLOAT
├── disk_usage      FLOAT
├── ssd_wear        FLOAT
├── camera_fps      FLOAT
├── dropped_frames  INT
├── uptime          INT
└── connection_rssi INT
```

### Continuous Aggregates (TimescaleDB)

Raw events kept 30–90 days. Continuous aggregates roll up:
- **Hourly**: vehicle counts (total + per-class), speed percentiles, event counts
- **Daily**: peak hour analysis, compliance rates (speed limit adherence), trends
- **Monthly**: year-over-year comparisons, infrastructure health scores

Queries for historical trends hit aggregates, not raw events.

---

## Layer 3: Cloud Platform

Multi-tenant SaaS on Kubernetes. Same Docker images as Local Server + cloud-specific services.

### Additional Services

| Service | Role |
|---------|------|
| **Device Management** | Fleet-wide device lifecycle (see below) |
| **Model Registry** | ML model versioning, A/B testing, staged rollout |
| **OTA Update Service** | Push firmware/models/plugins to edge devices |
| **Plugin Store** | Plugin catalog, licensing, per-customer module activation |
| **Billing** | Subscription management, usage metering per plugin |
| **Aggregated Analytics** | Cross-city dashboards (opt-in) |
| **License Server** | Software licensing for on-prem deployments (offline-capable) |
| **Support Portal** | Remote diagnostics, device SSH tunnel (opt-in) |
| **Calibration Manager** | Remote calibration review, approval, push |

### Device Management Subsystem

This deserves first-class status — fleet operations become critical at 100+ devices.

```
Device Management
│
├── Provisioning
│   ├── Certificate enrollment (auto, on first boot)
│   ├── Initial config push (location, camera URL, calibration)
│   └── Validation (test pattern, connectivity check, inference smoke test)
│
├── Health Monitoring
│   ├── CPU/GPU/memory/temperature (real-time + historical)
│   ├── SSD wear prediction
│   ├── Camera FPS + dropped frames
│   ├── Connectivity quality (latency, packet loss, RSSI)
│   ├── Inference latency (p95, max)
│   └── Anomaly detection (auto-flag degradation)
│
├── OTA Lifecycle
│   ├── Firmware updates (signed, staged rollout, rollback on failure)
│   ├── Model updates (canary → 10% → 50% → all, auto-rollback on metrics)
│   ├── Plugin updates (add/remove/refresh event plugins)
│   ├── Config updates (targeted or fleet-wide)
│   └── Update policy (maintenance window, max concurrent, speed tiers)
│
├── Remote Management
│   ├── Reboot / restart service
│   ├── Log streaming (real-time via WebSocket tunnel)
│   ├── Diagnostic tunnel (SSH, opt-in per session, audit-logged)
│   ├── Factory reset
│   └── LED pattern control (identify device physically)
│
└── Fleet Analytics
    ├── Uptime SLA tracking per device + aggregate
    ├── Failure prediction (SSD wear, thermal throttling, camera degradation)
    ├── Fleet-wide health dashboard
    ├── Software version compliance
    └── Certificate expiry monitoring + auto-renewal
```

### Multi-Tenancy

```
SaaS tier:     Shared DB, row-level tenant_id
Professional:  Schema per tenant
Enterprise:    Dedicated DB instance per tenant
```

Same codebase, switching strategy via config.

---

## Plugin System

The event plugin system is the monetization engine. Plugins turn the platform from "speed camera" into "traffic analytics for any use case."

### Architecture

```
Event Engine (C++)
├── Plugin Manifest (name, version, tier, dependencies, hooks)
├── Hook Points
│   ├── on_tracked_object(object) → Option<Event>      (per-object)
│   ├── on_frame(frame, objects) → Vec<Event>           (per-frame summary)
│   └── on_aggregate(window_data) → Option<Event>       (per-time-window)
├── Plugin Registry
│   ├── speed.so
│   ├── wrong_way.so
│   ├── congestion.so
│   ├── stopped_vehicle.so
│   └── ... (loaded at runtime from /var/cam/plugins/)
└── Sandbox
    ├── Resource limits (CPU time, memory, event throughput per plugin)
    ├── Crash isolation (plugin crash → restart plugin, not the pipeline)
    └── Rate limiting (max events/second per plugin)
```

### Plugin Lifecycle

```
Cloud Plugin Store → OTA push → Edge downloads + verifies signature → Hot-reload
                                                                   ↓
                                                            Restart on crash
                                                                   ↓
                                            Cloud can disable remotely (license expiry)
```

### Licensing Model

| Tier | Included Plugins | Revenue Model |
|------|-----------------|---------------|
| Starter | Speed | Subscription |
| Professional | Speed + Wrong Way + Congestion | Subscription |
| Enterprise | All plugins + custom development | Contract |
| Add-on | Individual plugins (School Zone, Flood, Parking...) | Per-plugin monthly |

This turns new features into recurring revenue without requiring a platform re-architecture.

### Plugin SDK

As the platform matures, partners will build plugins. The SDK is the toolchain.

```
cam-plugin new my_plugin    # Scaffold from template (C++ or Rust)
cam-plugin build            # Compile to .so (C++) or .wasm
cam-plugin sign             # Sign with developer key
cam-plugin test             # Run against mock Event Engine with test fixtures
cam-plugin upload           # Push to Plugin Store (requires approval)
cam-plugin version          # Bump version, update manifest
```

### Plugin ABI

The ABI is the contract between the Event Engine and plugins. Must be stable across versions.

```cpp
// cam_plugin.h — the public API (ABI-stable)
#define CAM_PLUGIN_API_VERSION 1

typedef struct {
    uint32_t api_version;
    const char* plugin_id;
    const char* version;
    uint32_t hooks;  // bitmask: 1<<on_tracked_object, 1<<on_frame, etc.
} CamPluginManifest;

typedef struct {
    double timestamp;
    uint32_t track_id;
    uint8_t class_id;
    float bbox[4];       // x, y, w, h (normalized 0-1)
    float confidence;
    float speed;         // km/h (0 if unavailable)
    uint8_t lane;
    float heading;
    const char* features; // JSON string from extractors
} CamTrackedObject;

// Plugin lifecycle
CamPluginManifest* cam_plugin_init();
void cam_plugin_cleanup();
uint32_t cam_plugin_on_tracked_object(CamTrackedObject* obj, CamEvent* out, uint32_t max_events);
```

The ABI is C-compatible (no name mangling), explicitly versioned, and guaranteed backward-compatible within a major API version.

### Plugin Store

```
Plugin Store (Cloud)
├── Public catalog (approved plugins, listed by tier)
├── Developer portal (upload, version management, download analytics)
├── Signing service (verify developer identity, sign binaries)
└── Customer portal (browse, enable/disable, configure)
```

Plugins are signed at the Cloud, verified on the Edge. A compromised plugin affects only itself.

---

## Event Model

Every event in the system shares a uniform envelope. Downstream consumers never parse plugin-specific formats.

```cpp
// Canonical event structure (C++ reference, mirrored in TypeScript)
struct Event {
    uuid_t          id;              // v7 UUID (timestamp-ordered)
    int64_t         timestamp_ns;    // nanosecond precision
    uuid_t          device_id;
    char            type[64];        // namespaced: "traffic.speed", "safety.wrong_way"
    char            plugin_id[64];   // originating plugin
    uint8_t         severity;        // 0=info, 1=warning, 2=critical
    float           confidence;      // 0.0–1.0
    uint8_t         source;          // 0=auto, 1=verified, 2=manual
    char*           metadata;        // JSON string (plugin-specific payload)
    uuid_t          clip_id;         // optional — nil if no clip
    char            twin_path[256];  // optional — digital twin entity path
};
```

JSON representation (for MQTT transport):

```json
{
  "id": "018f3f6a-7b3c-7d00-9a5c-3e8f1b2c4d5e",
  "ts": 1787654321000000000,
  "device": "dev_abc123",
  "type": "traffic.speed",
  "plugin": "speed",
  "severity": 2,
  "confidence": 0.97,
  "source": 0,
  "metadata": {
    "vehicle_class": "car",
    "speed_kmh": 72,
    "speed_limit": 45,
    "lane": 2,
    "direction": "N"
  },
  "clip_id": "clip_def456",
  "twin_path": "city/main_st/segment_3/lane_2"
}
```

All events use this envelope. The `type` field is namespaced (`traffic.*`, `safety.*`, `analytics.*`, `infrastructure.*`) so consumers can subscribe to categories. The `metadata` field carries plugin-specific payload; no downstream consumer ever needs to understand plugin internals.

---

## Calibration Service

Speed data without calibration is noise. This is treated as a discrete subsystem.

### Calibration Document

```json
{
  "id": "cal_abc123",
  "device_id": "dev_xyz789",
  "version": 4,
  "homography": [[...], [...], [...]],
  "lanes": [
    { "id": 1, "direction": "N", "boundary": [[x1,y1], [x2,y2], ...] }
  ],
  "zones": [
    { "id": "speed_trap_1", "type": "speed", "polygon": [...], "speed_limit": 45 }
  ],
  "reference_points": [
    { "pixel": [100, 200], "world": [0, 10], "label": "stop_bar" }
  ],
  "confidence": 0.94,
  "created_at": "2026-07-01T12:00:00Z",
  "created_by": "auto"
}
```

### Calibration Flow

```
1. Install camera
2. Technician places 4+ reference markers in frame (known real-world distances)
3. AI-assisted refinement (system proposes homography, human validates)
4. Validation run: measure known reference vehicle speeds, check error margin
5. Deploy to edge device
6. Version bump on every change
7. Alerts if confidence drops below threshold (camera may have moved)
```

Versioning is critical. Camera nudges happen. Having auditable, revertable calibration history prevents data quality disasters.

---

## Digital Twin

Events are temporal — they happen and pass. The Digital Twin is the persistent state of the road network.

```
Before (event-centric):
  Camera → Events → Query "what events happened on Lane 2?"

After (state-centric):
  Camera → Events → Update Twin → Query "what is Lane 2 doing right now?"
```

This is the difference between a log and a live dashboard.

### Hierarchy

```
City
├── RoadSegment (id, name, bounds, speed_limit)
│   ├── Lane (id, direction, current_speed, vehicle_count, occupancy%)
│   └── Lane
├── Intersection (id, congestion_level, phase, avg_wait)
│   ├── Approach (id, vehicle_count, queue_length)
│   └── Approach
├── SchoolZone (id, active, speed_compliance%, current_vehicles)
├── Crosswalk (id, pedestrian_activity, last_crossing)
└── Camera (id, status, calibration_version, connected_segments)
```

Each node maintains:
- **Current state**: latest known values
- **Last updated**: timestamp of last change
- **Confidence**: how reliable the current state is
- **Trend**: direction of change over last N minutes

### How It Works

```
Edge Event (vehicle_id=42, lane=2, speed=48)
    │
    ▼
Digital Twin update: RoadSegment_1.Lane_2.current_speed = 48
                     RoadSegment_1.Lane_2.last_vehicle_at = now
                     RoadSegment_1.Lane_2.vehicle_count++
    │
    ▼
Dashboard sees: Lane 2 flowing at 48 km/h (no query needed — state pushed)
```

### Edge vs Server Twin

| Location | Scope | Persistence | Update Freq |
|----------|-------|-------------|-------------|
| Edge | Road segments this camera sees | In-memory | Per-frame (30 FPS) |
| Local Server | Full municipality | In-memory + persisted to DB | Per-event |
| Cloud | All municipalities (opt-in) | Aggregated state snapshots | Per-minute |

Edge twin is ephemeral (rebuilt from detections on restart).
Server twin is persistent (recorded as state snapshots, queryable historically).

### Architecture Mapping

The Digital Twin lives as a service in the Local Server:

```
Internal Event Bus
    │
    ▼
┌──────────────────┐
│ Digital Twin     │  In-memory graph + persisted state
│ Service          │
│                  │  Input: Event → mutate twin state
│                  │  Output: state change notifications
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ WebSocket        │  Push state deltas to dashboard
│ Publisher        │  (changes only, not full state, ~100ms batches)
└──────┬───────────┘
       │
       ▼
┌──────────────────┐
│ TimescaleDB      │  Persist state snapshots
│ (twin_state)     │  Query: "What was Lane 2's speed at 3pm yesterday?"
└──────────────────┘
```

### State Snapshots Table

```
twin_state                               (hypertable)
├── id              UUID PK
├── twin_path       TEXT   (e.g. "city/road_segment_1/lane_2")
├── state           JSONB  (current values: speed, count, occupancy...)
├── confidence      FLOAT
└── recorded_at     TIMESTAMPTZ
```

This enables time-travel queries: "Show me every intersection's state at 5pm last Friday."

---

## Data Flow Summary

```
Real-time path (ms latency):
  Camera → Capture → Inference → Tracking → Features → Event Engine → MQTT → Server Ingestion
  → Event Bus → Persistence (DB) + WebSocket Publisher → Dashboard updates

Analytics path (minute latency):
  Events in DB → TimescaleDB continuous aggregate → API → Dashboard charts

Cloud sync path (configurable, opt-in):
  Local Server sync-agent → batched HTTPS → Cloud API → Cloud DB → Multi-city Dashboard

OTA path:
  Cloud Device Management → OTA Service → Edge OTA Agent → download → verify → hot-reload
```

---

## API Surface

### Edge → Server

| Transport | Protocol | Direction | Payload |
|-----------|----------|-----------|---------|
| MQTT | JSON events | Edge → Server | Event metadata (small, real-time) |
| HTTP/2 | Multipart | Edge → Server | Video clips (large, async) |
| HTTP/2 | JSON batch | Edge → Server | Batched event metadata (reconnect) |
| HTTP/2 | JSON | Edge → Server | Device telemetry (1–5s interval) |
| HTTP/2 | JSON | Edge → Server | Calibration validation results |
| HTTP/2 | JSON | Server → Edge | Device config (target) |
| MQTT | JSON cmd | Server → Edge | reboot, update_*, set_config, run_diagnostics |
| HTTP/2 | Binary | Server → Edge | OTA payload (firmware, model, plugin) |

### Dashboard → Server

REST + WebSocket:

```
GET    /api/v1/events          # Query events (filters, pagination)
GET    /api/v1/events/:id      # Single event with trajectory + clip
GET    /api/v1/analytics/speed # Speed stats (aggregated)
GET    /api/v1/analytics/counts
GET    /api/v1/analytics/congestion
GET    /api/v1/analytics/compliance  # Speed limit compliance %
GET    /api/v1/devices             # Fleet list + health summary
GET    /api/v1/devices/:id         # Single device detail
PUT    /api/v1/devices/:id/config  # Update device config
POST   /api/v1/devices/:id/command # Send command (reboot, update, etc.)
GET    /api/v1/devices/:id/telemetry # Historical telemetry
GET    /api/v1/calibrations        # Calibration records
POST   /api/v1/calibrations        # Create/update calibration
GET    /api/v1/clips/:id           # Serve video clip
GET    /api/v1/alerts              # Active alerts
POST   /api/v1/alerts/rules        # Configure alert rules
GET    /api/v1/plugins             # Available + enabled plugins
PUT    /api/v1/plugins/:id/enable  # Enable/disable plugin
```

Real-time via WebSocket:

```
Client → Server: subscribe(event_types[], device_ids[], filters)
Server → Client: { type: "event", payload: Event }
Server → Client: { type: "alert", payload: Alert }
Server → Client: { type: "device_status", payload: DeviceStatus }
Server → Client: { type: "health", payload: Telemetry }
```

---

## Security

| Concern | Mechanism |
|---------|-----------|
| Edge ↔ Server | TLS 1.3, mutual TLS (client cert for each device) |
| Server ↔ Cloud | TLS 1.3, API keys rotated monthly |
| Dashboard ↔ Server | TLS 1.3, JWT sessions, RBAC (admin, operator, viewer) |
| OTA payloads | Signed with offline-capable key, verified on device |
| Data-at-rest | Encrypted volumes (LUKS on edge, EBS/PD on cloud) |
| Video clips | Encrypted at rest (MinIO SSE), access-logged |
| Audit trail | All user actions + config changes logged, append-only |
| Plugin sandbox | Resource limits, crash isolation, rate limiting |
| Certificate lifecycle | Auto-enrollment on provision, auto-renewal before expiry |
| Network isolation | Edge devices can only talk to Server (no peer-to-peer) |

---

## Deployment Topologies

### Starter (SaaS)

```
Cloud (K8s):
  ┌──────────────────────────────────┐
  │ API + Event Bus + Analytics      │
  │ + Alerts + Persistence + WS Pub  │
  │ + Device Management + OTA        │
  │ + Calibration Manager            │
  │                                  │
  │ db: RDS PostgreSQL/TimescaleDB   │
  │ media: S3-compatible object store │
  │ mqtt: EMQX (managed or sidecar)  │
  └────────────┬─────────────────────┘
               │ MQTT + HTTP
          ╔════╧════╗
          ║  Edge   ║  devices connect directly to cloud
          ╚════════╝
```

### Professional (Hybrid)

```
Customer Site:
  ┌───────────────────────────────────┐
  │ Docker Compose:                   │
  │  traefik + api + event bus        │
  │  + analytics + alerts + ws        │
  │  + persistence + media + mqtt     │
  │  + sync-agent                     │
  │                                   │
  │  db: pg+timescaledb               │
  │  media: minio                     │
  └───────────────┬───────────────────┘
                  │ encrypted sync (aggregated + opt-in)
        ┌─────────┴─────────┐
        │  Your Cloud (K8s) │ fleet mgmt, aggregated analytics, plugins
        └───────────────────┘
```

### Enterprise (On-Prem)

```
Customer Data Center:
  ┌───────────────────────────────────────┐
  │ Same Docker Compose as Professional   │
  │                                       │
  │ No data leaves customer environment.  │
  │ Cloud only: license check (offline-   │
  │ capable, can go months without),      │
  │ support tunnel (opt-in, per-session)  │
  └───────────────────────────────────────┘
```

---

## ML Model Pipeline

```
Training (your infra, GPU cluster)
    │
    ▼
Model Registry (versioned, metadata tagged)
    │
    ├── Optimize: FP16 / INT8 quantization
    ├── Convert: TensorRT engine | ONNX | OpenVINO IR
    │
    ▼
OTA Distribution (staged rollout)
    │
    ▼
Edge Device (verify signature → deploy → A/B inference → promote/rollback)
```

- YOLOv8 variants: nano (60 FPS on Jetson Nano), small (45 FPS), medium (30 FPS)
- Quantization: FP16 default, INT8 for aggressive optimization
- Model packaged with calibration metadata (input size, class names, preprocessing params)
- Rollback: if key metrics (detection rate, false positives) degrade, auto-revert

---

## Cam Edge OS (Long-term Vision)

The edge device as an appliance, not a Linux box.

```
Cam Edge OS
│
├── Base
│   ├── Ubuntu Core / custom Yocto (minimal, read-only rootfs)
│   └── Signed boot chain
│
├── Runtime (service host)
│   ├── Runtime Manager   (service lifecycle, IPC bus, resource monitoring)
│   ├── cam-health        (health check — if Runtime dies, watchdog reboots)
│   └── Local IPC Bus     (in-process message bus for service-to-service calls)
│
├── Services (managed by Runtime)
│   ├── cam-inference     (inference engine — backend-agnostic)
│   ├── cam-vision        (tracking + feature extractors)
│   ├── cam-events        (plugin engine, sandboxed)
│   ├── cam-sync          (connectivity manager, store-and-forward)
│   ├── cam-buffer        (video circular buffer, clip extraction)
│   ├── cam-calibration   (calibration service, version management)
│   ├── cam-twin          (local digital twin — partial, just this camera's view)
│   └── cam-scheduler     (AI scheduler — frame budget allocation)
│
├── System Services
│   ├── cam-watchdog      (hardware health, auto-recovery, crash notification)
│   ├── cam-ota           (update orchestrator, rollback support)
│   ├── cam-certd         (certificate lifecycle, auto-renew)
│   ├── cam-diag          (metrics, logs, crash dump collection)
│   ├── cam-fw            (minimal firewall — only MQTT + HTTP/2 outbound)
│   └── cam-led           (LED patterns: power, connectivity, health, identify)
│
├── Management
│   ├── Local: minimal web UI (status only, no config)
│   ├── Cloud: full management via Device Management API
│   └── Emergency: physical reset button, serial console (locked)
│
└── Appliance Properties
    ├── Boots directly to Cam (no user-accessible shell)
    ├── Factory reset via pinhole button
    ├── Tamper-evident casing
    ├── Power-loss recovery (filesystem is read-only + overlay)
    └── UPS-backed graceful shutdown
```

Customers don't manage "a Linux server running our software."
They manage **Cam Edge** — an appliance that does one thing and does it well.

---

## ponytail: What's deliberately excluded

| Thing | Why not now | When to add |
|-------|-------------|-------------|
| **Service mesh** | Adds complexity without benefit at this scale | 50+ services running on K8s |
| **Event sourcing / CQRS** | TimescaleDB handles read+write load. CQRS adds write-path complexity | Analytics queries start impacting event ingestion |
| **Data lake (Parquet/S3)** | Raw events in TimescaleDB suffice. Offloading adds ETL code | Retention > 90 days or petabyte scale |
| **Kafka / Pulsar** | Heavy ops burden. MQTT + internal event bus covers every need | Cross-region event replication or 10K+ devices |
| **gRPC for edge** | HTTP/2 + MQTT cover everything. gRPC adds build complexity for C++ client | Bi-directional streaming with backpressure becomes a bottleneck |
| **Kubernetes on edge** | systemd + containers are simpler and more reliable | Fleet exceeds 1000 devices and OTA needs orchestration |
| **WASM plugins on edge** | .so is simpler for C++. WASM adds a runtime dependency | Third-party plugin developers who want language-agnostic sandboxing |
| **Custom hardware** | Jetson is proven. Custom hardware is a second company | Volume justifies NRE (50K+ units) |
| **Video analytics ML training pipeline** | Scope beyond Cam's core. Use existing ML ops tools | Team has dedicated ML engineers |
| **Real-time ML model retraining** | Federated learning at edge is bleeding-edge complexity | Model degradation becomes measurable and costly |
| **WASM plugin SDK** | C .so ABI is simpler, faster, and sufficient for partners writing C++/Rust | Third-party plugins need language-agnostic sandboxing |
| **Digital twin time-travel** | State snapshots provide basic historical query. Full time-travel needs a materialization view | Users consistently request "show me this intersection at 5pm yesterday" |
| **Partner plugin marketplace** | Plugin Store with manual approval suffices for v1 | 100+ plugins needing automated review, ratings, and revenue sharing |
| **Custom Cam OS image** | Ubuntu Server + systemd + custom packages = same operational behavior, less maintenance overhead | Fleet reaches 500+ devices and image management becomes a bottleneck |

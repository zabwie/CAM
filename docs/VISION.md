# Vision — Traffic Intelligence & Forensic Incident Platform

## Product direction

A deployable camera intelligence layer that turns existing road video into trusted vehicle tracks, calibrated motion measurements, real-time incident events, and reviewable incident packages.

The architecture is intentionally sensor-agnostic: cameras are the current perception source; radar, ANPR, and other sensors can later contribute independent observations without rewriting incident semantics.

## Capability status

Legend: **✓ implemented**, **◐ experimental / pilot-ready**, **○ planned**.

### Perception

- ✓ Vehicle detection: car, truck, bus, motorcycle.
- ✓ Class-agnostic duplicate suppression before tracking.
- ✓ ByteTrack identity continuity within a camera view.
- ✓ Downstream track-quality gate with maturity and instability rejection.
- ✓ ROI filtering.
- ✓ Scene/source-change reset.
- ○ Long-occlusion appearance ReID.
- ○ Cross-camera identity association.

### Motion and speed

- ✓ Homography road-plane projection.
- ✓ Robust trajectory-based speed estimation.
- ✓ Track-gap and outlier rejection.
- ✓ Per-reading confidence components and reason codes.
- ◐ Calibration-fit diagnostics.
- ○ Independent ground-truth speed validation against radar or controlled reference runs.

### Incident detection

- ◐ Vehicle-to-vehicle crash detection with pair attribution and impact-time timestamps.
- ✓ Hard-stop rejection as a crash false-positive guard.
- ✓ Uninvolved-nearby-vehicle attribution regression coverage.
- ✓ Internal near-miss state used by crash reasoning.
- ○ Standalone hard-braking event stream.
- ○ Stopped vehicle in travel lane.
- ○ Wrong-way movement.
- ○ General trajectory anomaly / off-road movement.
- ○ Standalone near-miss alerts.
- ○ Pedestrian/cyclist incident models.

### Incident capture

- ✓ Rolling annotated-video buffer.
- ✓ Synchronized telemetry CSV.
- ✓ Structured event metadata with impact and detection frames.
- ✓ Video SHA-256.
- ○ Signed multi-file manifest.
- ○ Raw-source preservation separate from annotated review video.
- ○ Immutable storage and custody/access audit log.

### Traffic analytics

- ○ Lane-level counts and turn movements.
- ○ Flow, occupancy, headway, and congestion metrics.
- ○ Historical aggregation and per-camera health.

### Operator product

- ◐ Basic Streamlit review UI.
- ○ Production alert queue and incident review workflow.
- ○ Multi-camera dashboard.
- ○ Search and evidence export.
- ○ Deployment/health monitoring.

## Deployment stages

| Stage | Scope | Goal |
|---|---|---|
| Engineering demo | Recorded and live single-camera feeds | Show trusted tracking, speed foundation, crash attribution, and incident capture |
| Paid pilot | 1–3 calibrated cameras | Measure real false-alarm rate, incident recall, latency, and operational workflow |
| Production | Multi-camera service | Add persistent storage, alert review, signed evidence manifests, health monitoring, and optional sensor fusion |

## Why confidence matters

A numerical output is not automatically trustworthy. Every measurement should carry enough context to explain why it was accepted: calibration quality, trajectory stability, visibility, zone position, track maturity, and incident evidence.

That makes the system auditable and suitable for independent evaluation. Legal admissibility or enforcement use requires separate validation and jurisdiction-specific controls; the software should preserve the information needed for that evaluation rather than claiming certification by itself.

## Commercial model

The long-term product is a per-camera or per-site software service covering:

- traffic perception and health monitoring;
- calibration and measurement quality;
- incident alerts and review;
- searchable traffic analytics;
- incident package retention and export.

The current codebase is the perception and incident-engine foundation for that service.

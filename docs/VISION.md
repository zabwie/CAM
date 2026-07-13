# Vision — Roadway Incident Intelligence & Operations Platform

## Product direction

The product is not a speedometer with a crash alert attached. It is an operational layer that turns existing roadway cameras into searchable incidents, preserved evidence, human review, and long-term roadway intelligence.

The core promise is:

> Detect what may matter, preserve the relevant evidence automatically, put it in front of a human quickly, and turn accumulated observations into decisions about where and when risk is increasing.

## Capability status

Legend: **✓ implemented**, **◐ pilot-ready / experimental**, **○ planned**.

### Perception

- ✓ Vehicle detection: car, truck, bus, motorcycle.
- ✓ Class-agnostic duplicate suppression before tracking.
- ✓ Canonical per-camera vehicle identity continuity.
- ✓ Downstream track-quality gate.
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
- ✓ Common hard-braking false-positive suppression.
- ✓ Uninvolved-nearby-vehicle attribution regression coverage.
- ✓ Occlusion/merge incident pathway.
- ✓ Internal near-miss state used by crash reasoning.
- ○ Standalone hard-braking event stream.
- ○ Stopped vehicle in travel lane.
- ○ Wrong-way movement.
- ○ General trajectory anomaly / off-road movement.
- ○ Standalone near-miss alerts.
- ○ Pedestrian/cyclist incident models.

### Incident workflow

- ✓ Rolling pre/post incident video buffer.
- ✓ Synchronized telemetry CSV.
- ✓ Structured incident metadata.
- ✓ Video SHA-256.
- ✓ Durable incident database.
- ✓ Search and filtering.
- ✓ Approve / dismiss / needs-information review decisions.
- ✓ Reviewer identity, notes, classification corrections, and review history.
- ✓ Evidence ZIP export.
- ✓ In-app notification outbox.
- ✓ Optional webhook delivery adapter.
- ✓ Human-feedback calibration dataset and gated learning loop.
- ○ Signed multi-file evidence manifest.
- ○ Immutable storage and full access audit log.

### Traffic analytics

- ✓ Valid speed-observation persistence.
- ✓ Average and maximum speed summaries.
- ✓ Speeding-rate analytics.
- ✓ Dangerous day-of-week analysis.
- ✓ Dangerous hour-of-day analysis.
- ✓ Location/camera hotspot ranking.
- ✓ Optional map visualization from registered camera coordinates.
- ○ Lane-level counts and turn movements.
- ○ Flow, occupancy, headway, and congestion metrics.
- ○ Per-camera uptime and health analytics.

### Operator product

- ✓ Operations overview.
- ✓ Searchable review queue.
- ✓ Incident video and evidence inspection.
- ✓ Evidence library.
- ✓ Analytics workspace.
- ✓ Live monitor.
- ✓ Feedback model and notification settings.
- ○ Authenticated multi-user access.
- ○ Role-based permissions.
- ○ Multi-node deployment and managed database.
- ○ Centralized fleet health monitoring.

## Human feedback strategy

The system does not perform unrestricted online retraining from individual review clicks.

Instead:

1. A detector creates a probable incident and preserves evidence.
2. A reviewer approves, dismisses, or requests more information.
3. Approved/dismissed decisions become supervised labels.
4. A gated calibration model activates only after enough balanced labels exist.
5. The learned score changes review priority while the original detector score and evidence remain visible.
6. The labeled dataset can be exported for larger offline training and independent validation.

This approach captures the practical benefit people often mean when they say “RLHF” while preserving auditability and avoiding uncontrolled model drift.

## Deployment stages

| Stage | Scope | Goal |
|---|---|---|
| Engineering demo | Recorded and live single-camera feeds | Prove tracking, speed foundation, incident attribution, and capture |
| Paid pilot | 1–3 calibrated cameras | Measure false-alarm rate, recall, latency, review workload, and time saved |
| Operational pilot | Multiple cameras / one agency team | Validate search, notifications, analytics, review process, and evidence export |
| Production | Multi-camera managed service | Add authentication, RBAC, managed database, health monitoring, backups, and formal validation |

## Commercial value

The product should be measured by operational outcomes, not only detector accuracy:

- time from event occurrence to a reviewable record;
- staff minutes spent finding relevant footage;
- percentage of incident candidates reviewed quickly;
- evidence retrieval time;
- false-alert burden per camera-day;
- number of automatically preserved incidents that would otherwise require manual search;
- visibility into recurring speeding periods and locations.

The commercial product is therefore a per-camera or per-site software service covering:

- traffic perception;
- incident discovery and evidence preparation;
- human review and feedback;
- searchable history;
- notifications;
- roadway analytics;
- operational reporting.

That is the basis for a recurring managed-service price, not the novelty of a single computer-vision model.

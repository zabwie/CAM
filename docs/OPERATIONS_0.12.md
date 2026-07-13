# Operations platform — v0.12.0

Version 0.12.0 turns the perception/crash engine into an operational incident-intelligence workflow.

## Durable operations database

A local SQLite repository now stores:

- camera registry and optional coordinates;
- incident candidates and finalized evidence paths;
- detector and human-feedback review scores;
- approve, dismiss, and needs-information review history;
- sampled valid speed observations and posted speed limits;
- notification outbox records.

The database uses WAL mode and indexed search fields for a single-node paid pilot. A multi-user production deployment should move the same repository boundary to a managed database and add authenticated API services.

## Review workflow

The dashboard now provides a searchable review queue with:

- status, date-window, location, camera, and free-text filtering;
- incident video playback;
- detector evidence inspection;
- approve, dismiss, or needs-information decisions;
- reviewer identity, notes, and corrected classification;
- full review history;
- portable evidence ZIP export.

Evidence packages include a manifest, incident metadata, review history, video, telemetry, and checksum files when available.

## Human-feedback learning

Human review is treated as supervised feedback, not unconstrained online retraining.

- Approved and dismissed incidents become labeled examples.
- A regularized logistic calibration model learns recurring evidence patterns.
- The model activates only after at least 20 labeled incidents with at least 5 examples from each class.
- The learned probability is blended with the original detector score.
- The learned score changes review priority; raw evidence is never silently deleted or suppressed.
- Feedback can be exported as JSONL for future offline model training and validation.

This design prevents a single mistaken operator click from immediately changing live crash detection behavior.

## Analytics

The operations database can aggregate:

- incident counts and review status;
- average and maximum observed speed;
- speeding rate relative to the configured posted limit;
- day-of-week speeding patterns;
- hour-of-day speeding patterns;
- per-location/per-camera hotspot rankings;
- optional map visualization when camera coordinates are registered.

The hotspot `risk_index` is an operational ranking only. It is not a calibrated crash-probability estimate.

## Notifications

Incident candidates are persisted to an in-app notification outbox. Optional webhook destinations can also be queued and delivered with explicit sent/failed status. Delivery failure does not remove incident evidence.

## Dashboard workspaces

The Streamlit application now exposes:

1. **Overview** — pending review, confirmed incidents, speeding rate, maximum speed, and top hotspots.
2. **Review queue** — search, filter, inspect, classify, and export incidents.
3. **Analytics** — dangerous days, dangerous hours, speed trends, and hotspot rankings/maps.
4. **Live monitor** — current video, trusted tracks, calibration, and speed state.
5. **Evidence library** — archived incident clips and continuous recordings.
6. **Settings** — feedback model state, dataset export, notification configuration, and storage paths.

## Verification

- `pytest -q`: **45 passing tests**.
- Streamlit application test runner: **0 exceptions**.
- Self-contained crash stress suite: **PASS**.
- Synthetic hard-braking false alerts: **0/40** at each tested jitter level.
- Synthetic collision detection: **5–120 FPS** supported in the current regression suite.
- 30% observation-dropout collision trials: **40/40 detected**.

## Remaining production boundaries

This release is a strong paid-pilot foundation, not a claim of zero defects or emergency-grade certification. Production deployment still needs authenticated multi-user access, role-based permissions, managed secrets, centralized health monitoring, deployment automation, backups, and larger real-world validation datasets.

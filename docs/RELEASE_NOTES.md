# Release notes — v0.12.0

## Operations workflow

- Added a durable SQLite operations database for incidents, reviews, cameras, speed observations, and notification records.
- Added searchable incident review with approve, dismiss, and needs-information decisions, reviewer identity, notes, and corrected classifications.
- Added evidence ZIP export containing a manifest, review history, clip, telemetry, and checksum files when available.
- Added camera registry metadata with optional latitude/longitude and posted speed limits.
- Added day-of-week, hour-of-day, speeding-rate, and hotspot analytics.
- Added optional hotspot map visualization for geocoded cameras.
- Added in-app notification records plus an auditable webhook delivery adapter.
- Added idempotent import of existing recorder event packages into the operations database.

## Human-feedback learning

- Added a gated supervised feedback model trained from approved/dismissed incident reviews.
- Feedback calibration activates only after minimum dataset-size and class-balance requirements are met.
- Learned scores influence review priority while preserving the base detector output and all raw evidence.
- Added JSONL feedback dataset export for future offline training and validation.

## Dashboard

- Rebuilt the Streamlit UI around six operational workspaces: Overview, Review Queue, Analytics, Live Monitor, Evidence Library, and Settings.
- Added free-text, status, time-window, location, and camera filters.
- Added incident video review, evidence inspection, review history, and evidence-package download.
- Added `traffic-intel-dashboard` console launcher.

## Verification

- `pytest -q`: **45 passing tests**.
- Streamlit application test runner: **0 exceptions**.
- Existing self-contained crash stress suite remains **PASS**.

---

# Release notes — v0.11.0

## Crash hardening

- Added pair-relative motion validation and common-mode braking suppression to eliminate the reproduced hard-braking false-positive pathway.
- Added sparse-sampling support for 5–8 FPS feeds; synthetic collision regression now spans 5–120 FPS.
- Added broad-phase pair pruning and per-frame velocity caching for crowded-scene performance.
- Added safe optical-flow reset on source resolution changes.
- Added malformed-detection filtering, deterministic duplicate-ID handling, FPS validation, and monotonic frame checks.
- Added 17 hardening regressions and a self-contained adversarial stress runner.
- Made package-root imports lazy so lightweight modules do not require the full ML runtime.
- Hardened dashboard upload paths and aligned package version metadata.

## Verification

- `pytest -q`: **38 passing tests**.
- Self-contained stress suite: **PASS**; see `validation/latest_stress_results.json` and `docs/HARDENING_0.11.md`.

---

# Release notes — v0.10.0

## Canonical vehicle identity

- Added `CanonicalIdentityManager` between ByteTrack and downstream analytics.
- Raw ByteTrack IDs are now diagnostic handles; speed and incident history use stable canonical vehicle IDs.
- Added conservative short-gap re-identification using predicted motion, scale, class consistency, and appearance descriptors when source pixels are available.
- Added raw-ID hijack protection so a tracker integer cannot silently jump onto a different physical vehicle and inherit its history.
- Added provisional new-ID suppression, preventing a temporary one-frame identity from flickering into the UI or analytics before a re-identification decision is possible.
- Added filtered display boxes while preserving raw geometry for contact/impact evidence.
- Added static-track relocation and large-scale-change guards to stop foreground false detections from absorbing newly arriving real vehicles.

## Validation

- `pytest -q`: **19 passing tests**.
- Added identity regressions for raw ID 40→63 continuity, ambiguous re-entry, raw-ID hijack, quality continuity after a stitch, and static false-track rejection.
- Supplied-clip identity replay: 0 adjacent high-overlap canonical switches, 0 remaining canonical fragmentation candidates, and 0 duplicate canonical assignments in either clip.
- Crash regression remains one event per source:
  - `crash.mp4`: impact 123, detected 127, canonical participants 11 and 19.
  - `crash2.mp4`: impact/detection 238, canonical participants 13 and 14.

---

# Release notes — v0.9.0

## Architecture

- Extracted calibration from the engine.
- Added typed runtime configuration.
- Added shared trusted detection domain record.
- Extracted track-quality/reacquisition logic.
- Extracted scene-change detection.
- Added canonical `TrafficIncidentPipeline` coordinator.
- Moved crash visualization state out of crash semantics.
- Added package metadata and command-line entry points.

## Tracking

- Preserved class-agnostic vehicle NMS.
- Added geometry-instability and bounding-box size-change penalties.
- Added reacquisition cooldown after large jumps/gaps.
- Added class-consistency contribution to track quality.

## Speed

- Calibration now smooths both bottom-center coordinates before homography projection.
- Trajectory confidence now includes robust fit quality, not only history duration and density.
- Existing jump rejection, robust line fitting, track-gap reset, and acceleration limiting are preserved.

## Crash detection

- Preserved pair-only event attribution and current supplied-video timings.
- Converted temporal windows to seconds.
- Normalized kinematics to a 30-FPS-equivalent rate so tuned motion thresholds behave consistently across source FPS.
- Added 15/30/60 FPS regression coverage.

## Recording and runtime

- Fixed pre-event segment overwrite risk by using monotonic segment IDs.
- Recorder metadata now reflects the actual configured pre/post windows.
- Added unique event directory IDs.
- Fixed live speed alerts to inspect all current vehicles.
- Added a canonical pipeline path shared by live and validation adapters.
- Removed the calibration tool's hard-coded macOS Matplotlib backend.

## Quality

- `pytest -q`: 14 passing tests.
- Editable package install verified with `python -m pip install -e . --no-deps`.
- CLI help paths verified for live, calibration, and crash validation.
- Cached source-video regression preserved:
  - `crash.mp4`: impact 123, confirmation 129, one event.
  - `crash2.mp4`: impact 238, confirmation 238, one event.

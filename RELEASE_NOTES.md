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

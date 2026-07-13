# Merge / Occlusion Crash Hardening

This build adds a second crash-evidence path for collisions where YOLO briefly
sees a second vehicle but ByteTrack never promotes it to a stable trusted track.

The detector now preserves frame-local raw vehicle detections and maintains a
short-lived weak-object memory. A `merge_occlusion_impact` candidate requires:

- a mature surviving trusted vehicle track;
- a recent weak/raw vehicle observation that was spatially distinct;
- disappearance of that weak observation for multiple frames;
- abrupt area and width expansion of the surviving track;
- geometry showing that the enlarged surviving box explains the lost vehicle's
  last location;
- incident-level clustering to prevent duplicate alerts.

This pathway does not lower the global tracker threshold and therefore does not
turn every low-confidence night detection into a trusted vehicle identity.

## Regression verification

- Project test suite: 21 passed.
- Real replay: `2026-07-12 15-27-31.mp4`, YOLO11n, imgsz 640, no optical flow,
  first 90 frames: one `merge_occlusion_impact` event, trigger frame 58,
  detected frame 65, score approximately 0.808.

The machine-readable replay output is stored at:
`validation/rebuild_15-27-31.json`.

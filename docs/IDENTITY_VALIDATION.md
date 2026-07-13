# Canonical identity validation — v0.10.0

## Goal

Prevent downstream speed and incident logic from confusing a ByteTrack association integer with a durable physical-vehicle identity.

The validation overlays show:

- **large `ID N`** — canonical vehicle identity used by analytics;
- **small `raw M`** — underlying ByteTrack association ID;
- generation, identity confidence, and track-quality diagnostics;
- `RE-ID` banners when a new raw tracker ID is conservatively stitched back to an existing physical vehicle.

## Supplied-clip results

| Clip | Adjacent high-overlap canonical switches | Remaining canonical fragmentation candidates | Duplicate canonical assignments | Crash participants |
|---|---:|---:|---:|---|
| `crash.mp4` | 0 | 0 | 0 | ID 11 + ID 19 |
| `crash2.mp4` | 0 | 0 | 0 | ID 13 + ID 14 |

Crash timing remained stable:

- `crash.mp4`: impact frame 123, detection frame 127;
- `crash2.mp4`: impact/detection frame 238.

## Failure cases specifically tested

1. A physical vehicle tracked as raw ID 40 disappears briefly and returns as raw ID 63: the canonical ID remains unchanged.
2. A new observation is equally plausible for two dormant vehicles: the system creates a new identity rather than guessing.
3. The same raw tracker integer jumps to a distant/different vehicle: the old canonical history is broken instead of inherited.
4. A mature track is conservatively re-identified: quality maturity can be preserved without visible ID flicker.
5. A large near-static foreground false track disappears and a different moving vehicle appears nearby: position alone cannot transfer the old identity.

## Interpretation

These are regression results on the two supplied clips. They demonstrate that the canonical layer repairs the observed tracker fragmentation and avoids the known false-stitch case. They are not a general MOT benchmark and should not be presented as HOTA, IDF1, or universal ID-switch rates until a labeled multi-video identity dataset is built.

# Pilot-readiness architecture

This change set adds the downstream controls needed for a 90-day, two-camera
municipal pilot without moving reporting policy into the perception engine.

## Runtime flow

```text
Live camera / RTSP
    ↓
LatestFrameCapture
  • continuously reads the socket
  • retains the newest frame only
  • emits wall-clock and monotonic timestamps
    ↓
VisionQualityMonitor
  • DAY / NIGHT_IR / LOW_LIGHT / GLARE / DEGRADED
    ↓
TrafficIncidentPipeline
  • YOLO / ByteTrack / canonical identity
  • timestamp-based world-space speed
  • crash candidates
    ↓
VehiclePassageAggregator
  • one finalized record per canonical vehicle
  • median representative speed
  • normalized speeding denominator
    ↓
AnalyticsStore (SQLite)
  • vehicle_passages
  • camera_health
  • incidents
    ↓
traffic-intel-report
```

## Run a pilot camera

```bash
traffic-intel-live \
  --camera rtsp://camera/stream \
  --camera-id urban-01 \
  --municipality "Example Municipality" \
  --location-id "Main St / Central Ave" \
  --calibration calib.json \
  --speed-limit 35 \
  --analytics-db pilot-analytics.db
```

Run the second camera in a separate process with a different `--camera-id` and
the same SQLite database path only when both processes share a local filesystem
that safely supports SQLite locking. Otherwise use one database per camera and
merge/export after the pilot.

## Generate the report JSON

```bash
traffic-intel-report \
  --db pilot-analytics.db \
  --timezone America/Puerto_Rico \
  --output pilot-report.json
```

The report identifies the highest observed speeding rate only among monitored
pilot locations. It includes per-site average, median, 85th-percentile speed,
qualifying vehicle counts, speeding rates, and highest-rate weekday/hour.

## Measurement semantics

- A vehicle counts once, after its canonical track is finalized.
- Speed metrics use the median valid speed for that vehicle.
- A vehicle is included in the speeding denominator only when it has a valid
  representative speed and a configured speed limit.
- Day, IR night, low-light, glare, and degraded measurements remain separable.
- Live speed uses monotonic receipt time, not inference-loop frame count.
- This remains a pilot analytics system, not a legally certified speed device.

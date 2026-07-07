# Traffic Intelligence System

Sensor-fusion traffic intelligence: detection, tracking, speed estimation, and violation alerts. Built for city-district deployment.

## Architecture

```
Camera feed → YOLO11m@1280 → ByteTrack → ROI filter → speed estimator → annotated video + CSV
```

| Component | What it does |
|-----------|-------------|
| **YOLO11m** | Detects vehicles (car, truck, bus, motorcycle) at native resolution |
| **ByteTrack** | Assigns stable IDs to vehicles across frames, handles brief occlusions |
| **ROI filter** | Keeps detections inside the drivable road polygon; ignores sky, sidewalk, clutter |
| **Speed estimator** | Homography → interpolated reference → single reference → no speed. Only activates with calibration |
| **Streamlit dashboard** | Review annotated video, violation list, speed histogram, traffic stats |

## Quick start

```bash
# Process a video (tracking only — no calibration needed)
python3 -c "
from traffic_intel.engine import TrafficEngine
e = TrafficEngine()
print(e.process_video('videos/clear.mp4', output_path='output.mp4'))
"

# Launch the dashboard
streamlit run traffic_intel/app.py -- --video videos/clear.mp4
```

## Calibration wizard

Speed requires calibration. Run this on a reference frame:

```bash
python3 traffic_intel/calibrate.py --image ref.jpg --output calib.json
```

Four steps, all optional:

| Step | What you do | What you get |
|------|-------------|-------------|
| **1a. Near reference** | Click 2 points on a known-length object (e.g. 10ft lane marking) | Basic mph scale at that depth |
| **1b. Far reference** | Click same object further away | Perspective-correct mph at all depths |
| **2. Homography** | Click 4 road corners forming a rectangle, enter dimensions | Full perspective-correct speed |
| **3. ROI polygon** | Click points around the drivable area | Filters out non-road detections |
| **4. Speed trap** | Click 2 pairs of lines with known distance between them | Line-crossing event logging |

Speed priority: **homography → interpolated references → single reference → no speed**.

## Usage with calibration

```bash
# With calibration (enables speed)
python3 -c "
from traffic_intel.engine import TrafficEngine, Calibration
e = TrafficEngine(calibration=Calibration.load('calib.json'), speed_limit=50)
print(e.process_video('traffic.mp4', output_path='annotated.mp4'))
"

# Dashboard with calibration
streamlit run traffic_intel/app.py -- --video traffic.mp4 --calibration calib.json
```

## Output

- **Annotated video** — bounding boxes with track IDs and speed (when calibrated)
- **CSV log** — every detection: frame, track ID, class, confidence, bbox, speed, violation
- **Summary** — unique vehicles, total detections, violations, avg/max speed

## Deployment phases

| Phase | What | Hardware needed |
|-------|------|----------------|
| **1. PoC** | Software demo on recorded/streaming footage | Any camera or recorded video |
| **2. Paid pilot** | 1-3 intersections with real calibration | City-approved camera location |
| **3. Production** | Custom models, ANPR, radar fusion, evidence packages | Jetson AGX Orin, radar, ANPR camera |

## Project structure

```
traffic_intel/
├── engine.py       # Core: detection → tracking → speed → violations
├── calibrate.py    # Interactive calibration wizard (matplotlib)
├── app.py          # Streamlit dashboard
└── data/           # Output videos, CSVs, calibration files
```

## Calibration reference

| Object | Standard length (US) | Standard length (metric) |
|--------|---------------------|--------------------------|
| Dashed lane marking | 10 ft | 3.0 m |
| Lane width | 12 ft | 3.7 m |
| Crosswalk stripe | 10-12 ft | 3.0-3.7 m |
| Parking space | 18 ft | 5.5 m |

## Edge cases

- **No calibration** — tracking only, no speed displayed
- **Single reference only** — approximate mph, perspective distortion at different depths
- **Two references (near + far)** — interpolated scale, good perspective correction
- **Homography** — full perspective correction, requires 4 accurate clicks
- **ROI only** — filters detections, useful even without speed

## Pitch

> *"I can demonstrate the system using your existing camera feeds or a camera at one approved intersection. I don't need to permanently install equipment until the district approves a pilot."*

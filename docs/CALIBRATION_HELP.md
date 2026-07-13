# Calibration Instructions (for friend)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

## 1. Calibration Wizard

Run this on the provided sample video. Extract a reference frame first:

```bash
# Grab a reference frame from the sample video
python3 -c "
import cv2
cap = cv2.VideoCapture('sample.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)  # frame ~3 seconds in
ret, frame = cap.read()
if ret:
    cv2.imwrite('ref_frame.jpg', frame)
cap.release()
"

# Run the calibration wizard on it
python -m traffic_intel.calibrate --image ref_frame.jpg --output calib.json
```

### Calibration wizard controls:

| Key | Action |
|-----|--------|
| **Left-click** | Add point |
| **Right-click** | Remove last point |
| **h** | Compute homography (after 4 points) |
| **r** | Toggle ROI / Homography mode |
| **c** | Clear all points |
| **s** | Save to file |
| **q** | Quit |

### Steps for proper calibration:

1. Click **4 points on the road** forming a rectangle (lane segment):
   - Pt 1: near-left corner of a lane
   - Pt 2: near-right corner of same lane
   - Pt 3: far-left corner further down the road
   - Pt 4: far-right corner further down the road
2. Press **`h`** to compute homography — auto-assigns world coords (lane width 3.7m, segment length 25m)
3. Adjust with `--lane-width` and `--segment-length` flags if different:
   ```bash
   python -m traffic_intel.calibrate --image ref_frame.jpg --output calib.json --lane-width 3.7 --segment-length 25.0
   ```
4. Press **`r`** to switch to ROI mode, then click around the **drivable road area** to exclude fences/sidewalks
5. Press **`s`** to save

### ⚠️ Important:
- All 4 points MUST be on the **same ground plane** (road surface)
- Points on fences, sidewalks, or buildings will give wrong speed measurements
- The fence in the view should be excluded via ROI polygon

---

## 2. Run live on Iriun webcam (index 0)

```bash
# Tracking only (no speed — no calibration loaded)
python3 -m traffic_intel.live --camera 0

# With calibration (enables speed tracking)
python3 -m traffic_intel.live --camera 0 --calibration calib.json

# With speed limit alert
python3 -m traffic_intel.live --camera 0 --calibration calib.json --speed-limit 50

# Custom model (yolo11m.pt is more accurate but slower)
python3 -m traffic_intel.live --camera 0 --calibration calib.json --model yolo11m.pt

# Higher inference resolution for better detection
python3 -m traffic_intel.live --camera 0 --calibration calib.json --imgsz 1280

# Save the annotated feed to video
python3 -m traffic_intel.live --camera 0 --calibration calib.json --save-video output.mp4
```

Press **`q`** in the preview window to stop.

---

## 3. Run on a video file instead

```bash
python3 -c "
from traffic_intel.engine import TrafficEngine, Calibration
e = TrafficEngine(calibration=Calibration.load('calib.json'))
print(e.process_video('sample.mp4', output_path='annotated.mp4'))
"
```

---

## 4. Camera index notes

- **Camera index 0** = Iriun phone camera (USB/web)
- **Camera index 1** = MacBook built-in camera
- For **RTSP/IP cameras**: `--camera rtsp://192.168.1.100:554/stream1`

---

## Files included

| File | Purpose |
|------|---------|
| `CALIBRATION_HELP.md` | This file |
| `validation/videos/` | Supplied crash-regression source clips |
| `calib.json` | (to be replaced after calibration) |
| `yolo11n.pt` | YOLO model (nano — fast but less accurate) |
| `traffic_intel/` | Core engine code |

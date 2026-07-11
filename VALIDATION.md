# Real-video validation

Validated against the supplied `clear.mp4` footage using the included
`results.csv` detections/tracks (1,660 video frames at 30 FPS).

## Results

- Full video replay completed: 1,660 / 1,660 frames.
- Detection rows replayed: 26,600.
- Tracks with calibrated, valid speed: 31.
- Valid speed samples: 1,564.
- Average valid displayed speed: 44.4 mph.
- Maximum valid displayed speed: 55.7 mph.
- The original speed column ranged from 2.9 mph to 169.6 mph; the revised
  trajectory estimator and calibration guard remove those extreme spikes.
- Regression suite: 8 tests passed.

## Reproduce

```bash
pytest -q

python3 -m traffic_intel.replay \
  --video clear.mp4 \
  --detections results.csv \
  --calibration calib.json \
  --output annotated_stable.mp4 \
  --output-csv stable_results.csv
```

The YOLO model weights are not bundled. The replay command is intentionally
provided so already-exported real detections can be re-tested and rendered
without rerunning or downloading the detector model.

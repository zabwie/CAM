"""Regression test using the supplied real detection log and calibration."""

from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "traffic_intel"))
from engine import Calibration
from speed import RobustSpeedEstimator


def test_real_log_has_no_implausible_speed_spikes():
    df = pd.read_csv(ROOT / "results.csv")
    calibration = Calibration.load(ROOT / "calib.json")
    H = calibration.H
    estimator = RobustSpeedEstimator(fps=30.0)
    values = []

    for frame, rows in df.groupby("frame", sort=True):
        for row in rows.itertuples(index=False):
            cx = (row.x1 + row.x2) / 2.0
            p = np.array([[[cx, float(row.y2)]]], dtype=np.float32)
            wx, wy = map(float, cv2.perspectiveTransform(p, H)[0, 0])
            if calibration.world_point_is_calibrated(wx, wy):
                mph = estimator.update(int(row.track_id), int(frame), wx, wy)
                if mph is not None:
                    values.append(mph)
        estimator.forget_stale(int(frame))

    assert len(values) > 1000
    assert max(values) < 70.0
    assert np.percentile(values, 1) > 25.0

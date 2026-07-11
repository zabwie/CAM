from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "traffic_intel"))
from engine import TrafficEngine


def test_vehicle_label_contains_classification_and_own_speed(monkeypatch):
    frame = np.zeros((200, 500, 3), dtype=np.uint8)
    drawn_text = []
    real_put_text = cv2.putText

    def capture_put_text(img, text, *args, **kwargs):
        drawn_text.append(text)
        return real_put_text(img, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_put_text)
    TrafficEngine._draw_vehicle_label(
        frame,
        x1=30,
        y1=100,
        track_id=12,
        class_name="car",
        speed_mph=47.4,
    )

    assert drawn_text == ["car | 47 MPH | #12"]
    assert np.count_nonzero(frame) > 0


def test_vehicle_label_shows_locking_state_without_fake_speed(monkeypatch):
    frame = np.zeros((200, 500, 3), dtype=np.uint8)
    drawn_text = []
    real_put_text = cv2.putText

    def capture_put_text(img, text, *args, **kwargs):
        drawn_text.append(text)
        return real_put_text(img, text, *args, **kwargs)

    monkeypatch.setattr(cv2, "putText", capture_put_text)
    TrafficEngine._draw_vehicle_label(
        frame,
        x1=30,
        y1=100,
        track_id=8,
        class_name="truck",
        speed_mph=None,
    )

    assert drawn_text == ["truck | -- MPH | #8"]

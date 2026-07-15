"""Crash annotation state and drawing helpers.

Visualization is intentionally outside the crash detector so event semantics do
not depend on UI persistence.  The module-level registry remains for backward
compatibility with the existing functional API and is explicitly reset on scene
changes and new pipeline instances.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

CRASH_VISUAL_TIMEOUT = 150
_crash_vehicles: dict[int, tuple[int, int, int, int, int]] = {}


def reset_crash_visuals() -> None:
    _crash_vehicles.clear()


def mark_crash_vehicles(
    involved: list[int],
    all_detections: list,
    frame_count: int,
    fallback_bboxes: Optional[dict[int, tuple[float, float, float, float]]] = None,
) -> None:
    fallback_bboxes = fallback_bboxes or {}
    for track_id in involved:
        detections = [
            d for d in all_detections
            if d.track_id == track_id and d.frame <= frame_count
        ]
        bbox = detections[-1].bbox if detections else fallback_bboxes.get(track_id)
        if bbox is not None:
            _crash_vehicles[track_id] = (
                frame_count,
                int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]),
            )


def update_crash_visuals(
    frame: np.ndarray, all_detections: list, frame_count: int
) -> None:
    expired: list[int] = []
    for track_id, (last_frame, x1, y1, x2, y2) in list(_crash_vehicles.items()):
        if frame_count - last_frame > CRASH_VISUAL_TIMEOUT:
            expired.append(track_id)
            continue
        current = [
            d for d in all_detections
            if d.frame == frame_count and d.track_id == track_id
        ]
        if current:
            x1, y1, x2, y2 = map(int, current[-1].bbox)
            _crash_vehicles[track_id] = (frame_count, x1, y1, x2, y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        label = f"CRASH #{track_id}"
        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        top = max(0, y1 - text_h - 8)
        cv2.rectangle(frame, (x1, top), (x1 + text_w + 10, y1), (0, 0, 255), -1)
        cv2.putText(
            frame,
            label,
            (x1 + 5, max(text_h + 1, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    for track_id in expired:
        _crash_vehicles.pop(track_id, None)


def draw_crash_boxes(
    frame: np.ndarray,
    candidate: Any,
    all_detections: list,
    frame_count: int,
) -> None:
    mark_crash_vehicles(
        candidate.involved_tracks,
        all_detections,
        frame_count,
        fallback_bboxes=candidate.involved_bboxes,
    )
    update_crash_visuals(frame, all_detections, frame_count)

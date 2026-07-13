"""Core perception pipeline: YOLO -> NMS -> ByteTrack -> trusted tracks -> speed.

The engine deliberately stops at trusted per-frame vehicle observations.  Crash
and other incident analyzers consume those observations independently so the
perception stack can evolve without embedding incident policy in the tracker.
"""

from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import supervision as sv
from supervision.tracker.byte_tracker.core import ByteTrack
from ultralytics import YOLO

from traffic_intel.config import EngineConfig, SceneChangeConfig, TrackingConfig
from traffic_intel.domain import Detection, RawVehicleDetection
from traffic_intel.core.identity import CanonicalIdentityManager, RawTrackObservation
from traffic_intel.core.tracking import TrackQualityGate
from traffic_intel.core.scene import SceneChangeDetector
from traffic_intel.motion.calibration import Calibration
from traffic_intel.motion.speed import RobustSpeedEstimator

VEHICLE_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck (COCO)


class TrafficEngine:
    """Convert frames into stable, analytics-ready vehicle observations.

    The constructor keeps the original keyword arguments for compatibility,
    while ``config=EngineConfig(...)`` is the preferred API for new code.
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        calibration: Optional[Calibration] = None,
        confidence: Optional[float] = None,
        fps: float = 30.0,
        imgsz: int = 1280,
        track_activation_threshold: Optional[float] = None,
        track_min_hits: Optional[int] = None,
        track_min_confidence: Optional[float] = None,
        vehicle_nms_threshold: Optional[float] = None,
        scene_cut_reset: Optional[bool] = None,
        retain_history: bool = True,
        config: Optional[EngineConfig] = None,
    ) -> None:
        base = config or EngineConfig(
            model_path=model_path,
            imgsz=imgsz,
            fps=fps,
            retain_history=retain_history,
        )
        tracking = base.tracking
        tracking = replace(
            tracking,
            detector_confidence=(
                tracking.detector_confidence if confidence is None else float(confidence)
            ),
            activation_threshold=(
                tracking.activation_threshold
                if track_activation_threshold is None
                else float(track_activation_threshold)
            ),
            min_hits=(tracking.min_hits if track_min_hits is None else int(track_min_hits)),
            min_confidence=(
                tracking.min_confidence
                if track_min_confidence is None
                else float(track_min_confidence)
            ),
            vehicle_nms_threshold=(
                tracking.vehicle_nms_threshold
                if vehicle_nms_threshold is None
                else float(vehicle_nms_threshold)
            ),
        )
        scene = base.scene_change
        if scene_cut_reset is not None:
            scene = replace(scene, enabled=bool(scene_cut_reset))
        self.config = replace(
            base,
            model_path=model_path if config is None else base.model_path,
            imgsz=imgsz if config is None else base.imgsz,
            fps=fps if config is None else base.fps,
            retain_history=retain_history if config is None else base.retain_history,
            tracking=tracking,
            scene_change=scene,
        )

        resolved_model = self._resolve_model(self.config.model_path)
        print(f"  Model: {resolved_model}")
        self.model = YOLO(resolved_model)
        self.calibration = calibration
        self.fps = float(self.config.fps)
        self.imgsz = int(self.config.imgsz)
        self.retain_history = bool(self.config.retain_history)

        self.tracker = self._build_tracker(self.fps)
        self.identity_manager = CanonicalIdentityManager(self.fps, tracking)
        self.track_gate = TrackQualityGate(self.fps, tracking)
        self.scene_detector = SceneChangeDetector(scene)
        self.speed_estimator = RobustSpeedEstimator(self.fps, self.config.speed)

        self.results: list[Detection] = []
        self.current_detections: list[Detection] = []
        self.current_raw_detections: list[RawVehicleDetection] = []
        self.frame_count = 0
        self.last_scene_cut = False
        self.scene_cut_reset = bool(self.config.scene_change.enabled)
        self.scene_warmup_remaining = self._warmup_frames()

    # ------------------------------------------------------------------
    # Public processing API
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty BGR image")

        self.frame_count += 1
        self.current_detections = []
        self.current_raw_detections = []
        if not self.retain_history:
            self.results.clear()

        self.last_scene_cut = self._detect_scene_cut(frame)
        if self.last_scene_cut and self.scene_cut_reset:
            self._reset_temporal_state(preserve_scene_reference=True)

        yolo_out = self.model(
            frame,
            verbose=False,
            conf=self.config.tracking.detector_confidence,
            imgsz=self.imgsz,
        )[0]
        dets = sv.Detections.from_ultralytics(yolo_out)
        if len(dets):
            dets = dets[np.isin(dets.class_id, list(VEHICLE_IDS))]
        if len(dets):
            dets = dets.with_nms(
                threshold=self.config.tracking.vehicle_nms_threshold,
                class_agnostic=True,
            )
        dets = self._filter_roi(dets)

        # Preserve detector-level evidence before ByteTrack and trust gating.
        # This is intentionally lightweight and frame-local; the incident
        # detector decides whether weak observations are meaningful.
        for i in range(len(dets)):
            class_id = int(dets.class_id[i])
            self.current_raw_detections.append(
                RawVehicleDetection(
                    frame=self.frame_count,
                    class_name=str(yolo_out.names.get(class_id, "vehicle")),
                    confidence=float(dets.confidence[i]),
                    bbox=tuple(map(int, np.round(dets.xyxy[i]))),
                )
            )

        tracks = self.tracker.update_with_detections(dets)

        annotated = frame.copy()
        self._draw_roi(annotated)
        warming_up = self.scene_warmup_remaining > 0
        self.scene_warmup_remaining = max(0, self.scene_warmup_remaining - 1)

        raw_observations: list[RawTrackObservation] = []
        for i in range(len(tracks)):
            if tracks.tracker_id[i] is None:
                continue
            raw_observations.append(
                RawTrackObservation(
                    tracker_id=int(tracks.tracker_id[i]),
                    class_id=int(tracks.class_id[i]),
                    confidence=float(tracks.confidence[i]),
                    bbox=np.asarray(tracks.xyxy[i], dtype=np.float32),
                )
            )

        identities = self.identity_manager.assign_batch(
            frame=self.frame_count,
            observations=raw_observations,
            image=frame,
        )
        for raw, identity in zip(raw_observations, identities):
            assessment = self.track_gate.update(
                frame=self.frame_count,
                track_id=identity.canonical_id,
                class_id=raw.class_id,
                confidence=raw.confidence,
                bbox=identity.raw_bbox,
                reidentified=identity.lifecycle in {"STITCHED", "REACQUIRED"},
                identity_confidence=identity.identity_confidence,
            )
            if warming_up or identity.provisional or not assessment.confirmed:
                continue

            class_name = str(yolo_out.names.get(assessment.stable_class_id, "vehicle"))
            raw_bbox = tuple(map(int, np.round(identity.raw_bbox)))
            filtered_bbox = tuple(map(int, np.round(identity.filtered_bbox)))
            detection = self._build_detection(
                track_id=identity.canonical_id,
                raw_track_id=identity.tracker_id,
                identity_generation=identity.generation,
                identity_confidence=identity.identity_confidence,
                identity_lifecycle=identity.lifecycle,
                class_name=class_name,
                detector_confidence=raw.confidence,
                track_quality=assessment.quality,
                bbox=raw_bbox,
                filtered_bbox=filtered_bbox,
            )
            self.results.append(detection)
            self.current_detections.append(detection)
            self._draw_detection(annotated, detection)

        self.speed_estimator.forget_stale(self.frame_count)
        self.track_gate.forget_stale(self.frame_count)
        self.identity_manager.forget_stale(self.frame_count)
        return annotated

    def process_video(
        self,
        video_path: str | Path,
        output_path: Optional[str | Path] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or self.fps)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._set_fps(fps)
        self.reset()

        writer: Optional[cv2.VideoWriter] = None
        if output_path:
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
            )
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Cannot open output video: {output_path}")

        count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated = self.process_frame(frame)
                if writer is not None:
                    writer.write(annotated)
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
                if progress_callback is not None and count % 30 == 0:
                    progress_callback(count, total)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        return self.summary()

    def reset(self) -> None:
        """Reset all temporal state before a new source or independent replay."""
        self.results.clear()
        self.current_detections.clear()
        self.current_raw_detections.clear()
        self.frame_count = 0
        self.last_scene_cut = False
        self.tracker = self._build_tracker(self.fps)
        self.identity_manager.reset(reset_counter=True)
        self.track_gate.reset()
        self.speed_estimator.reset()
        self.scene_detector.reset()
        self.scene_warmup_remaining = self._warmup_frames()
        if self.calibration is not None:
            self.calibration.reset_track_state()

    def results_csv(self, path: str | Path) -> None:
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "frame", "track_id", "raw_track_id", "identity_generation",
                "identity_confidence", "identity_lifecycle", "class", "confidence",
                "x1", "y1", "x2", "y2", "speed_mph", "speed_valid",
                "invalid_reason", "capture_ts", "meas_confidence",
                "cal_confidence", "traj_confidence", "vis_confidence",
                "zone_confidence", "track_quality",
            ])
            for d in self.results:
                writer.writerow([
                    d.frame, d.track_id, d.raw_track_id, d.identity_generation,
                    f"{d.identity_confidence:.3f}", d.identity_lifecycle,
                    d.class_name, f"{d.confidence:.3f}",
                    *d.bbox, f"{d.speed:.1f}", int(d.speed_valid), d.invalid_reason,
                    f"{d.capture_timestamp:.3f}", f"{d.measurement_confidence:.3f}",
                    f"{d.cal_confidence:.3f}", f"{d.traj_confidence:.3f}",
                    f"{d.vis_confidence:.3f}", f"{d.zone_confidence:.3f}",
                    f"{d.track_quality:.3f}",
                ])

    def summary(self) -> dict:
        tracks = {d.track_id for d in self.results}
        valid_speeds = [d.speed for d in self.results if d.speed_valid]
        return {
            "frames_processed": self.frame_count,
            "unique_vehicles": len(tracks),
            "total_detections": len(self.results),
            "valid_speed_readings": len(valid_speeds),
            "average_speed_mph": (
                round(float(np.mean(valid_speeds)), 2) if valid_speeds else None
            ),
        }



    # ------------------------------------------------------------------
    # Per-frame helpers
    # ------------------------------------------------------------------

    def _build_detection(
        self,
        *,
        track_id: int,
        raw_track_id: int,
        identity_generation: int,
        identity_confidence: float,
        identity_lifecycle: str,
        class_name: str,
        detector_confidence: float,
        track_quality: float,
        bbox: tuple[int, int, int, int],
        filtered_bbox: tuple[int, int, int, int],
    ) -> Detection:
        x1, y1, x2, y2 = filtered_bbox
        bbox_h = max(0, y2 - y1)
        speed: Optional[float] = None
        reason = "NO_CALIBRATION"
        in_calibrated_zone = False

        if bbox_h < self.config.min_bbox_height_for_speed:
            reason = "TOO_FAR"
        elif self.calibration is not None:
            cx, cy = (x1 + x2) / 2.0, float(y2)
            world = self.calibration.smoothed_world_from_image(track_id, cx, cy)
            if world is None:
                reason = "OUTSIDE_ZONE"
            else:
                in_calibrated_zone = True
                speed = self.speed_estimator.update(
                    track_id, self.frame_count, world[0], world[1]
                )
                reason = (
                    "VALID" if speed is not None else self.speed_estimator.last_reason(track_id)
                )

        confidence = self._measurement_confidence(
            track_id=track_id,
            bbox_height=bbox_h,
            in_calibrated_zone=in_calibrated_zone,
        )
        return Detection(
            frame=self.frame_count,
            track_id=track_id,
            class_name=class_name,
            confidence=float(detector_confidence),
            bbox=bbox,
            speed=float(speed) if speed is not None else 0.0,
            speed_valid=speed is not None,
            invalid_reason=reason,
            measurement_confidence=confidence["overall"],
            cal_confidence=confidence["calibration"],
            traj_confidence=confidence["trajectory"],
            vis_confidence=confidence["visibility"],
            zone_confidence=confidence["zone"],
            capture_timestamp=self.frame_count / self.fps,
            track_quality=float(track_quality),
            track_confirmed=True,
            raw_track_id=int(raw_track_id),
            identity_generation=int(identity_generation),
            identity_confidence=float(identity_confidence),
            identity_lifecycle=str(identity_lifecycle),
            filtered_bbox=filtered_bbox,
        )

    def _filter_roi(self, dets: sv.Detections) -> sv.Detections:
        roi = self.calibration.roi_polygon if self.calibration else None
        if not roi or not len(dets):
            return dets
        polygon = np.asarray(roi, dtype=np.int32)
        keep: list[int] = []
        for i, box in enumerate(dets.xyxy):
            x1, _, x2, y2 = map(float, box)
            anchor = ((x1 + x2) * 0.5, y2)
            if cv2.pointPolygonTest(polygon, anchor, False) >= 0:
                keep.append(i)
        return dets[keep] if keep else dets[:0]

    def _draw_roi(self, frame: np.ndarray) -> None:
        roi = self.calibration.roi_polygon if self.calibration else None
        if roi:
            cv2.polylines(
                frame, [np.asarray(roi, dtype=np.int32)], True, (0, 255, 255), 2
            )

    @staticmethod
    def _draw_detection(frame: np.ndarray, detection: Detection) -> None:
        x1, y1, x2, y2 = detection.display_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        speed_text = f"{detection.speed:.0f} MPH" if detection.speed_valid else "-- MPH"
        text = (
            f"{detection.class_name} | {speed_text} | "
            f"Q{detection.track_quality * 100:.0f}% | ID {detection.track_id}"
        )
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.52, 2
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        frame_h, frame_w = frame.shape[:2]
        left = max(0, min(x1, max(0, frame_w - text_w - 10)))
        text_y = max(text_h + baseline + 6, y1 - 8)
        top = max(0, text_y - text_h - baseline - 6)
        right = min(frame_w - 1, left + text_w + 10)
        bottom = min(frame_h - 1, text_y + baseline + 3)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (left + 5, text_y), font, scale,
            (0, 255, 0), thickness, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # Temporal state and confidence
    # ------------------------------------------------------------------

    def _detect_scene_cut(self, frame: np.ndarray) -> bool:
        return self.scene_detector.update(frame)

    def _reset_temporal_state(self, preserve_scene_reference: bool = False) -> None:
        self.tracker = self._build_tracker(self.fps)
        self.identity_manager.reset(reset_counter=False)
        self.track_gate.reset()
        self.speed_estimator.reset()
        if not preserve_scene_reference:
            self.scene_detector.reset()
        self.scene_warmup_remaining = self._warmup_frames()
        if self.calibration is not None:
            self.calibration.reset_track_state()

    def _set_fps(self, fps: float) -> None:
        self.fps = float(fps or 30.0)
        self.identity_manager.set_fps(self.fps)
        self.track_gate.set_fps(self.fps)
        self.speed_estimator.set_fps(self.fps)
        self.tracker = self._build_tracker(self.fps)
        self.scene_warmup_remaining = self._warmup_frames()

    def _build_tracker(self, fps: float) -> ByteTrack:
        cfg = self.config.tracking
        return ByteTrack(
            track_activation_threshold=cfg.activation_threshold,
            lost_track_buffer=max(1, int(round(cfg.lost_track_seconds * fps))),
            minimum_matching_threshold=cfg.minimum_matching_threshold,
            frame_rate=max(1, int(round(fps))),
        )

    def _warmup_frames(self) -> int:
        return max(3, int(round(self.config.scene_change.warmup_seconds * self.fps)))

    def _calibration_confidence(self) -> float:
        if not self.calibration or not self.calibration.calibration_quality:
            return 0.5
        quality = self.calibration.calibration_quality
        mean_res = quality.get("mean_reprojection_residual_m")
        if mean_res is None:
            mean_res = quality.get("mean_reprojection_error_m")
        if mean_res is None:
            return 0.5
        return float(np.clip(1.0 - float(mean_res) / 0.5, 0.0, 1.0))

    def _measurement_confidence(
        self, *, track_id: int, bbox_height: int, in_calibrated_zone: bool
    ) -> dict[str, float]:
        cal = self._calibration_confidence()
        traj = self.speed_estimator.trajectory_confidence(track_id, self.frame_count)
        visibility = min(max(bbox_height, 0) / 50.0, 1.0)
        zone = 1.0 if in_calibrated_zone else 0.3
        overall = float(np.clip(
            0.35 * cal + 0.40 * traj + 0.15 * visibility + 0.10 * zone,
            0.0, 1.0,
        ))
        return {
            "overall": overall,
            "calibration": cal,
            "trajectory": traj,
            "visibility": visibility,
            "zone": zone,
        }

    @staticmethod
    def _resolve_model(model_path: str) -> str:
        coreml = model_path.replace(".pt", ".mlpackage")
        return coreml if Path(coreml).exists() else model_path


__all__ = ["Calibration", "Detection", "EngineConfig", "TrafficEngine"]

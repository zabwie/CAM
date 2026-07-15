"""Runtime state and data processing for the operations dashboard."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np

# These will be updated once we finalise locations:
from traffic_intel.core.engine import TrafficEngine
from traffic_intel.core.pipeline import TrafficIncidentPipeline
from traffic_intel.recording.event_recorder import EventRecorderConfig, RollingSegmentBuffer
from traffic_intel.recording.archive import ArchiveRecorderConfig, ContinuousArchiveRecorder
from traffic_intel.motion.calibration import Calibration
from traffic_intel.ops.store import IncidentStore


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--camera", default="0", help="Índice de cámara o URL RTSP")
    parser.add_argument("--video", default=None, help="Video de prueba opcional")
    parser.add_argument("--calibration", default=None, help="JSON de calibración")
    parser.add_argument("--model", default="models/yolo11n.pt", help="Modelo YOLO")
    parser.add_argument("--imgsz", type=int, default=1280, help="Resolución de inferencia")
    parser.add_argument("--event-dir", default="events", help="Carpeta de incidentes")
    parser.add_argument("--archive-dir", default="archive", help="Carpeta de grabaciones continuas")
    parser.add_argument("--archive-segment-minutes", type=float, default=5.0)
    parser.add_argument("--retention-days", type=int, default=30)
    parser.add_argument("--pre-event-seconds", type=float, default=20.0)
    parser.add_argument("--post-event-seconds", type=float, default=10.0)
    parser.add_argument("--db-path", default="data/traffic_intel.db", help="Base de datos operativa SQLite")
    parser.add_argument("--speed-limit-mph", type=float, default=35.0, help="Límite de velocidad para analítica")
    args, _ = parser.parse_known_args()
    return args


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class LiveRuntime:
    cap: cv2.VideoCapture
    engine: TrafficEngine
    pipeline: TrafficIncidentPipeline
    recorder: RollingSegmentBuffer
    archive: ContinuousArchiveRecorder | None
    source_label: str
    source_value: str | int
    camera_fps: float
    store: IncidentStore
    started_at: float = field(default_factory=time.time)
    seen_track_ids: set[int] = field(default_factory=set)
    analysis_fps_samples: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    last_saved_event: str | None = None
    last_speed_sample_by_track: dict[int, float] = field(default_factory=dict)

    def close(self) -> None:
        try:
            self.cap.release()
        finally:
            try:
                self.recorder.close()
            finally:
                if self.archive is not None:
                    self.archive.close()


@dataclass
class IncidentRecord:
    key: str
    timestamp: float
    status: str
    title: str
    camera: str
    tracks: str = "—"
    score: float | None = None
    path: str | None = None
    description: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CLASS_TRANSLATIONS = {
    "car": "Car", "truck": "Truck", "bus": "Bus",
    "motorcycle": "Motorcycle", "bicycle": "Bicycle", "vehicle": "Vehicle",
}


def translate_class(name: str) -> str:
    return CLASS_TRANSLATIONS.get(str(name).lower(), str(name).replace("_", " ").title())


def safe_source_label(source: str | int) -> str:
    if isinstance(source, int):
        return f"Cámara local {source}"
    value = str(source)
    if value.lower().startswith(("rtsp://", "http://", "https://")):
        parts = urlsplit(value)
        host = parts.hostname or "cámara de red"
        port = f":{parts.port}" if parts.port else ""
        path = parts.path if parts.path and parts.path != "/" else ""
        return urlunsplit((parts.scheme, f"{host}{port}", path, "", ""))
    return Path(value).name or value


def resolve_source(session_state) -> str | int:
    if session_state.source_mode in {"Video de prueba", "Test video"}:
        path = session_state.uploaded_demo_path
        if not path:
            raise ValueError("Selecciona un video de prueba antes de iniciar.")
        return str(path)
    raw = str(session_state.source_value).strip()
    if not raw:
        raise ValueError("Especifica una cámara, índice o URL RTSP.")
    return int(raw) if raw.isdigit() else raw


def load_calibration(path_value: str) -> Calibration | None:
    path_value = str(path_value or "").strip()
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró la calibración: {path}")
    return Calibration.load(path)


def calibration_status(runtime: LiveRuntime | None) -> tuple[str, str]:
    if runtime is None or runtime.engine.calibration is None:
        return "Sin calibrar", "La velocidad no se mostrará como válida"
    grade = runtime.engine.calibration.quality_grade or "Disponible"
    return str(grade), "Calibración cargada"


def start_runtime(session_state, args: argparse.Namespace) -> LiveRuntime:
    source = resolve_source(session_state)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {safe_source_label(source)}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    calibration = load_calibration(session_state.calibration_path)
    engine = TrafficEngine(
        model_path=session_state.model_path,
        calibration=calibration,
        fps=fps,
        imgsz=int(session_state.imgsz),
        retain_history=False,
    )
    pipeline = TrafficIncidentPipeline(engine)
    source_label = safe_source_label(source)
    camera_name = str(session_state.camera_name or source_label).strip()
    recorder = RollingSegmentBuffer(
        fps=fps,
        output_dir=session_state.event_dir,
        config=EventRecorderConfig(
            pre_event_seconds=float(args.pre_event_seconds),
            post_event_seconds=float(args.post_event_seconds),
            camera_id=camera_name,
        ),
    )
    archive = None
    if bool(session_state.archive_enabled):
        archive = ContinuousArchiveRecorder(
            fps=fps,
            config=ArchiveRecorderConfig(
                output_dir=str(session_state.archive_dir),
                segment_seconds=float(session_state.archive_segment_minutes) * 60.0,
                retention_days=int(session_state.retention_days),
                municipality=str(session_state.municipality).strip(),
                location_name=str(session_state.location_name).strip(),
                camera_name=camera_name,
                source_label=source_label,
            ),
        )
    store = IncidentStore(str(session_state.db_path or args.db_path))
    municipality = str(session_state.municipality).strip()
    location_name = str(session_state.location_name).strip()
    store.upsert_camera(
        camera_id=f"{municipality}::{location_name}::{camera_name}",
        camera_name=camera_name,
        municipality=municipality,
        location=location_name,
        source=source_label,
        latitude=session_state.camera_latitude,
        longitude=session_state.camera_longitude,
        speed_limit_mph=float(session_state.speed_limit_mph),
        timezone_name=str(session_state.agency_timezone),
    )
    return LiveRuntime(
        cap=cap, engine=engine, pipeline=pipeline,
        recorder=recorder, archive=archive,
        source_label=source_label, source_value=source, camera_fps=fps, store=store,
    )


def stop_runtime(session_state) -> None:
    runtime = session_state.runtime
    if runtime is not None:
        try:
            runtime.close()
        except Exception:
            pass
    session_state.runtime = None
    session_state.monitoring = False


def incident_exists(session_state, key: str) -> bool:
    return any(item.get("key") == key for item in session_state.incidents)


def add_incident(session_state, record: IncidentRecord) -> None:
    if incident_exists(session_state, record.key):
        return
    payload = {
        "key": record.key, "timestamp": record.timestamp,
        "status": record.status, "title": record.title,
        "camera": record.camera, "tracks": record.tracks,
        "score": record.score, "path": record.path,
        "description": record.description,
    }
    session_state.incidents = ([payload] + session_state.incidents)[:100]


def register_saved_event(session_state, runtime: LiveRuntime, event_path: Path) -> None:
    event_json = event_path / "event.json"
    metadata: dict = {}
    if event_json.exists():
        try:
            metadata = json.loads(event_json.read_text())
        except (OSError, json.JSONDecodeError):
            metadata = {}
    event_id = str(metadata.get("event_id") or event_path.name)
    tracks = metadata.get("involved_tracks") or []
    score = metadata.get("score")
    trigger_type = str(metadata.get("trigger_type", "evento"))
    automatic = trigger_type.startswith("auto_")
    title = "Choque detectado" if automatic else "Evento guardado manualmente"
    tracks_text = ", ".join(f"#{t}" for t in tracks) if tracks else "—"
    score_value = float(score) if isinstance(score, (int, float)) else None
    description = str(metadata.get("description") or "Clip y telemetría disponibles.")

    # Persist the finalized evidence package in the operational database.
    runtime.store.upsert_incident({
        **metadata,
        "event_id": event_id,
        "detected_at": float(metadata.get("trigger_time_unix") or time.time()),
        "title": title,
        "detector_score": score_value,
        "clip_path": str(event_path / "clip.mp4"),
        "metadata_path": str(event_json),
        "status": "pending",
    })

    for item in session_state.incidents:
        same_auto_tracks = automatic and item.get("status") == "Capturando" and item.get("tracks") == tracks_text
        same_manual = (
            not automatic and item.get("status") == "Capturando"
            and "manual" in str(item.get("title", "")).lower()
        )
        if same_auto_tracks or same_manual:
            item.update({
                "key": f"saved:{event_id}", "status": "Guardado",
                "title": title, "tracks": tracks_text,
                "score": score_value, "path": str(event_path),
                "description": description,
            })
            return
    add_incident(session_state, IncidentRecord(
        key=f"saved:{event_id}",
        timestamp=float(metadata.get("trigger_time_unix") or time.time()),
        status="Guardado", title=title, camera=runtime.source_label,
        tracks=tracks_text, score=score_value, path=str(event_path),
        description=description,
    ))


def current_vehicle_rows(detections: list) -> list[dict]:
    rows = []
    for d in sorted(detections, key=lambda item: item.track_id):
        rows.append({
            "ID": f"#{d.track_id}",
            "Type": translate_class(d.class_name),
            "Speed": f"{d.speed:.1f} mph" if d.speed_valid else "—",
            "Confidence": f"{max(0.0, min(1.0, d.measurement_confidence)):.0%}"
            if d.measurement_confidence > 0 else "—",
        })
    return rows


def process_live_frame(session_state, runtime: LiveRuntime, args: argparse.Namespace) -> None:
    started = time.perf_counter()
    ok, frame = runtime.cap.read()
    if not ok or frame is None:
        raise RuntimeError("La cámara dejó de entregar video.")

    if runtime.archive is not None:
        runtime.archive.write_frame(frame, frame_number=runtime.engine.frame_count + 1, timestamp=time.time())

    result = runtime.pipeline.process_frame(frame, optical_flow=True)
    detections = result.detections
    runtime.seen_track_ids.update(d.track_id for d in detections)

    runtime.recorder.write_frame(result.annotated, detections, runtime.engine.frame_count)

    now = time.time()
    camera_name = str(session_state.camera_name or runtime.source_label).strip()
    # Sample each active vehicle at most once per second for operational analytics.
    sampled = []
    for det in detections:
        last = runtime.last_speed_sample_by_track.get(int(det.track_id), 0.0)
        if now - last >= 1.0:
            sampled.append(det)
            runtime.last_speed_sample_by_track[int(det.track_id)] = now
    if sampled:
        runtime.store.record_speed_observations(
            sampled, observed_at=now,
            municipality=str(session_state.municipality).strip(),
            location=str(session_state.location_name).strip(),
            camera=camera_name,
            speed_limit_mph=float(session_state.speed_limit_mph),
            timezone_name=str(session_state.agency_timezone),
        )

    for candidate in result.crashes:
        event_id = str(uuid.uuid4())
        feedback = runtime.store.feedback_model().predict(
            detector_score=float(candidate.score),
            evidence=candidate.evidence,
            trigger_type=f"auto_{candidate.reason}",
        )
        priority = "high" if feedback.probability >= 0.80 else "normal" if feedback.probability >= 0.50 else "low"
        candidate_key = f"candidate:{event_id}"
        add_incident(session_state, IncidentRecord(
            key=candidate_key, timestamp=now, status="Capturando",
            title="Posible choque detectado", camera=camera_name,
            tracks=", ".join(f"#{t}" for t in candidate.involved_tracks) or "—",
            score=float(candidate.score), description=candidate.description,
        ))
        runtime.store.upsert_incident({
            "event_id": event_id,
            "detected_at": now,
            "trigger_type": f"auto_{candidate.reason}",
            "title": "Possible collision",
            "municipality": str(session_state.municipality).strip(),
            "location": str(session_state.location_name).strip(),
            "camera": camera_name,
            "source": runtime.source_label,
            "timezone": str(session_state.agency_timezone),
            "latitude": session_state.camera_latitude,
            "longitude": session_state.camera_longitude,
            "status": "pending",
            "detector_score": float(candidate.score),
            "review_score": feedback.probability,
            "priority": priority,
            "involved_tracks": candidate.involved_tracks,
            "description": candidate.description,
            "evidence": candidate.evidence,
        })
        notification_payload = {
            "event_id": event_id,
            "title": "Possible collision",
            "camera": camera_name,
            "location": str(session_state.location_name).strip(),
            "municipality": str(session_state.municipality).strip(),
            "priority": priority,
            "detector_score": float(candidate.score),
            "review_score": feedback.probability,
            "detected_at": now,
        }
        runtime.store.queue_notification(
            event_id, channel="in_app", payload=notification_payload,
        )
        webhook_url = str(getattr(session_state, "notification_webhook_url", "") or "").strip()
        if webhook_url:
            runtime.store.queue_notification(
                event_id, channel="webhook", destination=webhook_url, payload=notification_payload,
            )
        runtime.recorder.trigger_auto(
            candidate.reason, runtime.engine,
            trigger_frame=candidate.trigger_frame,
            event_metadata={
                "event_id": event_id,
                "detected_frame": candidate.detected_frame,
                "score": round(candidate.score, 6),
                "review_score": round(feedback.probability, 6),
                "feedback_model_active": feedback.active,
                "involved_tracks": candidate.involved_tracks,
                "description": candidate.description,
                "evidence": candidate.evidence,
                "municipality": str(session_state.municipality).strip(),
                "location": str(session_state.location_name).strip(),
                "camera": camera_name,
                "source": runtime.source_label,
                "timezone": str(session_state.agency_timezone),
                "latitude": session_state.camera_latitude,
                "longitude": session_state.camera_longitude,
            },
        )

    if session_state.manual_capture_requested:
        session_state.manual_capture_requested = False
        runtime.recorder.trigger_manual(runtime.engine, event_metadata={
            "municipality": str(session_state.municipality).strip(),
            "location": str(session_state.location_name).strip(),
            "camera": str(session_state.camera_name or runtime.source_label).strip(),
            "source": runtime.source_label,
            "timezone": str(session_state.agency_timezone),
            "latitude": session_state.camera_latitude,
            "longitude": session_state.camera_longitude,
            "description": "Captura manual desde el centro de operaciones.",
        })
        add_incident(session_state, IncidentRecord(
            key=f"manual:{runtime.engine.frame_count}:{time.time_ns()}",
            timestamp=time.time(), status="Capturando",
            title="Captura manual iniciada", camera=runtime.source_label,
            description="Guardando la ventana previa y posterior al evento.",
        ))

    saved = runtime.recorder.last_saved_event
    if saved is not None and str(saved) != runtime.last_saved_event:
        runtime.last_saved_event = str(saved)
        register_saved_event(session_state, runtime, Path(saved))

    valid_speeds = [d.speed for d in detections if d.speed_valid]
    session_state.active_rows = current_vehicle_rows(detections)
    session_state.current_max_speed = max(valid_speeds) if valid_speeds else None
    session_state.current_avg_speed = float(np.mean(valid_speeds)) if valid_speeds else None
    session_state.last_frame_rgb = cv2.cvtColor(result.annotated, cv2.COLOR_BGR2RGB)

    elapsed = time.perf_counter() - started
    if elapsed > 0:
        runtime.analysis_fps_samples.append(1.0 / elapsed)
        session_state.analysis_fps = float(np.mean(runtime.analysis_fps_samples))


def format_time(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def scan_library(event_dir: str, archive_dir: str) -> list[dict]:
    items: list[dict] = []
    events_root = Path(event_dir)
    if events_root.exists():
        for event_json in events_root.rglob("event.json"):
            metadata = read_json_file(event_json)
            clip = event_json.parent / "clip.mp4"
            if not clip.exists():
                continue
            ts = float(metadata.get("trigger_time_unix") or clip.stat().st_mtime)
            trigger_type = str(metadata.get("trigger_type", "evento"))
            items.append({
                "kind": "Incidente" if trigger_type.startswith("auto_") else "Captura manual",
                "timestamp": ts,
                "municipality": str(metadata.get("municipality") or "Sin municipio"),
                "location": str(metadata.get("location") or "Sin ubicación"),
                "camera": str(metadata.get("camera") or metadata.get("camera_id") or "Sin cámara"),
                "path": str(clip), "metadata_path": str(event_json),
                "sha256": str(metadata.get("video_sha256") or ""),
                "duration": float(metadata.get("pre_event_seconds", 0)) + float(metadata.get("post_event_seconds", 0)),
                "tracks": metadata.get("involved_tracks") or [],
                "score": metadata.get("score"),
            })

    archive_root = Path(archive_dir)
    if archive_root.exists():
        for metadata_path in archive_root.rglob("*.json"):
            metadata = read_json_file(metadata_path)
            if metadata.get("type") != "surveillance":
                continue
            video = metadata_path.with_suffix(".mp4")
            if not video.exists():
                continue
            ts = float(metadata.get("start_time_unix") or video.stat().st_mtime)
            items.append({
                "kind": "Grabación continua", "timestamp": ts,
                "municipality": str(metadata.get("municipality") or "Sin municipio"),
                "location": str(metadata.get("location") or "Sin ubicación"),
                "camera": str(metadata.get("camera") or "Sin cámara"),
                "path": str(video), "metadata_path": str(metadata_path),
                "sha256": str(metadata.get("video_sha256") or ""),
                "duration": float(metadata.get("duration_seconds") or 0),
                "tracks": [], "score": None,
            })
    return sorted(items, key=lambda item: float(item["timestamp"]), reverse=True)

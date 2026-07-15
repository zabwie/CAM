"""Panel operativo en español para monitoreo de tránsito en vivo.

Diseño orientado a operación:
  - cámara en vivo como elemento principal
  - estado del sistema y vehículos activos a simple vista
  - detección de choques y captura de evidencia integradas
  - configuración fuera del flujo diario del operador
  - video grabado disponible únicamente como modo de prueba

Ejemplos:
    streamlit run traffic_intel/app.py -- --camera 0 --calibration calib.json
    streamlit run traffic_intel/app.py -- --camera rtsp://usuario:clave@host/stream
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Debe ser el primer comando de Streamlit.
st.set_page_config(
    page_title="Centro de Operaciones de Tránsito",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from . import __version__
    from .analytics import CameraHealthAccumulator, VehiclePassageAggregator
    from .analytics_store import AnalyticsStore
    from .capture import LatestFrameCapture
    from .core.engine import TrafficEngine
    from .core.pipeline import TrafficIncidentPipeline
    from .motion.calibration import Calibration
    from .recording.archive import ArchiveRecorderConfig, ContinuousArchiveRecorder
    from .recording.event_recorder import EventRecorderConfig, RollingSegmentBuffer
    from .vision_quality import VisionQualityMonitor
except ImportError:  # ejecución directa con Streamlit
    from traffic_intel import __version__
    from traffic_intel.analytics import CameraHealthAccumulator, VehiclePassageAggregator
    from traffic_intel.analytics_store import AnalyticsStore
    from traffic_intel.capture import LatestFrameCapture
    from traffic_intel.core.engine import TrafficEngine
    from traffic_intel.core.pipeline import TrafficIncidentPipeline
    from traffic_intel.motion.calibration import Calibration
    from traffic_intel.recording.archive import ArchiveRecorderConfig, ContinuousArchiveRecorder
    from traffic_intel.recording.event_recorder import EventRecorderConfig, RollingSegmentBuffer
    from traffic_intel.vision_quality import VisionQualityMonitor


# ---------------------------------------------------------------------------
# Configuración de CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("--analytics-db", default="analytics.db")
    parser.add_argument("--speed-limit", type=float, default=None)
    args, _ = parser.parse_known_args()
    return args


ARGS = _parse_args()


# ---------------------------------------------------------------------------
# Sistema visual
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
        :root {
            --ti-border: rgba(128, 128, 128, .22);
            --ti-muted: rgba(128, 128, 128, .84);
            --ti-soft: rgba(128, 128, 128, .08);
            --ti-green: #16a36a;
            --ti-amber: #d18a00;
            --ti-red: #d64242;
        }

        .block-container {
            max-width: 1680px;
            padding-top: .9rem;
            padding-bottom: 2rem;
        }

        [data-testid="stSidebar"] {
            border-right: 1px solid var(--ti-border);
        }

        [data-testid="stSidebar"] .block-container {
            padding-top: 1rem;
        }

        h1, h2, h3 {
            letter-spacing: -.025em;
        }

        .ti-kicker {
            color: var(--ti-muted);
            font-size: .72rem;
            font-weight: 750;
            letter-spacing: .095em;
            text-transform: uppercase;
        }

        .ti-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: .15rem;
        }

        .ti-status {
            display: inline-flex;
            align-items: center;
            gap: .45rem;
            font-size: .82rem;
            font-weight: 700;
            white-space: nowrap;
        }

        .ti-dot {
            width: .58rem;
            height: .58rem;
            border-radius: 999px;
            display: inline-block;
            background: #8c8c8c;
        }

        .ti-dot.live { background: var(--ti-green); box-shadow: 0 0 0 4px rgba(22,163,106,.11); }
        .ti-dot.alert { background: var(--ti-red); box-shadow: 0 0 0 4px rgba(214,66,66,.11); }

        .ti-subtitle {
            color: var(--ti-muted);
            margin-top: -.45rem;
            margin-bottom: .85rem;
        }

        .ti-card {
            border: 1px solid var(--ti-border);
            border-radius: 13px;
            padding: .9rem 1rem;
            background: rgba(255,255,255,.01);
            min-width: 0;
            overflow: hidden;
        }

        .ti-card-label {
            color: var(--ti-muted);
            font-size: .76rem;
            font-weight: 700;
            margin-bottom: .35rem;
        }

        .ti-card-value {
            font-size: clamp(1.08rem, 1.7vw, 1.48rem);
            font-weight: 760;
            letter-spacing: -.03em;
            line-height: 1.1;
            white-space: normal;
            overflow-wrap: anywhere;
        }

        .ti-card-detail {
            color: var(--ti-muted);
            font-size: .76rem;
            margin-top: .28rem;
            overflow-wrap: anywhere;
        }

        .ti-live-frame {
            border: 1px solid var(--ti-border);
            border-radius: 14px;
            overflow: hidden;
            background: #0c0d0f;
        }

        .ti-incident {
            border: 1px solid var(--ti-border);
            border-left: 4px solid var(--ti-red);
            border-radius: 10px;
            padding: .7rem .8rem;
            margin-bottom: .55rem;
        }

        .ti-incident.pending { border-left-color: var(--ti-amber); }
        .ti-incident.saved { border-left-color: var(--ti-green); }

        .ti-incident-title {
            font-weight: 740;
            margin-bottom: .18rem;
        }

        .ti-incident-meta {
            color: var(--ti-muted);
            font-size: .78rem;
        }

        .ti-empty {
            border: 1px dashed var(--ti-border);
            border-radius: 12px;
            padding: 1.4rem;
            color: var(--ti-muted);
            text-align: center;
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--ti-border);
            border-radius: 12px;
            padding: .72rem .85rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: .25rem;
            border-bottom: 1px solid var(--ti-border);
        }

        .stTabs [data-baseweb="tab"] {
            padding-left: .75rem;
            padding-right: .75rem;
        }

        /* Evita que tablas y medidores salgan de su contenedor. */
        [data-testid="stDataFrame"] {
            max-width: 100%;
            overflow: hidden;
        }

        button[kind="primary"] {
            min-height: 2.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Estado de sesión y runtime
# ---------------------------------------------------------------------------


@dataclass
class LiveRuntime:
    cap: cv2.VideoCapture | LatestFrameCapture
    engine: TrafficEngine
    pipeline: TrafficIncidentPipeline
    recorder: RollingSegmentBuffer
    archive: ContinuousArchiveRecorder | None
    quality_monitor: VisionQualityMonitor
    analytics_store: AnalyticsStore | None
    passage_aggregator: VehiclePassageAggregator | None
    health_accumulator: CameraHealthAccumulator | None
    source_label: str
    source_value: str | int
    camera_fps: float
    last_sequence: int = 0
    last_capture_timestamp: float = field(default_factory=time.time)
    started_at: float = field(default_factory=time.time)
    seen_track_ids: set[int] = field(default_factory=set)
    analysis_fps_samples: deque[float] = field(default_factory=lambda: deque(maxlen=30))
    last_saved_event: str | None = None

    def close(self) -> None:
        try:
            try:
                if self.analytics_store is not None and self.passage_aggregator is not None:
                    self.analytics_store.write_passages(self.passage_aggregator.flush())
                if self.analytics_store is not None and self.health_accumulator is not None:
                    final_health = self.health_accumulator.flush(self.last_capture_timestamp)
                    if final_health is not None:
                        self.analytics_store.write_camera_health([final_health])
            finally:
                if self.analytics_store is not None:
                    self.analytics_store.close()
        finally:
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


def _init_state() -> None:
    defaults = {
        "monitoring": False,
        "runtime": None,
        "last_frame_rgb": None,
        "active_rows": [],
        "current_max_speed": None,
        "current_avg_speed": None,
        "analysis_fps": 0.0,
        "incidents": [],
        "last_error": None,
        "manual_capture_requested": False,
        "source_mode": "Cámara / RTSP",
        "source_value": str(ARGS.camera),
        "calibration_path": ARGS.calibration or ("calib.json" if Path("calib.json").exists() else ""),
        "model_path": ARGS.model if ARGS.model in {"models/yolo11n.pt", "models/yolo11s.pt", "models/yolo11m.pt", "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"} else "models/yolo11n.pt",
        "imgsz": int(ARGS.imgsz),
        "event_dir": ARGS.event_dir,
        "archive_dir": ARGS.archive_dir,
        "archive_enabled": True,
        "analytics_enabled": True,
        "analytics_db": ARGS.analytics_db,
        "speed_limit": ARGS.speed_limit,
        "speed_limit_input": float(ARGS.speed_limit or 0.0),
        "archive_segment_minutes": float(ARGS.archive_segment_minutes),
        "retention_days": int(ARGS.retention_days),
        "municipality": "",
        "location_name": "",
        "camera_name": "",
        "uploaded_demo_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


CLASS_TRANSLATIONS = {
    "car": "Automóvil",
    "truck": "Camión",
    "bus": "Autobús",
    "motorcycle": "Motocicleta",
    "bicycle": "Bicicleta",
    "vehicle": "Vehículo",
}


def _translate_class(name: str) -> str:
    return CLASS_TRANSLATIONS.get(str(name).lower(), str(name).replace("_", " ").title())


def _safe_source_label(source: str | int) -> str:
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


def _resolve_source() -> str | int:
    if st.session_state.source_mode == "Video de prueba":
        path = st.session_state.uploaded_demo_path or ARGS.video
        if not path:
            raise ValueError("Selecciona un video de prueba antes de iniciar.")
        return str(path)

    raw = str(st.session_state.source_value).strip()
    if not raw:
        raise ValueError("Especifica una cámara, índice o URL RTSP.")
    return int(raw) if raw.isdigit() else raw


def _load_calibration(path_value: str) -> Calibration | None:
    path_value = str(path_value or "").strip()
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró la calibración: {path}")
    return Calibration.load(path)


def _calibration_status(runtime: LiveRuntime | None) -> tuple[str, str]:
    if runtime is None or runtime.engine.calibration is None:
        return "Sin calibrar", "La velocidad no se mostrará como válida"
    grade = runtime.engine.calibration.quality_grade or "Disponible"
    return str(grade), "Calibración cargada"


def _start_runtime() -> LiveRuntime:
    source = _resolve_source()
    is_demo = st.session_state.source_mode == "Video de prueba"
    if is_demo:
        cap: cv2.VideoCapture | LatestFrameCapture = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente: {_safe_source_label(source)}")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    else:
        cap = LatestFrameCapture(source)
        fps = cap.fps
    calibration = _load_calibration(st.session_state.calibration_path)
    engine = TrafficEngine(
        model_path=st.session_state.model_path,
        calibration=calibration,
        fps=fps,
        imgsz=int(st.session_state.imgsz),
        retain_history=False,
    )
    pipeline = TrafficIncidentPipeline(engine)
    source_label = _safe_source_label(source)
    camera_name = str(st.session_state.camera_name or source_label).strip()
    recorder = RollingSegmentBuffer(
        fps=fps,
        output_dir=st.session_state.event_dir,
        config=EventRecorderConfig(
            pre_event_seconds=float(ARGS.pre_event_seconds),
            post_event_seconds=float(ARGS.post_event_seconds),
            camera_id=camera_name,
        ),
    )
    archive = None
    if bool(st.session_state.archive_enabled):
        archive = ContinuousArchiveRecorder(
            fps=fps,
            config=ArchiveRecorderConfig(
                output_dir=str(st.session_state.archive_dir),
                segment_seconds=float(st.session_state.archive_segment_minutes) * 60.0,
                retention_days=int(st.session_state.retention_days),
                municipality=str(st.session_state.municipality).strip(),
                location_name=str(st.session_state.location_name).strip(),
                camera_name=camera_name,
                source_label=source_label,
            ),
        )

    analytics_store = None
    passage_aggregator = None
    health_accumulator = None
    if bool(st.session_state.analytics_enabled):
        analytics_store = AnalyticsStore(str(st.session_state.analytics_db))
        passage_aggregator = VehiclePassageAggregator(
            camera_id=camera_name,
            municipality=str(st.session_state.municipality).strip(),
            location_id=str(st.session_state.location_name).strip(),
            speed_limit_mph=st.session_state.speed_limit,
            calibration_id=(
                Path(str(st.session_state.calibration_path)).name
                if str(st.session_state.calibration_path).strip()
                else ""
            ),
            software_version=__version__,
        )
        health_accumulator = CameraHealthAccumulator(camera_name)

    return LiveRuntime(
        cap=cap,
        engine=engine,
        pipeline=pipeline,
        recorder=recorder,
        archive=archive,
        quality_monitor=VisionQualityMonitor(),
        analytics_store=analytics_store,
        passage_aggregator=passage_aggregator,
        health_accumulator=health_accumulator,
        source_label=source_label,
        source_value=source,
        camera_fps=fps,
    )


def _stop_runtime() -> None:
    runtime = st.session_state.runtime
    if runtime is not None:
        try:
            runtime.close()
        except Exception:
            pass
    st.session_state.runtime = None
    st.session_state.monitoring = False


def _incident_exists(key: str) -> bool:
    return any(item.get("key") == key for item in st.session_state.incidents)


def _add_incident(record: IncidentRecord) -> None:
    if _incident_exists(record.key):
        return
    payload = {
        "key": record.key,
        "timestamp": record.timestamp,
        "status": record.status,
        "title": record.title,
        "camera": record.camera,
        "tracks": record.tracks,
        "score": record.score,
        "path": record.path,
        "description": record.description,
    }
    st.session_state.incidents = ([payload] + st.session_state.incidents)[:100]


def _register_saved_event(runtime: LiveRuntime, event_path: Path) -> None:
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

    # Completa el registro provisional en vez de dejar un incidente eterno
    # en estado "Capturando".  Primero intentamos casar por vehículos; para
    # capturas manuales usamos el título provisional.
    for item in st.session_state.incidents:
        same_auto_tracks = automatic and item.get("status") == "Capturando" and item.get("tracks") == tracks_text
        same_manual = (
            not automatic
            and item.get("status") == "Capturando"
            and "manual" in str(item.get("title", "")).lower()
        )
        if same_auto_tracks or same_manual:
            item.update(
                {
                    "key": f"saved:{event_id}",
                    "status": "Guardado",
                    "title": title,
                    "tracks": tracks_text,
                    "score": score_value,
                    "path": str(event_path),
                    "description": description,
                }
            )
            return

    _add_incident(
        IncidentRecord(
            key=f"saved:{event_id}",
            timestamp=float(metadata.get("trigger_time_unix") or time.time()),
            status="Guardado",
            title=title,
            camera=runtime.source_label,
            tracks=tracks_text,
            score=score_value,
            path=str(event_path),
            description=description,
        )
    )


def _current_vehicle_rows(detections: list) -> list[dict]:
    rows = []
    for d in sorted(detections, key=lambda item: item.track_id):
        rows.append(
            {
                "ID": f"#{d.track_id}",
                "Tipo": _translate_class(d.class_name),
                "Velocidad": f"{d.speed:.1f} mph" if d.speed_valid else "—",
                "Confianza": f"{max(0.0, min(1.0, d.measurement_confidence)):.0%}"
                if d.measurement_confidence > 0
                else "—",
            }
        )
    return rows


def _process_live_frame(runtime: LiveRuntime) -> None:
    started = time.perf_counter()
    sequence_gap = 0
    if isinstance(runtime.cap, LatestFrameCapture):
        packet = runtime.cap.read_packet(
            after_sequence=runtime.last_sequence,
            timeout=5.0,
        )
        if packet is None:
            raise RuntimeError("La cámara dejó de entregar video.")
        sequence_gap = max(0, packet.sequence - runtime.last_sequence - 1)
        runtime.last_sequence = packet.sequence
        frame = packet.image
        capture_timestamp = packet.capture_timestamp
        monotonic_timestamp = packet.monotonic_timestamp
    else:
        ok, frame = runtime.cap.read()
        if not ok or frame is None:
            raise RuntimeError("La cámara dejó de entregar video.")
        media_timestamp = float(runtime.cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        if media_timestamp <= 0:
            media_timestamp = (runtime.engine.frame_count + 1) / runtime.camera_fps
        capture_timestamp = media_timestamp
        monotonic_timestamp = media_timestamp

    runtime.last_capture_timestamp = capture_timestamp
    quality = runtime.quality_monitor.update(frame)

    if runtime.archive is not None:
        runtime.archive.write_frame(
            frame,
            frame_number=runtime.engine.frame_count + 1,
            timestamp=time.time(),
        )

    result = runtime.pipeline.process_frame(
        frame,
        optical_flow=True,
        capture_timestamp=capture_timestamp,
        monotonic_timestamp=monotonic_timestamp,
        vision_state=quality.state,
    )
    detections = result.detections
    runtime.seen_track_ids.update(d.track_id for d in detections)

    if (
        runtime.analytics_store is not None
        and runtime.passage_aggregator is not None
        and runtime.health_accumulator is not None
    ):
        runtime.analytics_store.write_passages(
            runtime.passage_aggregator.update(
                detections,
                capture_timestamp=capture_timestamp,
                monotonic_timestamp=monotonic_timestamp,
                vision_state=quality.state,
            )
        )
        runtime.analytics_store.write_camera_health(
            runtime.health_accumulator.update(
                capture_timestamp=capture_timestamp,
                monotonic_timestamp=monotonic_timestamp,
                sequence_gap=sequence_gap,
                detections=detections,
                quality=quality,
            )
        )

    # Primero se escribe el frame para conservar correctamente la ventana previa.
    runtime.recorder.write_frame(result.annotated, detections, runtime.engine.frame_count)

    for candidate in result.crashes:
        candidate_key = (
            f"candidate:{candidate.trigger_frame}:"
            f"{'-'.join(map(str, sorted(candidate.involved_tracks)))}"
        )
        _add_incident(
            IncidentRecord(
                key=candidate_key,
                timestamp=time.time(),
                status="Capturando",
                title="Posible choque detectado",
                camera=runtime.source_label,
                tracks=", ".join(f"#{t}" for t in candidate.involved_tracks) or "—",
                score=float(candidate.score),
                description=candidate.description,
            )
        )
        runtime.recorder.trigger_auto(
            candidate.reason,
            runtime.engine,
            trigger_frame=candidate.trigger_frame,
            event_metadata={
                "detected_frame": candidate.detected_frame,
                "score": round(candidate.score, 6),
                "involved_tracks": candidate.involved_tracks,
                "description": candidate.description,
                "evidence": candidate.evidence,
                "municipality": str(st.session_state.municipality).strip(),
                "location": str(st.session_state.location_name).strip(),
                "camera": str(st.session_state.camera_name or runtime.source_label).strip(),
                "source": runtime.source_label,
                "vision_state": quality.state,
            },
        )
        if runtime.analytics_store is not None:
            runtime.analytics_store.write_incident(
                event_id=(
                    f"{runtime.source_label}:{capture_timestamp:.6f}:"
                    + "-".join(map(str, candidate.involved_tracks))
                ),
                camera_id=str(st.session_state.camera_name or runtime.source_label).strip(),
                occurred_at=capture_timestamp,
                incident_type=candidate.reason,
                score=candidate.score,
                involved_tracks=candidate.involved_tracks,
                metadata={
                    "description": candidate.description,
                    "trigger_frame": candidate.trigger_frame,
                    "detected_frame": candidate.detected_frame,
                    "vision_state": quality.state,
                },
            )

    if st.session_state.manual_capture_requested:
        st.session_state.manual_capture_requested = False
        runtime.recorder.trigger_manual(
            runtime.engine,
            event_metadata={
                "municipality": str(st.session_state.municipality).strip(),
                "location": str(st.session_state.location_name).strip(),
                "camera": str(st.session_state.camera_name or runtime.source_label).strip(),
                "source": runtime.source_label,
                "description": "Captura manual desde el centro de operaciones.",
            },
        )
        _add_incident(
            IncidentRecord(
                key=f"manual:{runtime.engine.frame_count}:{time.time_ns()}",
                timestamp=time.time(),
                status="Capturando",
                title="Captura manual iniciada",
                camera=runtime.source_label,
                description="Guardando la ventana previa y posterior al evento.",
            )
        )

    saved = runtime.recorder.last_saved_event
    if saved is not None and str(saved) != runtime.last_saved_event:
        runtime.last_saved_event = str(saved)
        _register_saved_event(runtime, Path(saved))

    valid_speeds = [d.speed for d in detections if d.speed_valid]
    st.session_state.active_rows = _current_vehicle_rows(detections)
    st.session_state.current_max_speed = max(valid_speeds) if valid_speeds else None
    st.session_state.current_avg_speed = float(np.mean(valid_speeds)) if valid_speeds else None
    st.session_state.last_frame_rgb = cv2.cvtColor(result.annotated, cv2.COLOR_BGR2RGB)

    elapsed = time.perf_counter() - started
    if elapsed > 0:
        runtime.analysis_fps_samples.append(1.0 / elapsed)
        st.session_state.analysis_fps = float(np.mean(runtime.analysis_fps_samples))


def _format_time(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _read_json_file(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@st.cache_data(ttl=5, show_spinner=False)
def _scan_library(event_dir: str, archive_dir: str) -> list[dict]:
    items: list[dict] = []

    events_root = Path(event_dir)
    if events_root.exists():
        for event_json in events_root.rglob("event.json"):
            metadata = _read_json_file(event_json)
            clip = event_json.parent / "clip.mp4"
            if not clip.exists():
                continue
            ts = float(metadata.get("trigger_time_unix") or clip.stat().st_mtime)
            trigger_type = str(metadata.get("trigger_type", "evento"))
            items.append(
                {
                    "kind": "Incidente" if trigger_type.startswith("auto_") else "Captura manual",
                    "timestamp": ts,
                    "municipality": str(metadata.get("municipality") or "Sin municipio"),
                    "location": str(metadata.get("location") or "Sin ubicación"),
                    "camera": str(metadata.get("camera") or metadata.get("camera_id") or "Sin cámara"),
                    "path": str(clip),
                    "metadata_path": str(event_json),
                    "sha256": str(metadata.get("video_sha256") or ""),
                    "duration": float(metadata.get("pre_event_seconds", 0)) + float(metadata.get("post_event_seconds", 0)),
                    "tracks": metadata.get("involved_tracks") or [],
                    "score": metadata.get("score"),
                }
            )

    archive_root = Path(archive_dir)
    if archive_root.exists():
        for metadata_path in archive_root.rglob("*.json"):
            metadata = _read_json_file(metadata_path)
            if metadata.get("type") != "surveillance":
                continue
            video = metadata_path.with_suffix(".mp4")
            if not video.exists():
                continue
            ts = float(metadata.get("start_time_unix") or video.stat().st_mtime)
            items.append(
                {
                    "kind": "Grabación continua",
                    "timestamp": ts,
                    "municipality": str(metadata.get("municipality") or "Sin municipio"),
                    "location": str(metadata.get("location") or "Sin ubicación"),
                    "camera": str(metadata.get("camera") or "Sin cámara"),
                    "path": str(video),
                    "metadata_path": str(metadata_path),
                    "sha256": str(metadata.get("video_sha256") or ""),
                    "duration": float(metadata.get("duration_seconds") or 0),
                    "tracks": [],
                    "score": None,
                }
            )

    return sorted(items, key=lambda item: float(item["timestamp"]), reverse=True)


def _status_html(live: bool) -> str:
    cls = "live" if live else ""
    text = "EN VIVO" if live else "DETENIDO"
    return f'<span class="ti-status"><span class="ti-dot {cls}"></span>{text}</span>'


def _summary_card(label: str, value: str, detail: str = "") -> None:
    st.markdown(
        f"""
        <div class="ti-card">
            <div class="ti-card-label">{label}</div>
            <div class="ti-card-value">{value}</div>
            <div class="ti-card-detail">{detail or '&nbsp;'}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Barra lateral: controles de operación
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="ti-kicker">Sistema de Tránsito</div>', unsafe_allow_html=True)
    st.subheader("Control de monitoreo")

    if st.session_state.monitoring and st.session_state.runtime is not None:
        st.success(f"Activo · {st.session_state.runtime.source_label}")
        stop_clicked = st.button("Detener monitoreo", use_container_width=True)
        if stop_clicked:
            _stop_runtime()
            st.rerun()

        if st.button("Guardar evento ahora", use_container_width=True):
            st.session_state.manual_capture_requested = True
    else:
        st.caption("Selecciona la fuente y comienza el monitoreo.")
        start_clicked = st.button("Iniciar monitoreo", type="primary", use_container_width=True)
        if start_clicked:
            try:
                st.session_state.runtime = _start_runtime()
                st.session_state.monitoring = True
                st.session_state.last_error = None
                st.rerun()
            except Exception as exc:
                st.session_state.last_error = str(exc)

    if st.session_state.last_error:
        st.error(st.session_state.last_error)

    st.divider()
    with st.expander("Fuente y configuración", expanded=not st.session_state.monitoring):
        source_mode = st.radio(
            "Modo",
            ["Cámara / RTSP", "Video de prueba"],
            key="source_mode",
            disabled=st.session_state.monitoring,
        )

        if source_mode == "Cámara / RTSP":
            st.text_input(
                "Cámara, índice o URL RTSP",
                key="source_value",
                disabled=st.session_state.monitoring,
                help="Ejemplos: 0 o rtsp://host/stream",
            )
        else:
            uploaded = st.file_uploader(
                "Video de prueba",
                type=["mp4", "avi", "mov"],
                disabled=st.session_state.monitoring,
            )
            if uploaded is not None and not st.session_state.monitoring:
                suffix = Path(uploaded.name).suffix or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uploaded.read())
                tmp.flush()
                tmp.close()
                st.session_state.uploaded_demo_path = tmp.name
                st.caption(f"Listo: {uploaded.name}")
            elif ARGS.video:
                st.caption(f"CLI: {Path(ARGS.video).name}")

        st.markdown("**Identificación de la cámara**")
        st.text_input(
            "Municipio",
            key="municipality",
            disabled=st.session_state.monitoring,
            placeholder="Ej. San Juan",
        )
        st.text_input(
            "Ubicación",
            key="location_name",
            disabled=st.session_state.monitoring,
            placeholder="Ej. Ave. Ponce de León / Hato Rey",
        )
        st.text_input(
            "Nombre de cámara",
            key="camera_name",
            disabled=st.session_state.monitoring,
            placeholder="Ej. Intersección 04",
        )

        st.markdown("**Analítica del piloto**")
        st.toggle(
            "Guardar observaciones y salud de cámara",
            key="analytics_enabled",
            disabled=st.session_state.monitoring,
        )
        if st.session_state.analytics_enabled:
            st.text_input(
                "Base de datos SQLite",
                key="analytics_db",
                disabled=st.session_state.monitoring,
            )
            st.number_input(
                "Límite configurado (mph)",
                min_value=0.0,
                max_value=150.0,
                step=1.0,
                key="speed_limit_input",
                disabled=st.session_state.monitoring,
                help="Use 0 para registrar velocidades sin clasificar exceso.",
            )
            st.session_state.speed_limit = (
                float(st.session_state.speed_limit_input)
                if float(st.session_state.speed_limit_input) > 0
                else None
            )

        st.markdown("**Archivo de vigilancia**")
        st.toggle(
            "Guardar grabación continua",
            key="archive_enabled",
            disabled=st.session_state.monitoring,
        )
        if st.session_state.archive_enabled:
            st.text_input(
                "Carpeta de archivo",
                key="archive_dir",
                disabled=st.session_state.monitoring,
            )
            st.selectbox(
                "Duración de cada segmento",
                options=[1.0, 5.0, 10.0, 15.0, 30.0],
                format_func=lambda value: f"{int(value)} min",
                key="archive_segment_minutes",
                disabled=st.session_state.monitoring,
            )
            st.number_input(
                "Retención (días)",
                min_value=1,
                max_value=3650,
                step=1,
                key="retention_days",
                disabled=st.session_state.monitoring,
            )

        st.text_input(
            "Archivo de calibración",
            key="calibration_path",
            disabled=st.session_state.monitoring,
            placeholder="calib.json",
        )
        st.selectbox(
            "Modelo",
            ["models/yolo11n.pt", "models/yolo11s.pt", "models/yolo11m.pt"],
            key="model_path",
            disabled=st.session_state.monitoring,
        )
        st.select_slider(
            "Resolución de inferencia",
            options=[640, 960, 1280, 1536],
            key="imgsz",
            disabled=st.session_state.monitoring,
        )

    st.divider()
    st.caption(f"v{__version__} · Panel operativo en vivo")


# ---------------------------------------------------------------------------
# Encabezado
# ---------------------------------------------------------------------------

runtime: LiveRuntime | None = st.session_state.runtime
source_title = runtime.source_label if runtime else _safe_source_label(st.session_state.source_value)

st.markdown(
    f"""
    <div class="ti-topline">
        <div class="ti-kicker">Centro de Operaciones</div>
        {_status_html(bool(st.session_state.monitoring and runtime is not None))}
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("Monitoreo de tránsito")
st.markdown(
    f'<div class="ti-subtitle">{source_title} · detección, seguimiento, velocidad e incidentes en tiempo real.</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Área viva: un fragmento actualiza únicamente el contenido operativo.
# ---------------------------------------------------------------------------

refresh_interval = "150ms" if st.session_state.monitoring else None


@st.fragment(run_every=refresh_interval)
def live_workspace() -> None:
    runtime_local: LiveRuntime | None = st.session_state.runtime

    if st.session_state.monitoring and runtime_local is not None:
        try:
            _process_live_frame(runtime_local)
        except Exception as exc:
            st.session_state.last_error = str(exc)
            _stop_runtime()
            st.error(f"Monitoreo detenido: {exc}")
            return

    # Señal de incidente reciente para que el operador no tenga que buscarla.
    recent_incident = next(
        (
            item
            for item in st.session_state.incidents
            if time.time() - float(item.get("timestamp", 0)) <= 8
            and item.get("status") == "Capturando"
        ),
        None,
    )
    if recent_incident:
        st.error(
            f"INCIDENTE DETECTADO · {recent_incident['title']} · "
            f"Vehículos {recent_incident.get('tracks', '—')}"
        )

    live_tab, incidents_tab, library_tab = st.tabs(["En vivo", "Incidentes", "Biblioteca"])

    with live_tab:
        feed_col, side_col = st.columns([2.35, 1], gap="large")

        with feed_col:
            st.markdown("### Cámara en vivo")
            if st.session_state.last_frame_rgb is not None:
                st.image(
                    st.session_state.last_frame_rgb,
                    use_container_width=True,
                    channels="RGB",
                )
            else:
                st.markdown(
                    '<div class="ti-empty">Inicia el monitoreo para ver la cámara en tiempo real.</div>',
                    unsafe_allow_html=True,
                )

            if runtime_local is not None:
                st.caption(
                    f"Fuente: {runtime_local.source_label} · "
                    f"FPS de cámara: {runtime_local.camera_fps:.1f} · "
                    f"FPS de análisis: {st.session_state.analysis_fps:.1f}"
                )

        with side_col:
            st.markdown("### Estado actual")
            active_count = len(st.session_state.active_rows)
            max_speed = st.session_state.current_max_speed
            calibration_value, calibration_detail = _calibration_status(runtime_local)

            a, b = st.columns(2)
            with a:
                _summary_card("Vehículos en escena", f"{active_count}", "Ahora mismo")
            with b:
                _summary_card(
                    "Velocidad máxima",
                    f"{max_speed:.1f} mph" if max_speed is not None else "—",
                    "Lectura actual válida" if max_speed is not None else "Sin lectura válida",
                )

            c, d = st.columns(2)
            with c:
                _summary_card(
                    "Rendimiento",
                    f"{st.session_state.analysis_fps:.1f} FPS",
                    "Procesamiento del modelo",
                )
            with d:
                _summary_card("Calibración", calibration_value, calibration_detail)

            st.markdown("#### Vehículos activos")
            if st.session_state.active_rows:
                active_df = pd.DataFrame(st.session_state.active_rows)
                st.dataframe(
                    active_df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(320, 42 + 35 * len(active_df)),
                    column_config={
                        "ID": st.column_config.TextColumn("ID", width="small"),
                        "Tipo": st.column_config.TextColumn("Tipo", width="medium"),
                        "Velocidad": st.column_config.TextColumn("Velocidad", width="medium"),
                        "Confianza": st.column_config.TextColumn("Confianza", width="small"),
                    },
                )
            else:
                st.caption("No hay vehículos confirmados en este momento.")

        st.write("")
        metric_cols = st.columns(4)
        total_seen = len(runtime_local.seen_track_ids) if runtime_local is not None else 0
        saved_count = sum(1 for item in st.session_state.incidents if item.get("status") == "Guardado")
        pending_count = sum(1 for item in st.session_state.incidents if item.get("status") == "Capturando")
        avg_speed = st.session_state.current_avg_speed
        metric_cols[0].metric("Vehículos vistos en la sesión", total_seen)
        metric_cols[1].metric("Incidentes guardados", saved_count)
        metric_cols[2].metric("Capturas en curso", pending_count)
        metric_cols[3].metric(
            "Velocidad promedio actual",
            f"{avg_speed:.1f} mph" if avg_speed is not None else "—",
        )

    with library_tab:
        st.markdown("### Biblioteca de vigilancia")
        st.caption(
            "Grabaciones continuas e incidentes guardados, organizados por municipio, ubicación, cámara y fecha."
        )

        library_items = _scan_library(
            str(st.session_state.event_dir),
            str(st.session_state.archive_dir),
        )
        if not library_items:
            st.markdown(
                '<div class="ti-empty">Todavía no hay grabaciones archivadas. Activa la grabación continua o guarda un incidente.</div>',
                unsafe_allow_html=True,
            )
        else:
            municipalities = ["Todos"] + sorted({item["municipality"] for item in library_items})
            locations = ["Todas"] + sorted({item["location"] for item in library_items})
            cameras = ["Todas"] + sorted({item["camera"] for item in library_items})
            kinds = ["Todos"] + sorted({item["kind"] for item in library_items})

            filter_cols = st.columns(4)
            municipality_filter = filter_cols[0].selectbox(
                "Municipio", municipalities, key="library_municipality"
            )
            location_filter = filter_cols[1].selectbox(
                "Ubicación", locations, key="library_location"
            )
            camera_filter = filter_cols[2].selectbox(
                "Cámara", cameras, key="library_camera"
            )
            kind_filter = filter_cols[3].selectbox(
                "Tipo", kinds, key="library_kind"
            )

            dates = [datetime.fromtimestamp(float(item["timestamp"])).date() for item in library_items]
            min_date, max_date = min(dates), max(dates)
            date_range = st.date_input(
                "Rango de fechas",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                key="library_date_range",
            )
            if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date = end_date = date_range

            filtered = []
            for item in library_items:
                item_date = datetime.fromtimestamp(float(item["timestamp"])).date()
                if municipality_filter != "Todos" and item["municipality"] != municipality_filter:
                    continue
                if location_filter != "Todas" and item["location"] != location_filter:
                    continue
                if camera_filter != "Todas" and item["camera"] != camera_filter:
                    continue
                if kind_filter != "Todos" and item["kind"] != kind_filter:
                    continue
                if item_date < start_date or item_date > end_date:
                    continue
                filtered.append(item)

            st.caption(f"{len(filtered)} grabación(es) encontradas")
            if not filtered:
                st.info("No hay grabaciones que coincidan con esos filtros.")
            else:
                options = {}
                for index, item in enumerate(filtered):
                    dt = datetime.fromtimestamp(float(item["timestamp"]))
                    label = (
                        f"{dt:%Y-%m-%d %H:%M:%S} · {item['kind']} · "
                        f"{item['municipality']} · {item['camera']}"
                    )
                    options[f"{label} · {index + 1}"] = item

                selected_label = st.selectbox(
                    "Seleccionar grabación",
                    list(options.keys()),
                    key="library_selected_item",
                )
                selected_item = options[selected_label]
                video_path = Path(selected_item["path"])

                detail_left, detail_right = st.columns([1.75, 1], gap="large")
                with detail_left:
                    st.video(str(video_path))
                with detail_right:
                    st.markdown("#### Detalles")
                    selected_dt = datetime.fromtimestamp(float(selected_item["timestamp"]))
                    st.write(f"**Fecha:** {selected_dt:%Y-%m-%d %H:%M:%S}")
                    st.write(f"**Tipo:** {selected_item['kind']}")
                    st.write(f"**Municipio:** {selected_item['municipality']}")
                    st.write(f"**Ubicación:** {selected_item['location']}")
                    st.write(f"**Cámara:** {selected_item['camera']}")
                    duration = float(selected_item.get("duration") or 0)
                    st.write(f"**Duración:** {duration:.1f} s")
                    digest = str(selected_item.get("sha256") or "")
                    if digest:
                        st.code(digest, language=None)

                    metadata_path = Path(selected_item["metadata_path"])
                    if metadata_path.exists():
                        st.download_button(
                            "Descargar metadatos",
                            data=metadata_path.read_bytes(),
                            file_name=metadata_path.name,
                            mime="application/json",
                            use_container_width=True,
                            key="library_download_metadata",
                        )

    with incidents_tab:
        st.markdown("### Incidentes recientes")
        st.caption("Los choques detectados aparecen aquí de inmediato; el clip queda disponible al terminar la ventana posterior al evento.")

        incidents = st.session_state.incidents
        if not incidents:
            st.markdown(
                '<div class="ti-empty">Todavía no hay incidentes en esta sesión.</div>',
                unsafe_allow_html=True,
            )
            return

        saved_incidents = [item for item in incidents if item.get("path")]
        list_col, detail_col = st.columns([1.05, 1.6], gap="large")

        with list_col:
            for item in incidents[:12]:
                status = item.get("status", "")
                css_status = "saved" if status == "Guardado" else "pending"
                score = item.get("score")
                score_text = f" · confianza {float(score):.0%}" if isinstance(score, (int, float)) else ""
                st.markdown(
                    f"""
                    <div class="ti-incident {css_status}">
                        <div class="ti-incident-title">{item.get('title', 'Incidente')}</div>
                        <div class="ti-incident-meta">
                            {_format_time(float(item.get('timestamp', time.time())))} · {status}{score_text}<br>
                            Vehículos: {item.get('tracks', '—')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with detail_col:
            st.markdown("#### Evidencia")
            if not saved_incidents:
                st.info("La evidencia aparecerá aquí cuando finalice la captura posterior al evento.")
            else:
                options = {
                    f"{_format_time(float(item['timestamp']))} · {item['title']}": item
                    for item in saved_incidents
                }
                selected_label = st.selectbox("Seleccionar incidente", list(options.keys()))
                selected = options[selected_label]
                event_path = Path(selected["path"])
                clip = event_path / "clip.mp4"
                if clip.exists():
                    st.video(str(clip))
                st.write(selected.get("description") or "Evidencia guardada.")

                meta_cols = st.columns(3)
                meta_cols[0].metric("Estado", selected.get("status", "—"))
                meta_cols[1].metric("Vehículos", selected.get("tracks", "—"))
                meta_cols[2].metric(
                    "Confianza",
                    f"{float(selected['score']):.0%}"
                    if isinstance(selected.get("score"), (int, float))
                    else "—",
                )

                event_json = event_path / "event.json"
                telemetry = event_path / "telemetry.csv"
                download_cols = st.columns(2)
                if event_json.exists():
                    download_cols[0].download_button(
                        "Descargar metadatos",
                        data=event_json.read_bytes(),
                        file_name=event_json.name,
                        mime="application/json",
                        use_container_width=True,
                    )
                if telemetry.exists():
                    download_cols[1].download_button(
                        "Descargar telemetría",
                        data=telemetry.read_bytes(),
                        file_name=telemetry.name,
                        mime="text/csv",
                        use_container_width=True,
                    )


live_workspace()

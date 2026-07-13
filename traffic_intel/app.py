"""Traffic Intelligence operations dashboard.

Examples:
    streamlit run traffic_intel/app.py -- --camera 0 --calibration calib.json
    streamlit run traffic_intel/app.py -- --camera rtsp://user:password@host/stream
"""

from __future__ import annotations

from pathlib import Path
import os

import streamlit as st

from traffic_intel.app.runtime import parse_args, process_live_frame, start_runtime, stop_runtime
from traffic_intel.app.views import (
    render_analytics_tab,
    render_css,
    render_incident_queue_tab,
    render_library_tab,
    render_monitoring_tab,
    render_overview_tab,
    render_settings_tab,
    render_sidebar,
)
from traffic_intel.ops.store import IncidentStore

st.set_page_config(
    page_title="Traffic Intelligence Operations",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)
render_css()
args = parse_args()


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
        "source_mode": "Camera / RTSP",
        "source_value": str(args.camera),
        "calibration_path": args.calibration or ("calib.json" if Path("calib.json").exists() else ""),
        "model_path": args.model,
        "imgsz": int(args.imgsz),
        "event_dir": args.event_dir,
        "archive_dir": args.archive_dir,
        "archive_enabled": True,
        "archive_segment_minutes": float(args.archive_segment_minutes),
        "retention_days": int(args.retention_days),
        "municipality": "",
        "location_name": "",
        "camera_name": "",
        "camera_latitude": None,
        "camera_longitude": None,
        "agency_timezone": "America/Puerto_Rico",
        "uploaded_demo_path": None,
        "db_path": args.db_path,
        "speed_limit_mph": float(args.speed_limit_mph),
        "reviewer_name": "Operator",
        "notification_webhook_url": os.environ.get("TRAFFIC_INTEL_WEBHOOK_URL", ""),
        "event_import_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()
store = IncidentStore(str(st.session_state.db_path or args.db_path))
import_signature = (
    str(Path(st.session_state.db_path or args.db_path).resolve()),
    str(Path(st.session_state.event_dir or args.event_dir).resolve()),
)
if st.session_state.event_import_signature != import_signature:
    store.ingest_event_directory(str(st.session_state.event_dir or args.event_dir))
    st.session_state.event_import_signature = import_signature

action = render_sidebar(st.session_state, store)
if action == "start":
    try:
        st.session_state.runtime = start_runtime(st.session_state, args)
        st.session_state.monitoring = True
        st.rerun()
    except Exception as exc:
        st.session_state.last_error = str(exc)
elif action == "stop":
    stop_runtime(st.session_state)
    st.rerun()

if st.session_state.monitoring and st.session_state.runtime is not None:
    try:
        process_live_frame(st.session_state, st.session_state.runtime, args)
    except Exception as exc:
        stop_runtime(st.session_state)
        st.session_state.last_error = str(exc)
        st.rerun()

if st.session_state.last_error:
    st.error(st.session_state.last_error)

(
    tab_overview,
    tab_queue,
    tab_analytics,
    tab_monitor,
    tab_library,
    tab_settings,
) = st.tabs(["Overview", "Review queue", "Analytics", "Live monitor", "Evidence library", "Settings"])

render_overview_tab(st.session_state, tab_overview, store)
render_incident_queue_tab(st.session_state, tab_queue, store)
render_analytics_tab(st.session_state, tab_analytics, store)
render_monitoring_tab(st.session_state, tab_monitor)
render_library_tab(st.session_state, tab_library, args)
render_settings_tab(st.session_state, tab_settings, store)

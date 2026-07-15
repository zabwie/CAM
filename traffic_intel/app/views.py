"""Streamlit UI components for the agency operations dashboard."""

from __future__ import annotations

import datetime as dt
import json
import tempfile
import time
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import streamlit as st

from traffic_intel.app.runtime import calibration_status, format_time, scan_library
from traffic_intel.ops.store import IncidentStore
from traffic_intel.ops.evidence import build_evidence_package
from traffic_intel.ops.notifications import WebhookDispatcher


STATUS_LABELS = {
    "pending": "Pending review",
    "approved": "Approved incident",
    "dismissed": "Dismissed",
    "needs_info": "Needs more information",
}


def format_datetime(ts: float, timezone_name: str) -> str:
    try:
        zone = ZoneInfo(str(timezone_name or "UTC"))
    except ZoneInfoNotFoundError:
        zone = ZoneInfo("UTC")
    return dt.datetime.fromtimestamp(float(ts), tz=zone).strftime("%Y-%m-%d %H:%M:%S %Z")


def render_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                --ti-border: rgba(128, 128, 128, .22);
                --ti-muted: rgba(128, 128, 128, .86);
                --ti-soft: rgba(128, 128, 128, .08);
                --ti-green: #16a36a;
                --ti-amber: #d18a00;
                --ti-red: #d64242;
                --ti-blue: #2f73d9;
            }
            .block-container { max-width: 1740px; padding-top: .9rem; padding-bottom: 2rem; }
            [data-testid="stSidebar"] { border-right: 1px solid var(--ti-border); }
            h1, h2, h3 { letter-spacing: -.025em; }
            .ti-kicker { color: var(--ti-muted); font-size: .72rem; font-weight: 750; letter-spacing: .095em; text-transform: uppercase; }
            .ti-card { border: 1px solid var(--ti-border); border-radius: 13px; padding: .9rem 1rem; background: rgba(255,255,255,.01); min-width: 0; overflow: hidden; }
            .ti-card-label { color: var(--ti-muted); font-size: .76rem; font-weight: 700; margin-bottom: .35rem; }
            .ti-card-value { font-size: clamp(1.08rem, 1.7vw, 1.48rem); font-weight: 760; letter-spacing: -.03em; line-height: 1.1; overflow-wrap: anywhere; }
            .ti-card-detail { color: var(--ti-muted); font-size: .76rem; margin-top: .28rem; overflow-wrap: anywhere; }
            .ti-incident { border: 1px solid var(--ti-border); border-left: 4px solid var(--ti-red); border-radius: 10px; padding: .72rem .82rem; margin-bottom: .55rem; }
            .ti-incident.pending { border-left-color: var(--ti-amber); }
            .ti-incident.approved { border-left-color: var(--ti-green); }
            .ti-incident.dismissed { border-left-color: #777; }
            .ti-incident.needs_info { border-left-color: var(--ti-blue); }
            .ti-incident-title { font-weight: 740; margin-bottom: .18rem; }
            .ti-incident-meta { color: var(--ti-muted); font-size: .78rem; }
            .ti-empty { border: 1px dashed var(--ti-border); border-radius: 12px; padding: 1.4rem; color: var(--ti-muted); text-align: center; }
            div[data-testid="stMetric"] { border: 1px solid var(--ti-border); border-radius: 12px; padding: .72rem .85rem; }
            .stTabs [data-baseweb="tab-list"] { gap: .25rem; border-bottom: 1px solid var(--ti-border); }
            button[kind="primary"] { min-height: 2.75rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def summary_card(label: str, value: str, detail: str = "") -> None:
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


def render_sidebar(session_state, store: IncidentStore) -> str | None:
    with st.sidebar:
        st.markdown('<div class="ti-kicker">Traffic Intelligence</div>', unsafe_allow_html=True)
        st.subheader("Operations")
        pending = store.analytics_summary().get("incidents_pending", 0)
        st.caption(f"{pending} incident{'s' if pending != 1 else ''} waiting for review")

        if session_state.monitoring and session_state.runtime is not None:
            st.success(f"Monitoring · {session_state.runtime.source_label}")
            if st.button("Stop monitoring", width="stretch"):
                return "stop"
            if st.button("Save event now", width="stretch"):
                session_state.manual_capture_requested = True
        else:
            session_state.source_mode = st.radio(
                "Video source", ["Camera / RTSP", "Test video"],
                index=0 if session_state.source_mode != "Test video" else 1,
            )
            if session_state.source_mode == "Camera / RTSP":
                session_state.source_value = st.text_input("Camera index or RTSP URL", value=str(session_state.source_value))
            else:
                uploaded = st.file_uploader("Upload test video", type=["mp4", "avi", "mov", "mkv"])
                if uploaded is not None:
                    suffix = Path(uploaded.name).suffix.lower()
                    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
                        st.error("Unsupported video format.")
                        return None
                    tmp = Path(tempfile.mkdtemp(prefix="traffic-intel-upload-")) / f"upload{suffix}"
                    tmp.write_bytes(uploaded.read())
                    session_state.uploaded_demo_path = str(tmp)

        st.divider()
        st.markdown("#### Deployment context")
        session_state.municipality = st.text_input("Agency / municipality", value=str(session_state.municipality))
        session_state.location_name = st.text_input("Location / intersection", value=str(session_state.location_name))
        session_state.camera_name = st.text_input("Camera name", value=str(session_state.camera_name))
        coord_cols = st.columns(2)
        lat_text = coord_cols[0].text_input(
            "Latitude (optional)",
            value="" if session_state.camera_latitude is None else str(session_state.camera_latitude),
        )
        lon_text = coord_cols[1].text_input(
            "Longitude (optional)",
            value="" if session_state.camera_longitude is None else str(session_state.camera_longitude),
        )
        try:
            session_state.camera_latitude = float(lat_text) if lat_text.strip() else None
            session_state.camera_longitude = float(lon_text) if lon_text.strip() else None
        except ValueError:
            st.warning("Latitude and longitude must be numeric.")
        session_state.agency_timezone = st.text_input(
            "Agency timezone",
            value=str(session_state.agency_timezone),
            help="IANA timezone, for example America/Puerto_Rico or America/New_York.",
        )
        try:
            ZoneInfo(str(session_state.agency_timezone))
        except ZoneInfoNotFoundError:
            st.warning("Unknown IANA timezone. Monitoring will fail until this is corrected.")
        session_state.speed_limit_mph = st.number_input(
            "Posted speed limit (mph)", min_value=5.0, max_value=100.0,
            value=float(session_state.speed_limit_mph), step=5.0,
        )

        with st.expander("Model and calibration", expanded=False):
            model_options = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
            session_state.model_path = st.selectbox(
                "YOLO model", model_options,
                index=model_options.index(session_state.model_path) if session_state.model_path in model_options else 0,
            )
            sizes = [640, 1280, 1600]
            session_state.imgsz = st.selectbox(
                "Inference resolution", sizes,
                index=sizes.index(int(session_state.imgsz)) if int(session_state.imgsz) in sizes else 1,
            )
            session_state.calibration_path = st.text_input("Calibration JSON", value=str(session_state.calibration_path))

        with st.expander("Continuous archive", expanded=False):
            session_state.archive_enabled = st.checkbox("Enable archive", value=bool(session_state.archive_enabled))
            if session_state.archive_enabled:
                session_state.archive_dir = st.text_input("Archive directory", value=str(session_state.archive_dir))
                session_state.archive_segment_minutes = st.number_input(
                    "Minutes per segment", min_value=1.0,
                    value=float(session_state.archive_segment_minutes), step=1.0,
                )
                session_state.retention_days = st.number_input(
                    "Retention days", min_value=1, value=int(session_state.retention_days)
                )

        if not session_state.monitoring and st.button("Start monitoring", type="primary", width="stretch"):
            return "start"
    return None


def _date_range(days: int) -> tuple[float, float]:
    end = time.time()
    return end - max(1, days) * 86400, end


def render_overview_tab(session_state, tab, store: IncidentStore) -> None:
    with tab:
        st.subheader("Operations overview")
        summary = store.analytics_summary()
        cols = st.columns(4, gap="small")
        with cols[0]:
            summary_card("Pending review", str(summary["incidents_pending"]), "Human decisions still required")
        with cols[1]:
            summary_card("Approved incidents", str(summary["incidents_approved"]), "Confirmed by reviewers")
        with cols[2]:
            summary_card("Speeding rate", f"{summary['speeding_rate']:.1%}", "Samples above limit + 5 mph")
        with cols[3]:
            summary_card("Maximum observed speed", f"{summary['max_speed_mph']:.1f} mph" if summary["speed_samples"] else "—", f"{summary['speed_samples']:,} valid speed samples")

        left, right = st.columns([1.35, 1.0], gap="large")
        with left:
            st.markdown("### Incidents needing attention")
            pending = store.list_incidents(status="pending", limit=8)
            if not pending:
                st.markdown('<div class="ti-empty">No incidents are waiting for review.</div>', unsafe_allow_html=True)
            for item in pending:
                score = item.get("review_score") if item.get("review_score") is not None else item.get("detector_score")
                score_text = f"{float(score):.3f}" if score is not None else "—"
                st.markdown(
                    f"""
                    <div class="ti-incident pending">
                        <div class="ti-incident-title">{item['title']}</div>
                        <div class="ti-incident-meta">{item.get('location') or 'Unspecified location'} · {item.get('camera') or 'Unspecified camera'} · {format_datetime(item['detected_at'], session_state.agency_timezone)}</div>
                        <div class="ti-incident-meta">Priority: {item.get('priority','normal').title()} · Review score: {score_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with right:
            st.markdown("### Highest-risk locations")
            hotspots = store.hotspots(days=90)[:8]
            if hotspots:
                frame = pd.DataFrame(hotspots)[["location", "camera", "approved_incidents", "speeding_rate", "risk_index"]]
                frame.columns = ["Location", "Camera", "Approved incidents", "Speeding rate", "Risk index"]
                st.dataframe(frame, width="stretch", hide_index=True, column_config={"Speeding rate": st.column_config.NumberColumn(format="%.1%%")})
                mapped = pd.DataFrame([
                    {"lat": row["latitude"], "lon": row["longitude"], "risk_index": row["risk_index"]}
                    for row in hotspots
                    if row.get("latitude") is not None and row.get("longitude") is not None
                ])
                if not mapped.empty:
                    st.map(mapped, latitude="lat", longitude="lon", size="risk_index")
            else:
                st.markdown('<div class="ti-empty">Hotspot rankings appear after speed or incident data is collected.</div>', unsafe_allow_html=True)

        model = store.feedback_model()
        st.caption(
            f"Human-feedback model: {'active' if model.active_ else 'collecting labels'} · "
            f"{model.examples_} reviewed examples ({model.positives_} approved / {model.negatives_} dismissed)."
        )


def _incident_card(item: dict, timezone_name: str) -> None:
    status = str(item.get("status") or "pending")
    score = item.get("review_score") if item.get("review_score") is not None else item.get("detector_score")
    score_text = f"{float(score):.3f}" if score is not None else "—"
    stamp = format_datetime(float(item["detected_at"]), timezone_name)
    st.markdown(
        f"""
        <div class="ti-incident {status}">
            <div class="ti-incident-title">{item.get('title') or 'Incident'}</div>
            <div class="ti-incident-meta">{STATUS_LABELS.get(status, status.title())} · {item.get('priority','normal').title()} priority · score {score_text}</div>
            <div class="ti-incident-meta">{item.get('location') or 'Unspecified location'} · {item.get('camera') or 'Unspecified camera'} · {stamp}</div>
            <div class="ti-incident-meta">{item.get('description') or ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_incident_queue_tab(session_state, tab, store: IncidentStore) -> None:
    with tab:
        st.subheader("Incident review queue")
        cameras = store.list_cameras()
        locations = sorted({str(row.get("location") or "") for row in cameras if row.get("location")})
        camera_names = sorted({str(row.get("camera_name") or "") for row in cameras if row.get("camera_name")})
        f1, f2, f3 = st.columns([1.4, 1, 1])
        with f1:
            query = st.text_input("Search", placeholder="Location, camera, event ID, description")
        with f2:
            status = st.selectbox("Status", ["pending", "all", "approved", "dismissed", "needs_info"], format_func=lambda x: STATUS_LABELS.get(x, "All statuses"))
        with f3:
            window = st.selectbox("Time window", [7, 30, 90, 365, 0], index=2, format_func=lambda x: "All time" if x == 0 else f"Last {x} days")
        f4, f5, f6 = st.columns([1, 1, 1])
        with f4:
            location_filter = st.selectbox("Location", ["All"] + locations)
        with f5:
            camera_filter = st.selectbox("Camera", ["All"] + camera_names)
        with f6:
            limit = st.selectbox("Results", [25, 50, 100, 200], index=1)
        incidents = store.list_incidents(
            status=status, query=query,
            location=None if location_filter == "All" else location_filter,
            camera=None if camera_filter == "All" else camera_filter,
            start_time=None if window == 0 else time.time() - window * 86400,
            limit=limit,
        )
        if not incidents:
            st.markdown('<div class="ti-empty">No incidents match the current filters.</div>', unsafe_allow_html=True)
            return

        reviewer = st.text_input("Reviewer", value=str(session_state.reviewer_name), key="queue_reviewer")
        session_state.reviewer_name = reviewer
        for item in incidents:
            event_id = item["event_id"]
            with st.container(border=True):
                _incident_card(item, session_state.agency_timezone)
                left, right = st.columns([1.15, 1.0], gap="large")
                with left:
                    clip_value = str(item.get("clip_path") or "").strip()
                    clip = Path(clip_value) if clip_value else None
                    if clip is not None and clip.is_file():
                        st.video(str(clip))
                    else:
                        st.caption("The incident record exists, but its finalized clip is not available yet.")
                    with st.expander("Evidence details"):
                        st.json({
                            "event_id": event_id,
                            "detector_score": item.get("detector_score"),
                            "review_score": item.get("review_score"),
                            "tracks": item.get("involved_tracks"),
                            "evidence": item.get("evidence"),
                        })
                with right:
                    notes = st.text_area("Review notes", key=f"notes_{event_id}", placeholder="Optional context, disposition, or correction")
                    corrected = st.selectbox(
                        "Classification", ["collision", "near_miss", "hard_braking", "false_positive", "other"],
                        key=f"class_{event_id}",
                    )
                    c1, c2, c3 = st.columns(3)
                    if c1.button("Approve", key=f"approve_{event_id}", type="primary", width="stretch"):
                        store.review_incident(event_id, decision="approve", reviewer=reviewer, notes=notes, corrected_type=corrected)
                        store.refresh_review_scores()
                        st.rerun()
                    if c2.button("Dismiss", key=f"dismiss_{event_id}", width="stretch"):
                        store.review_incident(event_id, decision="dismiss", reviewer=reviewer, notes=notes, corrected_type=corrected)
                        store.refresh_review_scores()
                        st.rerun()
                    if c3.button("Needs info", key=f"info_{event_id}", width="stretch"):
                        store.review_incident(event_id, decision="needs_info", reviewer=reviewer, notes=notes, corrected_type=corrected)
                        st.rerun()
                    export_key = f"evidence_export_{event_id}"
                    if st.button("Build evidence package", key=f"build_{event_id}", width="stretch"):
                        session_state[export_key] = str(build_evidence_package(store, event_id))
                    export_value = str(session_state.get(export_key, "")).strip()
                    export_path = Path(export_value) if export_value else None
                    if export_path is not None and export_path.is_file():
                        with export_path.open("rb") as handle:
                            st.download_button(
                                "Download evidence ZIP", data=handle, file_name=export_path.name,
                                key=f"download_evidence_{event_id}", width="stretch",
                            )
                    reviews = store.reviews_for_incident(event_id)
                    if reviews:
                        latest = reviews[0]
                        st.caption(f"Latest review: {latest['decision']} by {latest['reviewer'] or 'unknown reviewer'}")


def render_analytics_tab(session_state, tab, store: IncidentStore) -> None:
    with tab:
        st.subheader("Roadway analytics")
        days = st.selectbox("Analysis window", [7, 30, 90, 180, 365], index=2, format_func=lambda x: f"Last {x} days")
        summary = store.analytics_summary(start_time=time.time() - days * 86400)
        cols = st.columns(4)
        cols[0].metric("Incident candidates", summary["incidents_total"])
        cols[1].metric("Approved incidents", summary["incidents_approved"])
        cols[2].metric("Average speed", f"{summary['avg_speed_mph']:.1f} mph" if summary["speed_samples"] else "—")
        cols[3].metric("Speeding rate", f"{summary['speeding_rate']:.1%}")

        day_rows = store.dangerous_days(days=days)
        hour_rows = store.dangerous_hours(days=days)
        hotspot_rows = store.hotspots(days=days)
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("### Most dangerous days by speeding behavior")
            if day_rows:
                df = pd.DataFrame(day_rows).set_index("weekday")
                st.bar_chart(df[["speeding_rate"]])
                st.dataframe(pd.DataFrame(day_rows), width="stretch", hide_index=True)
            else:
                st.caption("No valid speed observations yet.")
        with c2:
            st.markdown("### Speeding by hour")
            if hour_rows:
                df = pd.DataFrame(hour_rows).set_index("hour")
                st.line_chart(df[["speeding_rate", "avg_speed_mph"]])
                st.dataframe(pd.DataFrame(hour_rows), width="stretch", hide_index=True)
            else:
                st.caption("No valid speed observations yet.")

        st.markdown("### Hotspots")
        if hotspot_rows:
            st.dataframe(pd.DataFrame(hotspot_rows), width="stretch", hide_index=True)
            mapped = pd.DataFrame([
                {"lat": row["latitude"], "lon": row["longitude"], "risk_index": row["risk_index"]}
                for row in hotspot_rows
                if row.get("latitude") is not None and row.get("longitude") is not None
            ])
            if not mapped.empty:
                st.map(mapped, latitude="lat", longitude="lon", size="risk_index")
        else:
            st.caption("Hotspot rankings appear after observations have been collected.")
        st.caption("Risk index is an operational ranking based on reviewed incidents and speeding prevalence; it is not a predicted crash probability.")


def render_monitoring_tab(session_state, tab) -> None:
    with tab:
        if not session_state.monitoring or not session_state.runtime:
            st.info("Start monitoring from the sidebar to view the live camera feed.")
            return
        runtime = session_state.runtime
        cols = st.columns(4, gap="small")
        with cols[0]:
            summary_card("Vehicles in scene", f"{len(st.session_state.active_rows)}", "Current trusted tracks")
        with cols[1]:
            cal_value, cal_detail = calibration_status(runtime)
            summary_card("Calibration", cal_value, cal_detail)
        with cols[2]:
            max_speed = st.session_state.current_max_speed
            summary_card("Maximum speed", f"{max_speed:.0f} mph" if max_speed is not None else "—", "Current valid measurements")
        with cols[3]:
            avg_speed = st.session_state.current_avg_speed
            summary_card("Average speed", f"{avg_speed:.1f} mph" if avg_speed is not None else "—", f"Analysis: {st.session_state.analysis_fps:.1f} fps")
        if st.session_state.last_frame_rgb is not None:
            st.image(st.session_state.last_frame_rgb, width="stretch")
        if st.session_state.active_rows:
            st.markdown("### Active vehicles")
            st.dataframe(st.session_state.active_rows, width="stretch", hide_index=True)


def render_library_tab(session_state, tab, args) -> None:
    with tab:
        st.subheader("Evidence library")
        event_dir = str(session_state.event_dir or args.event_dir)
        archive_dir = str(session_state.archive_dir or args.archive_dir)
        library = scan_library(event_dir, archive_dir)
        if not library:
            st.markdown('<div class="ti-empty">No incident clips or archived recordings were found.</div>', unsafe_allow_html=True)
            return
        q = st.text_input("Filter evidence", placeholder="Location, camera, municipality")
        if q.strip():
            term = q.lower().strip()
            library = [item for item in library if term in " ".join(str(item.get(k, "")) for k in ("municipality", "location", "camera", "kind")).lower()]
        for idx, item in enumerate(library):
            with st.expander(f"{item['kind']} · {item['camera']} · {format_datetime(item['timestamp'], session_state.agency_timezone)}"):
                st.caption(f"{item.get('municipality')} · {item.get('location')} · duration {item.get('duration', 0):.1f}s")
                path = Path(item["path"])
                if path.exists():
                    st.video(str(path))
                    with path.open("rb") as handle:
                        st.download_button("Download video", data=handle, file_name=path.name, key=f"dl_{idx}_{path.name}")
                if item.get("sha256"):
                    st.code(item["sha256"], language=None)


def render_settings_tab(session_state, tab, store: IncidentStore) -> None:
    with tab:
        st.subheader("System settings")
        session_state.event_dir = st.text_input("Incident event directory", value=str(session_state.event_dir))
        session_state.db_path = st.text_input("Operations database", value=str(session_state.db_path))
        st.caption(f"Current database: {store.path}")
        model = store.feedback_model()
        st.markdown("### Human-feedback learning")
        st.write(
            "Approved and dismissed incidents are stored as supervised labels. The feedback model activates only after "
            f"at least {model.min_examples} usable reviews with at least {model.min_per_class} examples in each class."
        )
        st.metric("Reviewed training examples", model.examples_)
        if st.button("Recalculate review scores", width="content"):
            count = store.refresh_review_scores()
            st.success(f"Updated {count} incident review scores.")
        export_path = Path(session_state.db_path).with_name("feedback_export.jsonl")
        if st.button("Export feedback dataset", width="content"):
            path = store.export_feedback_jsonl(export_path)
            st.success(f"Exported labels to {path}")
        st.markdown("### Notifications")
        session_state.notification_webhook_url = st.text_input(
            "Webhook URL (optional)",
            value=str(session_state.notification_webhook_url),
            type="password",
            help="New incident candidates will be queued for this webhook. Delivery is explicit and auditable.",
        )
        webhook_queued = [item for item in store.queued_notifications(limit=100) if item.get("channel") == "webhook"]
        if webhook_queued and st.button("Send queued webhook notifications"):
            results = WebhookDispatcher(store).dispatch_queued(limit=100)
            sent = sum(result.status == "sent" for result in results)
            failed = sum(result.status == "failed" for result in results)
            st.info(f"Webhook delivery finished: {sent} sent, {failed} failed.")
        st.markdown("### Notification outbox")
        queued = store.queued_notifications(limit=100)
        st.metric("Queued notifications", len(queued))
        if queued:
            st.dataframe(pd.DataFrame(queued), width="stretch", hide_index=True)
            if st.button("Acknowledge queued in-app notifications"):
                for item in queued:
                    if item.get("channel") == "in_app":
                        store.mark_notification(int(item["notification_id"]), status="acknowledged")
                st.rerun()

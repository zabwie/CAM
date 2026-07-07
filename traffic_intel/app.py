"""
Streamlit dashboard for Traffic Intelligence System.

Shows:
  - Annotated video with speed overlay
  - Real-time traffic stats
  - Violation list
  - Speed distribution chart

Usage:
    streamlit run app.py -- --video <video> --calibration <calib.json>
"""

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

# Must be first page config
st.set_page_config(page_title="Traffic Intelligence", layout="wide")

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from engine import TrafficEngine, Calibration


# ---------------------------------------------------------------------------
# Parse CLI args (passed after --)
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="Path to traffic video")
    ap.add_argument("--calibration", default=None, help="Path to calibration JSON")
    ap.add_argument("--speed-limit", type=float, default=50, help="Speed limit km/h")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    return ap.parse_args()


ARGS = _parse_args()


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state():
    for k in ["engine", "summary", "processed", "violations_df", "annotated_video"]:
        if k not in st.session_state:
            st.session_state[k] = None


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Traffic Intelligence")
st.sidebar.caption("Phase 1 — Proof of Concept")

uploaded_video = st.sidebar.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])
uploaded_calib = st.sidebar.file_uploader("Upload calibration JSON", type=["json"])

speed_limit = st.sidebar.number_input("Speed limit (km/h)", value=50, step=5)
model_choice = st.sidebar.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
                                     index=0)

process_btn = st.sidebar.button("▶ Process Video", type="primary", use_container_width=True)

# -- Determine video & calib sources ----------------------------------------
video_source = None
calib_obj = None

if uploaded_video:
    # Save to temp file
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video.write(uploaded_video.read())
    video_source = tmp_video.name
elif ARGS.video and Path(ARGS.video).exists():
    video_source = ARGS.video

if uploaded_calib:
    calib_obj = Calibration.load(uploaded_calib)
elif ARGS.calibration and Path(ARGS.calibration).exists():
    calib_obj = Calibration.load(ARGS.calibration)

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("🚦 Traffic Intelligence System")
st.markdown("**Detection · Tracking · Speed Estimation · Violation Alerts**")

if not video_source:
    st.info("Upload a video file or specify `--video` on the command line.")
    st.stop()

# Show preview
col1, col2 = st.columns(2)
with col1:
    st.metric("Video", Path(video_source).name if not uploaded_video else "uploaded")
with col2:
    if calib_obj:
        st.metric("Calibration", "loaded ✅")
    else:
        st.metric("Calibration", "none — no speed estimates")

# -- Process ----------------------------------------------------------------
if process_btn:
    with st.spinner("Processing video…"):
        cal = calib_obj
        engine = TrafficEngine(
            model_path=model_choice,
            calibration=cal,
            speed_limit_kmh=speed_limit,
        )

        # Output annotated video
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        # Progress bar
        progress_bar = st.progress(0, text="Processing…")
        status_text = st.empty()

        def _progress(count, total):
            pct = min(count / max(total, 1), 1.0)
            progress_bar.progress(pct)
            status_text.text(f"Frame {count} / {total}")

        summary = engine.process_video(
            video_source, output_path=out_path,
            progress_callback=_progress,
        )

        # Violations
        viols = engine.get_violations()
        viol_df = pd.DataFrame([
            dict(Frame=v.frame, Track=v.track_id, Type=v.class_name,
                 Speed=f"{v.speed_kmh:.0f}" if v.speed_kmh else "")
            for v in viols
        ])

        st.session_state.engine = engine
        st.session_state.summary = summary
        st.session_state.processed = True
        st.session_state.violations_df = viol_df
        st.session_state.annotated_video = out_path

        progress_bar.empty()
        status_text.empty()

# -- Display results --------------------------------------------------------
if st.session_state.processed:
    engine = st.session_state.engine
    summary = st.session_state.summary

    # Summary cards
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1:
        st.metric("Vehicles Tracked", summary["unique_vehicles"])
    with sc2:
        st.metric("Violations", summary["violations"])
    with sc3:
        st.metric("Avg Speed", f'{summary["avg_speed_kmh"]} km/h')
    with sc4:
        st.metric("Max Speed", f'{summary["max_speed_kmh"]} km/h')
    with sc5:
        st.metric("Frames", summary["frames_processed"])

    # Annotated video
    if st.session_state.annotated_video:
        st.subheader("Annotated Video")
        st.video(st.session_state.annotated_video)

    # Violations table
    viol_df = st.session_state.violations_df
    if not viol_df.empty:
        st.subheader(f"⛔ Violations ({len(viol_df)} events)")
        st.dataframe(viol_df, use_container_width=True, hide_index=True)
    else:
        st.success("No speed violations detected.")

    # Speed distribution
    speeds = [d.speed_kmh for d in engine.results if d.speed_kmh is not None]
    if speeds:
        st.subheader("Speed Distribution")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.hist(speeds, bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
        ax.axvline(summary["speed_limit_kmh"], color="red", linestyle="--",
                    label=f"Limit {summary['speed_limit_kmh']} km/h")
        ax.set_xlabel("Speed (km/h)")
        ax.set_ylabel("Detections")
        ax.legend()
        st.pyplot(fig)

    # Traffic by class
    class_counts = pd.Series([d.class_name for d in engine.results]).value_counts()
    if not class_counts.empty:
        st.subheader("Vehicle Types")
        st.bar_chart(class_counts)

    # CSV download
    csv_buffer = tempfile.NamedTemporaryFile(delete=False, suffix=".csv").name
    engine.results_csv(csv_buffer)
    with open(csv_buffer) as f:
        st.sidebar.download_button(
            "📥 Download Results CSV",
            data=f,
            file_name="traffic_results.csv",
            mime="text/csv",
        )
    os.unlink(csv_buffer)

    # Cleanup tmp annotated video (on rerun)
    # (streamlit handles cleanup of its own temps)

else:
    st.info("Click **▶ Process Video** to start.")

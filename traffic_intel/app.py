"""
Streamlit dashboard for Traffic Intelligence System.

Shows:
  - Annotated video with detection overlay
  - Traffic stats
  - Vehicle type distribution

Usage:
    streamlit run app.py -- --video <video>
"""

import argparse
import csv
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

from engine import TrafficEngine, Calibration


# ---------------------------------------------------------------------------
# Parse CLI args (passed after --)
# ---------------------------------------------------------------------------

def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="Path to traffic video")
    ap.add_argument("--calibration", default=None, help="Path to calibration JSON")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO model path")
    return ap.parse_args()


ARGS = _parse_args()


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state():
    for k in ["engine", "summary", "processed", "annotated_video"]:
        if k not in st.session_state:
            st.session_state[k] = None


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("Traffic Intelligence")
st.sidebar.caption("Phase 1 — Proof of Concept")

uploaded_video = st.sidebar.file_uploader("Upload traffic video", type=["mp4", "avi", "mov"])

model_choice = st.sidebar.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
                                     index=0)

process_btn = st.sidebar.button("▶ Process Video", type="primary", use_container_width=True)

# -- Determine video source ------------------------------------------------
video_source = None

if uploaded_video:
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video.write(uploaded_video.read())
    video_source = tmp_video.name
elif ARGS.video and Path(ARGS.video).exists():
    video_source = ARGS.video

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

st.title("Traffic Intelligence System")
st.markdown("**Detection · Tracking**")

if not video_source:
    st.info("Upload a video file or specify `--video` on the command line.")
    st.stop()

# Show preview
col1, col2 = st.columns(2)
with col1:
    st.metric("Video", Path(video_source).name if not uploaded_video else "uploaded")

# -- Process ----------------------------------------------------------------
if process_btn:
    with st.spinner("Processing video…"):
        calib = None
        if ARGS.calibration:
            calib = Calibration.load(ARGS.calibration)

        engine = TrafficEngine(
            model_path=model_choice,
            calibration=calib,
        )

        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

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

        st.session_state.engine = engine
        st.session_state.summary = summary
        st.session_state.processed = True
        st.session_state.annotated_video = out_path

        progress_bar.empty()
        status_text.empty()

# -- Display results --------------------------------------------------------
if st.session_state.processed:
    engine = st.session_state.engine
    summary = st.session_state.summary

    # Summary cards
    speeds = [d.speed_mph for d in engine.results if d.speed_mph > 0]
    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.metric("Vehicles Tracked", summary["unique_vehicles"])
    with sc2:
        st.metric("Total Detections", summary["total_detections"])
    with sc3:
        st.metric("Frames", summary["frames_processed"])
    with sc4:
        avg_speed = f"{np.mean(speeds):.0f}" if speeds else "—"
        st.metric("Avg Speed (mph)", avg_speed)

    # Annotated video
    if st.session_state.annotated_video:
        st.subheader("Annotated Video")
        st.video(st.session_state.annotated_video)

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

else:
    st.info("Click **▶ Process Video** to start.")

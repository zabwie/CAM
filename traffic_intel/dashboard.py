"""Console launcher for the Streamlit operations dashboard."""

from __future__ import annotations

import os
import sys
from pathlib import Path


HELP = """Traffic Intelligence operations dashboard

Usage:
  traffic-intel-dashboard [dashboard application options]

Common options passed to the application:
  --camera SOURCE             Camera index, video path, or RTSP URL
  --calibration PATH          Calibration JSON
  --model PATH                YOLO model path
  --imgsz N                   Inference image size
  --event-dir PATH            Incident evidence directory
  --archive-dir PATH          Continuous archive directory
  --db-path PATH              Operations SQLite database
  --speed-limit-mph MPH       Posted speed limit for analytics

Example:
  traffic-intel-dashboard --camera 0 --calibration calib.json --speed-limit-mph 35
"""


def main() -> None:
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(HELP)
        return
    try:
        from streamlit.web import cli as stcli
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise SystemExit(
            "Dashboard dependencies are missing. Install with: pip install 'traffic-intel[dashboard]'"
        ) from exc

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    app_path = Path(__file__).with_name("app.py")
    application_args = sys.argv[1:]
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.headless=true",
        "--",
        *application_args,
    ]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()

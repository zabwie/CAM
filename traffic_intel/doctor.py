"""Environment and project readiness checks."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .calibration import Calibration


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Traffic Intelligence runtime readiness")
    parser.add_argument("--model", default="models/yolo11n.pt")
    parser.add_argument("--calibration", default=None)
    return parser.parse_args()


def main() -> None:
    args = _args()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python", sys.version_info >= (3, 10), platform.python_version()))
    checks.append(("OpenCV", True, cv2.__version__))
    checks.append(("NumPy", True, np.__version__))

    model = Path(args.model)
    checks.append(("Model", model.exists(), str(model.resolve()) if model.exists() else str(model)))

    if args.calibration:
        try:
            calibration = Calibration.load(args.calibration)
            valid = calibration.H is not None and bool(calibration.world_points)
            detail = f"{args.calibration} ({calibration.quality_grade})"
        except Exception as exc:  # operational CLI: report, do not hide failure
            valid = False
            detail = f"{args.calibration}: {exc}"
        checks.append(("Calibration", valid, detail))

    print(f"Traffic Intelligence v{__version__}")
    failures = 0
    for name, ok, detail in checks:
        status = "OK" if ok else "FAIL"
        print(f"[{status:4}] {name:12} {detail}")
        failures += int(not ok)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

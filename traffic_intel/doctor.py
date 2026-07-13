"""Environment and project readiness checks."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from traffic_intel.motion.calibration import Calibration


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Traffic Intelligence runtime readiness")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--calibration", default=None)
    return parser.parse_args()


def main() -> None:
    args = _args()
    checks: list[tuple[str, str, str]] = []
    checks.append(("Python", "OK" if sys.version_info >= (3, 10) else "FAIL", platform.python_version()))
    checks.append(("OpenCV", "OK", cv2.__version__))
    checks.append(("NumPy", "OK", np.__version__))

    model = Path(args.model)
    if model.exists():
        checks.append(("Model", "OK", str(model.resolve())))
    elif model.parent == Path(".") and model.suffix == ".pt":
        checks.append((
            "Model",
            "WARN",
            f"{model} not cached locally; Ultralytics may download it on first use",
        ))
    else:
        checks.append(("Model", "FAIL", f"missing local model: {model}"))

    if args.calibration:
        try:
            calibration = Calibration.load(args.calibration)
            valid = calibration.H is not None and bool(calibration.world_points)
            detail = f"{args.calibration} ({calibration.quality_grade})"
        except Exception as exc:  # operational CLI: report, do not hide failure
            valid = False
            detail = f"{args.calibration}: {exc}"
        checks.append(("Calibration", "OK" if valid else "FAIL", detail))

    print(f"Traffic Intelligence v{__version__}")
    failures = 0
    for name, status, detail in checks:
        print(f"[{status:4}] {name:12} {detail}")
        failures += int(status == "FAIL")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Traffic Intelligence package.

The package root intentionally avoids importing the full perception runtime at
module import time.  Lightweight consumers (domain models, calibration, crash
analysis, tests) should not require optional detector/tracker dependencies just
to import :mod:`traffic_intel`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from traffic_intel.config import EngineConfig, SceneChangeConfig, TrackingConfig
from traffic_intel.domain import Detection
from traffic_intel.motion.calibration import Calibration

__all__ = [
    "Calibration",
    "Detection",
    "EngineConfig",
    "SceneChangeConfig",
    "TrackingConfig",
    "TrafficEngine",
    "TrafficIncidentPipeline",
]

__version__ = "0.12.0"

_LAZY_EXPORTS = {
    "TrafficEngine": ("traffic_intel.core.engine", "TrafficEngine"),
    "TrafficIncidentPipeline": ("traffic_intel.core.pipeline", "TrafficIncidentPipeline"),
}


def __getattr__(name: str) -> Any:
    """Load heavyweight runtime exports only when they are actually requested."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

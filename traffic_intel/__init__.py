"""Traffic Intelligence package.

Heavy model/tracker dependencies are imported lazily so analytics/reporting
utilities can run on machines that do not have the inference stack installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analytics import VehiclePassage, VehiclePassageAggregator
    from .analytics_store import AnalyticsStore
    from .config import EngineConfig, SceneChangeConfig, TrackingConfig
    from .domain import Detection
    from .motion.calibration import Calibration
    from .core.engine import TrafficEngine
    from .core.pipeline import TrafficIncidentPipeline
    from .vision_quality import VisionQualityMonitor, VisionQualitySample

__all__ = [
    "AnalyticsStore",
    "Calibration",
    "Detection",
    "EngineConfig",
    "SceneChangeConfig",
    "TrackingConfig",
    "TrafficEngine",
    "TrafficIncidentPipeline",
    "VehiclePassage",
    "VehiclePassageAggregator",
    "VisionQualityMonitor",
    "VisionQualitySample",
]

__version__ = "0.12.0"

_EXPORTS = {
    "AnalyticsStore": (".analytics_store", "AnalyticsStore"),
    "Calibration": (".motion.calibration", "Calibration"),
    "Detection": (".domain", "Detection"),
    "EngineConfig": (".config", "EngineConfig"),
    "SceneChangeConfig": (".config", "SceneChangeConfig"),
    "TrackingConfig": (".config", "TrackingConfig"),
    "TrafficEngine": (".core.engine", "TrafficEngine"),
    "TrafficIncidentPipeline": (".core.pipeline", "TrafficIncidentPipeline"),
    "VehiclePassage": (".analytics", "VehiclePassage"),
    "VehiclePassageAggregator": (".analytics", "VehiclePassageAggregator"),
    "VisionQualityMonitor": (".vision_quality", "VisionQualityMonitor"),
    "VisionQualitySample": (".vision_quality", "VisionQualitySample"),
}


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

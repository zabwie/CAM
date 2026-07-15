"""Operational incident intelligence, review workflows, and analytics."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "FeedbackModel", "FeedbackPrediction", "IncidentStore",
    "build_evidence_package", "incident_features",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "FeedbackModel": ("traffic_intel.ops.feedback", "FeedbackModel"),
    "FeedbackPrediction": ("traffic_intel.ops.feedback", "FeedbackPrediction"),
    "incident_features": ("traffic_intel.ops.feedback", "incident_features"),
    "IncidentStore": ("traffic_intel.ops.store", "IncidentStore"),
    "build_evidence_package": ("traffic_intel.ops.evidence", "build_evidence_package"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

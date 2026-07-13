"""Domain models for the agency operations layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class IncidentStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    DISMISSED = "dismissed"
    NEEDS_INFO = "needs_info"


class ReviewDecision(StrEnum):
    APPROVE = "approve"
    DISMISS = "dismiss"
    NEEDS_INFO = "needs_info"


@dataclass(slots=True)
class Incident:
    event_id: str
    detected_at: float
    trigger_type: str
    title: str
    municipality: str = ""
    location: str = ""
    camera: str = ""
    source: str = ""
    status: str = IncidentStatus.PENDING.value
    detector_score: float | None = None
    review_score: float | None = None
    priority: str = "normal"
    involved_tracks: list[int] = field(default_factory=list)
    description: str = ""
    clip_path: str = ""
    metadata_path: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0


@dataclass(slots=True)
class Review:
    review_id: int
    event_id: str
    decision: str
    reviewer: str
    notes: str
    corrected_type: str
    created_at: float

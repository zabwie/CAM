"""Typed runtime configuration for the traffic-intelligence pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from .speed import SpeedEstimatorConfig


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    detector_confidence: float = 0.10
    activation_threshold: float = 0.30
    minimum_matching_threshold: float = 0.80
    lost_track_seconds: float = 1.0
    min_hits: int = 3
    min_confidence: float = 0.28
    vehicle_nms_threshold: float = 0.55
    max_normalized_jump: float = 2.5
    max_size_ratio_step: float = 2.2
    reacquire_hits: int = 2
    stale_seconds: float = 2.0

    # Canonical identity / short-gap re-identification.  Raw ByteTrack IDs are
    # treated as ephemeral association handles; these settings control when a
    # new raw ID may inherit the same physical-vehicle identity.
    identity_stitch_seconds: float = 1.50
    identity_min_stitch_gap_seconds: float = 0.10
    identity_provisional_hits: int = 4
    identity_state_seconds: float = 2.50
    identity_min_stitch_score: float = 0.58
    identity_ambiguity_margin: float = 0.08
    identity_max_prediction_error: float = 1.35
    identity_max_size_ratio: float = 4.50
    identity_min_appearance: float = 0.24
    identity_large_scale_ratio: float = 2.50
    identity_large_scale_min_appearance: float = 0.68
    identity_static_speed_norm: float = 0.012
    identity_static_relocation_error: float = 0.25
    identity_static_min_appearance: float = 0.65

    # Existing raw-ID hijack protection.  These are intentionally conservative
    # so a genuine collision-induced motion impulse does not split an identity.
    identity_hijack_max_jump: float = 3.25
    identity_hijack_max_size_ratio: float = 3.60
    identity_hijack_appearance_jump: float = 1.45
    identity_hijack_min_appearance: float = 0.16

    # Lightweight display/anchor filtering.  Raw boxes remain available to
    # incident analyzers; filtered boxes are used for stable UI and road anchors.
    identity_position_alpha: float = 0.64
    identity_size_alpha: float = 0.46
    identity_velocity_alpha: float = 0.24


@dataclass(frozen=True, slots=True)
class SceneChangeConfig:
    enabled: bool = True
    width: int = 160
    height: int = 90
    pixel_delta_threshold: int = 32
    mean_delta_threshold: float = 15.0
    changed_fraction_threshold: float = 0.15
    warmup_seconds: float = 0.35


@dataclass(frozen=True, slots=True)
class EngineConfig:
    model_path: str = "yolo11n.pt"
    imgsz: int = 1280
    fps: float = 30.0
    retain_history: bool = True
    min_bbox_height_for_speed: int = 12
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    scene_change: SceneChangeConfig = field(default_factory=SceneChangeConfig)
    speed: SpeedEstimatorConfig = field(default_factory=SpeedEstimatorConfig)

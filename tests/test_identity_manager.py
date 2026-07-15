from __future__ import annotations

import numpy as np

from traffic_intel.config import TrackingConfig
from traffic_intel.core.identity import CanonicalIdentityManager, RawTrackObservation
from traffic_intel.core.tracking import TrackQualityGate


def _obs(raw_id: int, x: float, y: float, *, w: float = 80, h: float = 48) -> RawTrackObservation:
    return RawTrackObservation(
        tracker_id=raw_id,
        class_id=2,
        confidence=0.90,
        bbox=np.array([x, y, x + w, y + h], dtype=np.float32),
    )


def test_new_raw_tracker_id_is_stitched_to_same_canonical_vehicle() -> None:
    cfg = TrackingConfig(identity_min_stitch_score=0.50)
    manager = CanonicalIdentityManager(fps=30, config=cfg)

    canonical = None
    for frame in range(1, 11):
        assignment = manager.assign_batch(
            frame=frame,
            observations=[_obs(40, 100 + 5 * frame, 220)],
        )[0]
        canonical = assignment.canonical_id

    # Three missing frames, then ByteTrack returns the same physical car under
    # a completely different raw tracker ID.
    assignment = manager.assign_batch(
        frame=14,
        observations=[_obs(63, 100 + 5 * 14, 220)],
    )[0]

    assert assignment.canonical_id == canonical
    assert assignment.tracker_id == 63
    assert assignment.generation == 2
    assert assignment.lifecycle == "STITCHED"
    assert manager.stitch_events[-1]["old_tracker_id"] == 40
    assert manager.stitch_events[-1]["new_tracker_id"] == 63


def test_ambiguous_reentry_is_not_force_stitched() -> None:
    cfg = TrackingConfig(
        identity_min_stitch_score=0.45,
        identity_ambiguity_margin=0.12,
    )
    manager = CanonicalIdentityManager(fps=30, config=cfg)

    a = manager.assign_batch(frame=1, observations=[_obs(1, 100, 200)])[0]
    b = manager.assign_batch(frame=1, observations=[_obs(2, 220, 200)])[0]
    assert a.canonical_id != b.canonical_id

    # The new observation falls nearly midway between two equally plausible
    # dormant vehicles.  Conservative behavior is to create a new identity.
    c = manager.assign_batch(frame=5, observations=[_obs(99, 160, 200)])[0]
    assert c.canonical_id not in {a.canonical_id, b.canonical_id}
    assert c.lifecycle == "NEW"


def test_raw_id_hijack_breaks_identity_instead_of_reusing_history() -> None:
    manager = CanonicalIdentityManager(fps=30)
    first = None
    for frame in range(1, 8):
        first = manager.assign_batch(
            frame=frame,
            observations=[_obs(7, 100 + frame, 200)],
        )[0]

    hijacked = manager.assign_batch(
        frame=8,
        observations=[_obs(7, 900, 80, w=180, h=110)],
    )[0]

    assert hijacked.canonical_id != first.canonical_id
    assert hijacked.discontinuity
    assert hijacked.lifecycle == "NEW_AFTER_BREAK"
    assert manager.discontinuity_events


def test_track_quality_remains_confirmed_after_strong_canonical_stitch() -> None:
    cfg = TrackingConfig(min_hits=3, min_confidence=0.20, identity_min_stitch_score=0.50)
    manager = CanonicalIdentityManager(fps=30, config=cfg)
    gate = TrackQualityGate(fps=30, config=cfg)

    canonical_id = None
    for frame in range(1, 8):
        identity = manager.assign_batch(
            frame=frame,
            observations=[_obs(40, 100 + 4 * frame, 220)],
        )[0]
        canonical_id = identity.canonical_id
        assessment = gate.update(
            frame=frame,
            track_id=canonical_id,
            class_id=2,
            confidence=0.90,
            bbox=identity.raw_bbox,
            reidentified=False,
            identity_confidence=identity.identity_confidence,
        )
    assert assessment.confirmed

    identity = manager.assign_batch(
        frame=12,
        observations=[_obs(63, 100 + 4 * 12, 220)],
    )[0]
    assessment = gate.update(
        frame=12,
        track_id=identity.canonical_id,
        class_id=2,
        confidence=0.90,
        bbox=identity.raw_bbox,
        reidentified=True,
        identity_confidence=identity.identity_confidence,
    )

    assert identity.canonical_id == canonical_id
    assert assessment.confirmed


def test_static_false_track_does_not_absorb_relocated_moving_vehicle() -> None:
    """A near-static foreground blob must not donate its identity to a new car."""
    cfg = TrackingConfig(
        identity_min_stitch_score=0.45,
        identity_provisional_hits=3,
        identity_static_speed_norm=0.02,
        identity_static_relocation_error=0.20,
        identity_static_min_appearance=0.65,
    )
    manager = CanonicalIdentityManager(fps=30, config=cfg)

    old = None
    for frame in range(1, 9):
        old = manager.assign_batch(
            frame=frame,
            observations=[_obs(11, 500, 420, w=260, h=180)],
        )[0]

    # A different, much smaller vehicle appears nearby after the static blob
    # disappears.  Without strong appearance evidence, position alone is not
    # sufficient to inherit the old canonical identity.
    new = manager.assign_batch(
        frame=13,
        observations=[_obs(15, 610, 485, w=110, h=70)],
    )[0]

    assert new.canonical_id != old.canonical_id
    assert new.lifecycle == "NEW"


def test_provisional_identity_re_evaluated_against_dormant() -> None:
    """A young provisional identity is held open for re-evaluation against older
    dormant vehicles rather than being immediately accepted."""
    cfg = TrackingConfig(
        identity_min_stitch_score=0.58,
        identity_provisional_hits=4,
        identity_min_stitch_gap_seconds=0.10,
    )
    manager = CanonicalIdentityManager(fps=30, config=cfg)

    # Dormant vehicle established at frame 1.
    dormant = manager.assign_batch(frame=1, observations=[_obs(10, 200, 220)])[0]

    # Frame 3: new raw ID 20 appears far from the dormant.  Gap = 2 frames,
    # which is below min_stitch_gap (3 frames at 30 FPS), so stitch is
    # blocked.  A separate provisional identity is created.
    provisional = manager.assign_batch(frame=3, observations=[_obs(20, 100, 100)])[0]
    assert provisional.canonical_id != dormant.canonical_id
    assert provisional.lifecycle == "NEW"

    # Frame 6: same raw ID 20 returns at the far position.  The provisional
    # identity is still young (< 4 hits) so it enters the unmatched/re-evaluation
    # path in _preserve_mappings.  The dormant stitch attempt is gated by
    # _stitch_score — the prediction error from (200,220) → (100,100) over
    # 5 frames exceeds max_prediction_error, so it stays separate.
    continued = manager.assign_batch(frame=6, observations=[_obs(20, 100, 100)])[0]
    assert continued.canonical_id == provisional.canonical_id
    assert continued.lifecycle in ("CONTINUING", "NEW")

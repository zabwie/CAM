"""Human-feedback calibration for incident triage.

This module intentionally does *not* mutate the production crash detector online.
Human review creates supervised labels.  A small regularized logistic model may
then learn recurring false-positive/true-positive patterns and produce a second
"review score" used for queue ordering.  Raw evidence remains preserved.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


FEATURE_NAMES = (
    "detector_score",
    "contact",
    "interaction_risk",
    "kinematic",
    "flow",
    "aftermath",
    "gap_norm",
    "disc_a",
    "disc_b",
    "depth_consistency",
    "anchor_distance",
    "pair_relative_dv",
    "delta_v_cosine",
    "pre_heading_cosine",
    "common_braking",
    "merge_area_ratio",
    "merge_width_ratio",
    "lost_vehicle_confidence",
    "lost_vehicle_persistence",
    "lost_vehicle_proximity",
    "merged_box_absorption",
    "missing_frames",
    "is_merge_path",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def incident_features(
    detector_score: float | None,
    evidence: dict[str, Any] | str | None,
    trigger_type: str = "",
) -> np.ndarray:
    """Create a stable numeric feature vector from stored incident evidence."""
    if isinstance(evidence, str):
        try:
            evidence = json.loads(evidence)
        except json.JSONDecodeError:
            evidence = {}
    evidence = dict(evidence or {})
    values = {
        "detector_score": _finite(detector_score),
        "is_merge_path": float("merge" in str(trigger_type).lower()),
    }
    for name in FEATURE_NAMES:
        if name not in values:
            values[name] = _finite(evidence.get(name))
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FeedbackPrediction:
    probability: float
    active: bool
    training_examples: int
    reason: str


class FeedbackModel:
    """Regularized logistic calibration model trained from reviewed incidents.

    Activation is gated by minimum data volume and class balance.  Until those
    gates are met, the base detector score is returned unchanged.
    """

    def __init__(
        self,
        *,
        min_examples: int = 20,
        min_per_class: int = 5,
        l2: float = 0.08,
        learning_rate: float = 0.08,
        epochs: int = 500,
    ) -> None:
        self.min_examples = int(min_examples)
        self.min_per_class = int(min_per_class)
        self.l2 = float(l2)
        self.learning_rate = float(learning_rate)
        self.epochs = int(epochs)
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.examples_: int = 0
        self.positives_: int = 0
        self.negatives_: int = 0
        self.active_: bool = False

    def fit(self, rows: Iterable[dict[str, Any]]) -> "FeedbackModel":
        features: list[np.ndarray] = []
        labels: list[float] = []
        for row in rows:
            decision = str(row.get("decision", "")).lower()
            if decision not in {"approve", "dismiss"}:
                continue
            features.append(incident_features(
                row.get("detector_score"), row.get("evidence"), row.get("trigger_type", "")
            ))
            labels.append(1.0 if decision == "approve" else 0.0)

        self.examples_ = len(labels)
        self.positives_ = int(sum(labels))
        self.negatives_ = self.examples_ - self.positives_
        self.active_ = (
            self.examples_ >= self.min_examples
            and self.positives_ >= self.min_per_class
            and self.negatives_ >= self.min_per_class
        )
        if not self.active_:
            self.mean_ = self.scale_ = self.weights_ = None
            return self

        x = np.vstack(features)
        y = np.asarray(labels, dtype=np.float64)
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        z = (x - self.mean_) / self.scale_
        z = np.column_stack([np.ones(len(z)), z])
        w = np.zeros(z.shape[1], dtype=np.float64)

        # Reweight the minority class so a city's early review imbalance does not
        # teach the model to predict only the majority decision.
        pos_weight = self.examples_ / max(2.0 * self.positives_, 1.0)
        neg_weight = self.examples_ / max(2.0 * self.negatives_, 1.0)
        sample_weight = np.where(y > 0.5, pos_weight, neg_weight)

        for _ in range(self.epochs):
            logits = np.clip(z @ w, -30.0, 30.0)
            pred = 1.0 / (1.0 + np.exp(-logits))
            grad = (z.T @ ((pred - y) * sample_weight)) / len(z)
            grad[1:] += self.l2 * w[1:]
            w -= self.learning_rate * grad
        self.weights_ = w
        return self

    def predict(
        self,
        *,
        detector_score: float | None,
        evidence: dict[str, Any] | str | None,
        trigger_type: str = "",
    ) -> FeedbackPrediction:
        base = min(1.0, max(0.0, _finite(detector_score, 0.5)))
        if not self.active_ or self.mean_ is None or self.scale_ is None or self.weights_ is None:
            return FeedbackPrediction(
                probability=base,
                active=False,
                training_examples=self.examples_,
                reason="insufficient_review_labels",
            )
        x = incident_features(detector_score, evidence, trigger_type)
        z = (x - self.mean_) / self.scale_
        z = np.concatenate([[1.0], z])
        learned = float(1.0 / (1.0 + np.exp(-np.clip(z @ self.weights_, -30.0, 30.0))))
        # Keep the calibrated score anchored to the detector until a much larger
        # review corpus exists.  This avoids a small local dataset dominating.
        blend = min(0.70, 0.35 + 0.01 * max(0, self.examples_ - self.min_examples))
        probability = (1.0 - blend) * base + blend * learned
        return FeedbackPrediction(
            probability=float(np.clip(probability, 0.0, 1.0)),
            active=True,
            training_examples=self.examples_,
            reason="human_feedback_calibration",
        )

"""Out-of-distribution (OOD) scoring for live gaze feature vectors.

Why this exists: leave-person-out evaluation showed the trained regressor's
worst failures are not evenly-distributed noise -- they concentrate in
specific people/sessions where input features (head_tx for one participant,
pitch for another) fall outside the training distribution, and the model
extrapolates badly rather than failing loudly (see investigation notes,
Lab ML writeup). This module scores how "unlike training" a feature vector
is, so the caller can hold/suppress a prediction instead of acting on a
likely-catastrophic one. Convention: HIGHER score = MORE anomalous.

Three methods, same (fit(X), score(X)) interface, see scripts/evaluate_ood.py
for which one actually wins on this project's data and why.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

METHODS = ("mahalanobis", "isolation_forest", "lof")


class OODDetector(Protocol):
    def score(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class MahalanobisDetector:
    """Distance from the training mean, scaled by the training covariance.
    Cheapest method: fit is closed-form, score is one matrix multiply.
    Assumes the training features form roughly one ellipsoidal cloud --
    weak assumption once multiple participants/setups are pooled (see
    scripts/evaluate_ood.py results)."""

    mean_: np.ndarray
    inv_cov_: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, reg: float = 1e-3) -> "MahalanobisDetector":
        mean = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov = cov + reg * np.eye(cov.shape[0])  # regularize: avoid singular cov on near-constant features
        return cls(mean, np.linalg.inv(cov))

    def score(self, X: np.ndarray) -> np.ndarray:
        d = X - self.mean_
        return np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", d, self.inv_cov_, d), 0.0))


class IsolationForestDetector:
    """Tree-based, no distributional assumption -- isolates points that
    split off from the rest of the data in few random cuts. Handles
    multi-modal data (e.g. two participants' setups forming separate
    clusters) better than Mahalanobis; cheap enough at inference (traverse
    a small forest) for a live per-frame loop."""

    def __init__(self, model):
        self._model = model

    @classmethod
    def fit(cls, X: np.ndarray, **kwargs) -> "IsolationForestDetector":
        from sklearn.ensemble import IsolationForest

        params = dict(n_estimators=200, contamination="auto", random_state=0)
        params.update(kwargs)
        return cls(IsolationForest(**params).fit(X))

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(X)  # sklearn: lower score_samples = more anomalous; flip sign


class LOFDetector:
    """Density-based: compares a point's local density to its neighbours'.
    Best at catching person-specific pockets of weirdness (matches the
    'different feature dominates per person' finding), but needs a
    nearest-neighbour query against the full training set per live frame --
    more memory/latency than the other two at this data scale (150k+ rows)."""

    def __init__(self, model):
        self._model = model

    @classmethod
    def fit(cls, X: np.ndarray, n_neighbors: int = 35, **kwargs) -> "LOFDetector":
        from sklearn.neighbors import LocalOutlierFactor

        return cls(LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, **kwargs).fit(X))

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(X)


_FACTORIES = {
    "mahalanobis": MahalanobisDetector.fit,
    "isolation_forest": IsolationForestDetector.fit,
    "lof": LOFDetector.fit,
}


def fit_detector(method: str, X: np.ndarray, **kwargs) -> OODDetector:
    if method not in _FACTORIES:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    return _FACTORIES[method](X, **kwargs)


@dataclass
class GazeConfidenceGate:
    """Live decision policy sitting between a raw prediction and the
    smoother: trust / hold-last / suppress, based on an OOD score vs a
    fitted threshold. Mirrors AttentionGate's existing pause pattern
    (fixation/attention_gate.py) rather than inventing a new idiom.

    - score <= threshold            -> "trust": feed the smoother as normal.
    - score > threshold, but the
      last trusted frame was recent  -> "hold": reuse the last trusted point,
      (<= max_hold_ms)                  do NOT feed the raw OOD point to the
                                         Kalman smoother (an OOD point can
                                         itself corrupt the smoother's
                                         velocity state -- see smoothing.py
                                         investigation notes).
    - score > threshold, held too
      long                           -> "suppress": same as face-lost; the
                                         dwell trigger pauses rather than
                                         act on a stale or OOD position.
    """

    threshold: float
    max_hold_ms: float = 1500.0
    _last_trusted_point: np.ndarray | None = None
    _last_trusted_ms: float | None = None

    def decide(self, score: float, point: np.ndarray, timestamp_ms: float) -> tuple[str, np.ndarray | None]:
        """-> (decision, point_to_use). point_to_use is None for 'suppress'."""
        if score <= self.threshold:
            self._last_trusted_point = point
            self._last_trusted_ms = timestamp_ms
            return "trust", point
        if self._last_trusted_ms is not None and (timestamp_ms - self._last_trusted_ms) <= self.max_hold_ms:
            return "hold", self._last_trusted_point
        return "suppress", None

    def reset(self) -> None:
        self._last_trusted_point = None
        self._last_trusted_ms = None

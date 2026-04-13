"""
Layer 2: HDBSCAN density clustering and temporal recurrence detection.
Implements Section 4.4 of the paper (Equations 2, 3).
"""
import numpy as np
import hdbscan
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from app.config import get_settings

settings = get_settings()


@dataclass
class RecurrenceResult:
    is_recurring: bool
    strength: float
    pattern: str  # 'weekly' | 'monthly' | 'none'
    mean_interval: float = 0.0
    std_interval: float = 0.0


def detect_recurrence(timestamps: list) -> RecurrenceResult:
    """
    Equation 2: Classify a merchant as recurring if:
    N >= 2 AND |mean_interval - expected| < 7 days AND std < 7 days
    """
    if len(timestamps) < 2:
        return RecurrenceResult(False, 0.0, 'none')

    sorted_ts = sorted(timestamps)
    intervals = [
        (sorted_ts[i+1] - sorted_ts[i]).days
        for i in range(len(sorted_ts) - 1)
    ]

    mean_interval = float(np.mean(intervals))
    std_interval = float(np.std(intervals))

    tolerance = 7  # days

    for expected, pattern in [(30, 'monthly'), (7, 'weekly')]:
        if abs(mean_interval - expected) < tolerance and std_interval < tolerance:
            within_tol = sum(
                1 for iv in intervals
                if abs(iv - expected) < tolerance
            )
            strength = within_tol / len(intervals)
            return RecurrenceResult(
                is_recurring=True,
                strength=round(strength, 3),
                pattern=pattern,
                mean_interval=mean_interval,
                std_interval=std_interval,
            )

    return RecurrenceResult(False, 0.0, 'none', mean_interval, std_interval)


class HDBSCANClusterer:
    """Manages HDBSCAN clustering over a user's transaction feature vectors."""

    def __init__(self):
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=settings.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=settings.HDBSCAN_MIN_SAMPLES,
            metric='euclidean',
            prediction_data=True
        )
        self.is_fitted = False

    def fit(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Fit HDBSCAN. Returns cluster labels (-1 = noise)."""
        if len(feature_matrix) < settings.HDBSCAN_MIN_CLUSTER_SIZE:
            return np.full(len(feature_matrix), -1)
        self.clusterer.fit(feature_matrix)
        self.is_fitted = True
        return self.clusterer.labels_

    def get_soft_membership(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Get soft membership probabilities (cluster stability scores)."""
        if not self.is_fitted:
            raise RuntimeError("Clusterer must be fitted before predicting membership.")
        soft_clusters = hdbscan.membership_vector(self.clusterer, feature_matrix)
        return soft_clusters

    def predict_single(self, feature_vector: np.ndarray) -> tuple:
        """Predict cluster and soft membership for a single new transaction."""
        if not self.is_fitted:
            return -1, 0.0
        try:
            labels, strengths = hdbscan.approximate_predict(
                self.clusterer, feature_vector.reshape(1, -1)
            )
            return int(labels[0]), float(strengths[0])
        except Exception:
            return -1, 0.0


def compute_behavioural_confidence(
    recurrence_strength: float,
    cluster_stability: float,
    alpha_rec: float = 0.55
) -> float:
    """Equation 3: C_beh = alpha_rec * S_rec + (1 - alpha_rec) * S_cluster"""
    return round(
        alpha_rec * recurrence_strength + (1 - alpha_rec) * cluster_stability, 4
    )

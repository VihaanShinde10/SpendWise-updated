"""
Layer 2: HDBSCAN density clustering and temporal recurrence detection.
Implements Section 4.4 of the paper (Equations 2, 3).
"""
import logging
import numpy as np
import hdbscan
from datetime import datetime
from typing import Optional
from dataclasses import dataclass
from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Minimum transactions needed for a meaningful recurrence std estimate.
# With N=2 there is exactly one interval, so std=0 always passes any
# tolerance check — that is a false positive, not a recurrence signal.
_MIN_RECURRENCE_SAMPLES = 3


@dataclass
class RecurrenceResult:
    is_recurring: bool
    strength: float
    pattern: str        # 'weekly' | 'monthly' | 'none'
    mean_interval: float = 0.0
    std_interval: float = 0.0


def detect_recurrence(timestamps: list) -> RecurrenceResult:
    """
    Equation 2: Classify a merchant as recurring if:
      N >= 3  AND  |mean_interval - expected| < 7 days  AND  std < 7 days

    Parameters
    ----------
    timestamps : list of datetime objects (timezone-aware or all naive —
                 must not mix aware and naive).

    Returns
    -------
    RecurrenceResult with is_recurring, strength, pattern, mean_interval,
    std_interval populated.
    """
    if not isinstance(timestamps, (list, tuple)):
        raise TypeError(
            f"timestamps must be a list or tuple, got {type(timestamps).__name__}"
        )

    # Need at least _MIN_RECURRENCE_SAMPLES to compute a meaningful std
    if len(timestamps) < _MIN_RECURRENCE_SAMPLES:
        return RecurrenceResult(False, 0.0, 'none')

    # Validate every element is a datetime
    for i, ts in enumerate(timestamps):
        if not isinstance(ts, datetime):
            raise TypeError(
                f"timestamps[{i}] must be a datetime, got {type(ts).__name__!r}"
            )

    sorted_ts = sorted(timestamps)

    # Compute intervals; guard against negative values caused by clock skew
    # or mixed-timezone inputs (e.g. aware vs naive).
    raw_intervals = [
        (sorted_ts[i + 1] - sorted_ts[i]).days
        for i in range(len(sorted_ts) - 1)
    ]

    # Any negative interval means timestamps are mis-ordered after sort,
    # which indicates a timezone inconsistency — fail fast.
    if any(iv < 0 for iv in raw_intervals):
        logger.warning(
            "detect_recurrence: negative interval(s) detected after sorting "
            "(%s). Likely mixed-timezone datetimes. Returning non-recurring.",
            [iv for iv in raw_intervals if iv < 0],
        )
        return RecurrenceResult(False, 0.0, 'none')

    intervals = raw_intervals
    mean_interval = float(np.mean(intervals))
    std_interval  = float(np.std(intervals, ddof=0))   # population std (N-1 intervals)

    tolerance = 7  # days

    for expected, pattern in [(30, 'monthly'), (7, 'weekly')]:
        if abs(mean_interval - expected) < tolerance and std_interval < tolerance:
            within_tol = sum(
                1 for iv in intervals if abs(iv - expected) < tolerance
            )
            # strength = fraction of intervals that conform to the pattern
            # len(intervals) == N - 1, which is the correct denominator here.
            strength = within_tol / len(intervals)
            return RecurrenceResult(
                is_recurring=True,
                strength=round(strength, 3),
                pattern=pattern,
                mean_interval=round(mean_interval, 4),
                std_interval=round(std_interval, 4),
            )

    return RecurrenceResult(
        is_recurring=False,
        strength=0.0,
        pattern='none',
        mean_interval=round(mean_interval, 4),
        std_interval=round(std_interval, 4),
    )


class HDBSCANClusterer:
    """Manages HDBSCAN clustering over a user's transaction feature vectors."""

    def __init__(self):
        # prediction_data=True is mandatory — approximate_predict raises a
        # hard RuntimeError without it and the message is not obvious.
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=settings.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=settings.HDBSCAN_MIN_SAMPLES,
            metric='euclidean',
            prediction_data=True,
        )
        self.is_fitted     = False
        self.insufficient_data = False   # True when fit() skipped due to too few rows

    def fit(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Fit HDBSCAN on feature_matrix.

        Returns cluster labels array (-1 = noise).
        Sets self.insufficient_data = True (and returns all-noise labels)
        when the matrix has fewer rows than min_cluster_size — the caller
        can check this flag to distinguish "not yet fitted" from
        "fitted but data was too sparse".
        """
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)

        if feature_matrix.ndim != 2:
            raise ValueError(
                f"feature_matrix must be 2-D, got shape {feature_matrix.shape}"
            )

        if len(feature_matrix) < settings.HDBSCAN_MIN_CLUSTER_SIZE:
            logger.debug(
                "HDBSCANClusterer.fit: only %d rows (min_cluster_size=%d) — "
                "skipping fit, returning all-noise labels.",
                len(feature_matrix), settings.HDBSCAN_MIN_CLUSTER_SIZE,
            )
            self.insufficient_data = True
            return np.full(len(feature_matrix), -1, dtype=np.int32)

        self.clusterer.fit(feature_matrix)
        self.is_fitted         = True
        self.insufficient_data = False
        return self.clusterer.labels_

    def get_soft_membership(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Get soft membership probabilities for every point in feature_matrix.

        Uses hdbscan.all_points_membership_vectors — the correct API for a
        batch of points.  hdbscan.membership_vector (singular) is for a
        single vector and raises or silently truncates on a 2-D input.

        Returns a 2-D array of shape (n_points, n_clusters).
        """
        if not self.is_fitted:
            raise RuntimeError(
                "HDBSCANClusterer must be fitted before calling get_soft_membership."
            )
        feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
        if feature_matrix.ndim != 2:
            raise ValueError(
                f"feature_matrix must be 2-D, got shape {feature_matrix.shape}"
            )
        return hdbscan.all_points_membership_vectors(self.clusterer)

    def predict_single(self, feature_vector: np.ndarray) -> tuple:
        """
        Predict cluster label and soft-membership strength for one new transaction.

        Returns
        -------
        (cluster_label: int, strength: float)
        cluster_label == -1 means noise / unassigned.

        Raises
        ------
        ValueError  : if feature_vector has the wrong shape.
        RuntimeError: if the clusterer has not been fitted (is_fitted is False).
        """
        if not self.is_fitted:
            # Caller must check is_fitted / insufficient_data before calling.
            logger.debug(
                "predict_single called before clusterer is fitted — returning noise."
            )
            return -1, 0.0

        feature_vector = np.asarray(feature_vector, dtype=np.float32).ravel()
        expected_dim = self.clusterer.min_samples  # not the right dim check —
        # Derive expected dimension from the fitted exemplars instead.
        trained_dim = self.clusterer._raw_data.shape[1]
        if feature_vector.shape[0] != trained_dim:
            raise ValueError(
                f"feature_vector has {feature_vector.shape[0]} dims but "
                f"clusterer was trained on {trained_dim}-dim vectors"
            )

        try:
            labels, strengths = hdbscan.approximate_predict(
                self.clusterer, feature_vector.reshape(1, -1)
            )
            return int(labels[0]), float(strengths[0])
        except Exception as exc:
            # Log the real error instead of silently discarding it.
            logger.warning(
                "approximate_predict failed: %s — returning noise label.", exc
            )
            return -1, 0.0


def compute_behavioural_confidence(
    recurrence_strength: float,
    cluster_stability: float,
    alpha_rec: float = 0.55,
) -> float:
    """
    Equation 3: C_beh = alpha_rec * S_rec + (1 - alpha_rec) * S_cluster

    Parameters
    ----------
    recurrence_strength : float in [0, 1]
    cluster_stability   : float in [0, 1]
    alpha_rec           : mixing weight, must be in [0, 1]
    """
    if not (0.0 <= alpha_rec <= 1.0):
        raise ValueError(
            f"alpha_rec must be in [0, 1], got {alpha_rec!r}"
        )
    recurrence_strength = float(np.clip(recurrence_strength, 0.0, 1.0))
    cluster_stability   = float(np.clip(cluster_stability,   0.0, 1.0))

    c_beh = alpha_rec * recurrence_strength + (1.0 - alpha_rec) * cluster_stability
    return round(c_beh, 4)
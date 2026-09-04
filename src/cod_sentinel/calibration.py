"""Probability calibration selected without touching the test split."""

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _probabilities(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if not np.isfinite(result).all():
        raise ValueError("Calibration probabilities must be finite.")
    return np.clip(result, 0.0, 1.0)


class IdentityCalibrator:
    name = "identity"

    def fit(self, probabilities: np.ndarray, labels: np.ndarray):
        _probabilities(probabilities)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return _probabilities(probabilities)


class SigmoidCalibrator:
    name = "sigmoid"

    def __init__(self, random_seed: int) -> None:
        self.model = LogisticRegression(random_state=random_seed)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray):
        values = _probabilities(probabilities).reshape(-1, 1)
        self.model.fit(values, np.asarray(labels, dtype=int))
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        values = _probabilities(probabilities).reshape(-1, 1)
        return self.model.predict_proba(values)[:, 1]


class IsotonicCalibrator:
    name = "isotonic"

    def __init__(self) -> None:
        self.model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )

    def fit(self, probabilities: np.ndarray, labels: np.ndarray):
        self.model.fit(
            _probabilities(probabilities),
            np.asarray(labels, dtype=int),
        )
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        return _probabilities(self.model.predict(_probabilities(probabilities)))


@dataclass(frozen=True)
class CalibrationChoice:
    calibrator: object
    method: str
    validation_brier: float
    candidate_brier: dict[str, float]


def select_calibrator(
    calibration_probabilities: np.ndarray,
    calibration_labels: np.ndarray,
    validation_probabilities: np.ndarray,
    validation_labels: np.ndarray,
    *,
    random_seed: int,
) -> CalibrationChoice:
    """Fit on calibration data and choose method on validation data."""

    candidates = (
        IdentityCalibrator(),
        SigmoidCalibrator(random_seed),
        IsotonicCalibrator(),
    )
    scores: dict[str, float] = {}
    fitted: dict[str, object] = {}
    for calibrator in candidates:
        calibrator.fit(calibration_probabilities, calibration_labels)
        predictions = calibrator.predict(validation_probabilities)
        scores[calibrator.name] = float(
            brier_score_loss(validation_labels, predictions)
        )
        fitted[calibrator.name] = calibrator

    method = min(scores, key=lambda name: (scores[name], name))
    return CalibrationChoice(
        calibrator=fitted[method],
        method=method,
        validation_brier=scores[method],
        candidate_brier=scores,
    )


def expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    """Return equal-width expected calibration error."""

    if bins <= 0:
        raise ValueError("bins must be positive.")
    truth = np.asarray(labels, dtype=int)
    predicted = _probabilities(probabilities)
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(np.digitize(predicted, edges[1:-1]), bins - 1)
    error = 0.0
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        if mask.any():
            error += float(mask.mean()) * abs(
                float(predicted[mask].mean()) - float(truth[mask].mean())
            )
    return error

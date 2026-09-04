import numpy as np
import pytest

from cod_sentinel.calibration import (
    expected_calibration_error,
    select_calibrator,
)


def test_calibrator_selection_outputs_valid_probabilities() -> None:
    calibration_probabilities = np.linspace(0.05, 0.95, 100)
    calibration_labels = (
        calibration_probabilities > np.linspace(0.0, 1.0, 100)
    ).astype(int)
    validation_probabilities = np.linspace(0.01, 0.99, 80)
    validation_labels = (validation_probabilities > 0.5).astype(int)

    choice = select_calibrator(
        calibration_probabilities,
        calibration_labels,
        validation_probabilities,
        validation_labels,
        random_seed=42,
    )
    predictions = choice.calibrator.predict(validation_probabilities)

    assert choice.method in {"identity", "sigmoid", "isotonic"}
    assert set(choice.candidate_brier) == {"identity", "sigmoid", "isotonic"}
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0.0) & (predictions <= 1.0)).all()


def test_ece_is_zero_for_perfect_grouped_probabilities() -> None:
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(labels, probabilities, bins=2) == 0.0


def test_ece_rejects_invalid_bin_count() -> None:
    with pytest.raises(ValueError, match="bins must be positive"):
        expected_calibration_error(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            bins=0,
        )

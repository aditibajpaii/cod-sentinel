import subprocess
import sys

import numpy as np
import pytest

from cod_sentinel.configuration import ProjectConfig, SplitSizes
from cod_sentinel.generator import generate_synthetic_world
from cod_sentinel.models import ModelBundle, TARGETS, train_model_bundle


@pytest.fixture(scope="module")
def small_bundle() -> tuple[ModelBundle, object]:
    config = ProjectConfig(
        random_seed=19,
        dataset_size=800,
        splits=SplitSizes(
            train=500,
            calibration=100,
            validation=100,
            test=100,
        ),
    )
    observable, oracle, _ = generate_synthetic_world(config)
    return train_model_bundle(observable, oracle, random_seed=19), observable


def test_training_produces_every_required_outcome_model(small_bundle) -> None:
    bundle, _ = small_bundle

    assert set(bundle.models) == set(TARGETS)
    assert bundle.metadata["test_split_accessed"] is False
    for model in bundle.models.values():
        assert model.estimator_name in {
            "hist_gradient_boosting",
            "logistic_regression",
        }
        assert model.calibration_method in {"identity", "sigmoid", "isotonic"}
    assert bundle.models["otp_rto"].training_condition == "otp_completed"
    assert (
        bundle.models["prepaid_failure"].training_condition
        == "prepaid_converted"
    )


def test_bundle_predicts_valid_probabilities_for_unknown_categories(
    small_bundle,
) -> None:
    bundle, observable = small_bundle
    row = observable.iloc[[0]].copy()
    row["category"] = "never-seen-category"
    row["pincode"] = "never-seen-zone"

    predictions = bundle.predict_all(row)

    assert set(predictions) == set(TARGETS)
    for values in predictions.values():
        assert values.shape == (1,)
        assert np.isfinite(values).all()
        assert ((values >= 0.0) & (values <= 1.0)).all()


def test_model_bundle_round_trip(small_bundle, tmp_path) -> None:
    bundle, observable = small_bundle
    path = tmp_path / "model_bundle.joblib"

    bundle.save(path)
    loaded = ModelBundle.load(path)

    expected = bundle.predict_all(observable.iloc[[0]])
    actual = loaded.predict_all(observable.iloc[[0]])
    for name in TARGETS:
        np.testing.assert_allclose(actual[name], expected[name])


def test_model_bundle_reloads_in_fresh_process(small_bundle, tmp_path) -> None:
    bundle, _ = small_bundle
    path = tmp_path / "model_bundle.joblib"
    bundle.save(path)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from cod_sentinel.models import ModelBundle; "
                f"bundle = ModelBundle.load({str(path)!r}); "
                "assert len(bundle.models) == 5"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_model_bundle_rejects_stale_version(small_bundle, tmp_path) -> None:
    bundle, _ = small_bundle
    path = tmp_path / "model_bundle.joblib"
    original = bundle.metadata["model_version"]
    bundle.metadata["model_version"] = "obsolete-model-v0"
    bundle.save(path)

    with pytest.raises(ValueError, match="version mismatch"):
        ModelBundle.load(path)

    bundle.metadata["model_version"] = original


def test_direct_module_training_is_rejected() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "cod_sentinel.models"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "Use `make train`" in completed.stderr

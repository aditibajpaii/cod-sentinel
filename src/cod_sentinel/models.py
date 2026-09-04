"""Training and serialization for supervised simulator outcome models."""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from cod_sentinel.calibration import expected_calibration_error, select_calibrator
from cod_sentinel.configuration import ARTIFACTS_DIR, DEFAULT_CONFIG
from cod_sentinel.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RUNTIME_FEATURES,
    select_runtime_features,
)
from cod_sentinel.generator import (
    OBSERVABLE_PATH,
    ORACLE_PATH,
    SPLIT_MANIFEST_PATH,
)
from cod_sentinel.integrity import (
    sha256_file,
    validate_split_manifest,
    verify_hash_sidecar,
    write_hash_sidecar,
)
from cod_sentinel.leakage import LEAKAGE_REPORT_PATH
from cod_sentinel.versioning import (
    CALIBRATION_VERSION,
    DATASET_VERSION,
    FEATURE_VERSION,
    MODEL_VERSION,
)

MODEL_BUNDLE_PATH = ARTIFACTS_DIR / "model_bundle.joblib"
MODEL_METADATA_PATH = ARTIFACTS_DIR / "model_metadata.json"

TARGETS = {
    "cod_rto": "cod_rto",
    "otp_completion": "otp_completed",
    "otp_rto": "otp_rto_if_completed",
    "prepaid_conversion": "prepaid_converted",
    "prepaid_failure": "prepaid_failed_delivery",
}
TARGET_CONDITIONS = {
    "otp_rto": "otp_completed",
    "prepaid_failure": "prepaid_converted",
}


@dataclass
class OutcomeModel:
    name: str
    target_column: str
    estimator_name: str
    estimator: object
    calibration_method: str
    calibrator: object
    validation_metrics: dict[str, object]
    training_condition: str | None = None

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        runtime = select_runtime_features(features)
        raw = self.estimator.predict_proba(runtime)[:, 1]
        calibrated = np.asarray(self.calibrator.predict(raw), dtype=float)
        if not np.isfinite(calibrated).all():
            raise ValueError(f"{self.name} produced non-finite probabilities.")
        if ((calibrated < 0.0) | (calibrated > 1.0)).any():
            raise ValueError(f"{self.name} produced invalid probabilities.")
        return calibrated


@dataclass
class ModelBundle:
    models: dict[str, OutcomeModel]
    metadata: dict[str, object]

    def predict_all(self, features: pd.DataFrame) -> dict[str, np.ndarray]:
        missing = set(TARGETS).difference(self.models)
        if missing:
            raise ValueError(f"Model bundle is incomplete: {sorted(missing)}")
        return {name: self.models[name].predict(features) for name in TARGETS}

    def save(self, path: Path | str = MODEL_BUNDLE_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        write_hash_sidecar(path)

    @classmethod
    def load(cls, path: Path | str = MODEL_BUNDLE_PATH) -> "ModelBundle":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model bundle not found: {path}")
        verify_hash_sidecar(path)
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise ValueError("Artifact is not a COD Sentinel ModelBundle.")
        bundle.validate_versions()
        return bundle

    def validate_versions(self) -> None:
        expected = {
            "model_version": MODEL_VERSION,
            "calibration_version": CALIBRATION_VERSION,
            "dataset_version": DATASET_VERSION,
            "feature_version": FEATURE_VERSION,
        }
        mismatches = {
            name: (self.metadata.get(name), value)
            for name, value in expected.items()
            if self.metadata.get(name) != value
        }
        if mismatches:
            raise ValueError(f"Model bundle version mismatch: {mismatches}")


def _logistic_pipeline(random_seed: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(NUMERIC_FEATURES),
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(max_iter=1_000, random_state=random_seed),
            ),
        ]
    )


def _histogram_pipeline(random_seed: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                list(CATEGORICAL_FEATURES),
            ),
            (
                "numeric",
                SimpleImputer(strategy="median"),
                list(NUMERIC_FEATURES),
            ),
        ]
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.08,
                    max_iter=160,
                    max_leaf_nodes=20,
                    l2_regularization=0.1,
                    random_state=random_seed,
                ),
            ),
        ]
    )


def _candidate_estimators(random_seed: int) -> dict[str, Pipeline]:
    return {
        "hist_gradient_boosting": _histogram_pipeline(random_seed),
        "logistic_regression": _logistic_pipeline(random_seed),
    }


def _train_outcome_model(
    name: str,
    target_column: str,
    train_features: pd.DataFrame,
    train_labels: np.ndarray,
    calibration_features: pd.DataFrame,
    calibration_labels: np.ndarray,
    validation_features: pd.DataFrame,
    validation_labels: np.ndarray,
    *,
    random_seed: int,
) -> OutcomeModel:
    choices: list[OutcomeModel] = []
    for estimator_name, estimator in _candidate_estimators(random_seed).items():
        estimator.fit(train_features, train_labels)
        calibration_raw = estimator.predict_proba(calibration_features)[:, 1]
        validation_raw = estimator.predict_proba(validation_features)[:, 1]
        calibration = select_calibrator(
            calibration_raw,
            calibration_labels,
            validation_raw,
            validation_labels,
            random_seed=random_seed,
        )
        validation_calibrated = calibration.calibrator.predict(validation_raw)
        metrics: dict[str, object] = {
            "brier": float(
                brier_score_loss(validation_labels, validation_calibrated)
            ),
            "pr_auc": float(
                average_precision_score(validation_labels, validation_calibrated)
            ),
            "ece": expected_calibration_error(
                validation_labels,
                validation_calibrated,
            ),
            "calibration_candidates_brier": calibration.candidate_brier,
        }
        choices.append(
            OutcomeModel(
                name=name,
                target_column=target_column,
                estimator_name=estimator_name,
                estimator=estimator,
                calibration_method=calibration.method,
                calibrator=calibration.calibrator,
                validation_metrics=metrics,
            )
        )

    return min(
        choices,
        key=lambda model: (
            float(model.validation_metrics["brier"]),
            model.estimator_name,
        ),
    )


def train_model_bundle(
    observable: pd.DataFrame,
    oracle: pd.DataFrame,
    *,
    random_seed: int = DEFAULT_CONFIG.random_seed,
) -> ModelBundle:
    """Train, calibrate, and select models without reading test labels."""

    development = observable.loc[
        observable["split"].isin(("train", "calibration", "validation"))
    ].copy()
    development_ids = set(development["order_id"])
    development_oracle = oracle.loc[
        oracle["order_id"].isin(development_ids),
        ["order_id", *TARGETS.values()],
    ]
    labels = development[["order_id", "split"]].merge(
        development_oracle,
        on="order_id",
        validate="one_to_one",
    )
    split_features: dict[str, pd.DataFrame] = {}
    split_labels: dict[str, pd.DataFrame] = {}
    for split in ("train", "calibration", "validation"):
        mask = labels["split"] == split
        split_features[split] = select_runtime_features(
            development.loc[mask]
        ).reset_index(drop=True)
        split_labels[split] = labels.loc[mask].reset_index(drop=True)

    models: dict[str, OutcomeModel] = {}
    for name, target_column in TARGETS.items():
        condition = TARGET_CONDITIONS.get(name)
        masks = {
            split: (
                split_labels[split][condition].to_numpy(dtype=bool)
                if condition is not None
                else np.ones(len(split_labels[split]), dtype=bool)
            )
            for split in ("train", "calibration", "validation")
        }
        models[name] = _train_outcome_model(
            name,
            target_column,
            split_features["train"].loc[masks["train"]].reset_index(drop=True),
            split_labels["train"]
            .loc[masks["train"], target_column]
            .to_numpy(),
            split_features["calibration"]
            .loc[masks["calibration"]]
            .reset_index(drop=True),
            split_labels["calibration"]
            .loc[masks["calibration"], target_column]
            .to_numpy(),
            split_features["validation"]
            .loc[masks["validation"]]
            .reset_index(drop=True),
            split_labels["validation"]
            .loc[masks["validation"], target_column]
            .to_numpy(),
            random_seed=random_seed,
        )
        models[name].training_condition = condition

    metadata: dict[str, object] = {
        "model_version": MODEL_VERSION,
        "calibration_version": CALIBRATION_VERSION,
        "dataset_version": DATASET_VERSION,
        "feature_version": FEATURE_VERSION,
        "random_seed": random_seed,
        "runtime_features": list(RUNTIME_FEATURES),
        "selection_split": "validation",
        "training_splits_used": sorted(labels["split"].unique().tolist()),
        "test_split_accessed": bool((labels["split"] == "test").any()),
        "models": {
            name: {
                "target_column": model.target_column,
                "estimator": model.estimator_name,
                "calibration": model.calibration_method,
                "validation_metrics": model.validation_metrics,
                "training_condition": model.training_condition,
            }
            for name, model in models.items()
        },
    }
    return ModelBundle(models=models, metadata=metadata)


def train_from_artifacts(
    observable_path: Path = OBSERVABLE_PATH,
    oracle_path: Path = ORACLE_PATH,
    leakage_report_path: Path = LEAKAGE_REPORT_PATH,
    split_manifest_path: Path = SPLIT_MANIFEST_PATH,
    bundle_path: Path = MODEL_BUNDLE_PATH,
    metadata_path: Path = MODEL_METADATA_PATH,
) -> ModelBundle:
    """Train only after a passing leakage report exists."""

    if not leakage_report_path.exists():
        raise FileNotFoundError("Run leakage gates before model training.")
    leakage_report = json.loads(leakage_report_path.read_text(encoding="utf-8"))
    if leakage_report.get("passed") is not True:
        raise ValueError("Leakage report did not pass.")
    expected_hashes = leakage_report.get("artifact_hashes", {})
    for name, path in {
        "observable": observable_path,
        "oracle": oracle_path,
        "split_manifest": split_manifest_path,
    }.items():
        if expected_hashes.get(name) != sha256_file(path):
            raise ValueError(f"Leakage approval is stale for {name}.")

    observable = pd.read_csv(observable_path, parse_dates=["ordered_at"])
    oracle = pd.read_csv(oracle_path)
    validate_split_manifest(observable, split_manifest_path, DEFAULT_CONFIG)
    bundle = train_model_bundle(observable, oracle)
    bundle.save(bundle_path)
    metadata_path.write_text(
        json.dumps(bundle.metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return bundle


def main() -> None:
    bundle = train_from_artifacts()
    print(f"Saved {len(bundle.models)} outcome models to {MODEL_BUNDLE_PATH}")
    for name, model in bundle.models.items():
        print(
            f"{name}: {model.estimator_name} + {model.calibration_method}; "
            f"validation Brier={model.validation_metrics['brier']:.4f}"
        )


if __name__ == "__main__":
    raise SystemExit(
        "Direct module execution would create non-portable pickle metadata. "
        "Use `make train`."
    )

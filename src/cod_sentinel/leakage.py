"""Executable leakage gates for generated artifacts."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from cod_sentinel.configuration import ARTIFACTS_DIR, DEFAULT_CONFIG
from cod_sentinel.features import (
    CATEGORICAL_FEATURES,
    FEATURE_PROVENANCE,
    HISTORY_FEATURES,
    NUMERIC_FEATURES,
    ORACLE_DENYLIST,
    RUNTIME_FEATURES,
    add_prior_history_features,
    select_runtime_features,
)
from cod_sentinel.generator import OBSERVABLE_PATH, ORACLE_PATH

LEAKAGE_REPORT_PATH = ARTIFACTS_DIR / "leakage_report.json"


def validate_feature_contract(observable: pd.DataFrame) -> None:
    """Validate explicit allowlist, denylist, and provenance coverage."""

    select_runtime_features(observable)
    if set(FEATURE_PROVENANCE) != set(RUNTIME_FEATURES):
        missing = set(RUNTIME_FEATURES).difference(FEATURE_PROVENANCE)
        extra = set(FEATURE_PROVENANCE).difference(RUNTIME_FEATURES)
        raise ValueError(
            f"Feature provenance mismatch; missing={sorted(missing)}, "
            f"extra={sorted(extra)}."
        )


def validate_physical_separation(
    observable: pd.DataFrame,
    oracle: pd.DataFrame,
) -> None:
    """Require order_id to be the only shared observable/oracle column."""

    shared = set(observable.columns).intersection(oracle.columns)
    if shared != {"order_id"}:
        raise ValueError(
            f"Observable/oracle artifacts share forbidden columns: "
            f"{sorted(shared.difference({'order_id'}))}"
        )
    if observable["order_id"].duplicated().any() or oracle["order_id"].duplicated().any():
        raise ValueError("Observable and oracle order_id values must be unique.")
    if set(observable["order_id"]) != set(oracle["order_id"]):
        raise ValueError("Observable and oracle artifacts contain different orders.")


def validate_prior_history(
    observable: pd.DataFrame,
    oracle: pd.DataFrame,
) -> None:
    """Recompute history from prior logged outcomes and compare exactly."""

    logged_columns = [
        "order_id",
        "logged_action",
        "logged_converted",
        "logged_delivered",
        "logged_rto",
    ]
    raw = observable.drop(columns=list(HISTORY_FEATURES) + ["split"]).merge(
        oracle[logged_columns],
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    raw["ordered_at"] = pd.to_datetime(raw["ordered_at"], errors="raise")
    recomputed = add_prior_history_features(raw)
    recomputed = recomputed.set_index("order_id").loc[
        observable["order_id"],
        list(HISTORY_FEATURES),
    ]
    recorded = observable.set_index("order_id").loc[
        observable["order_id"],
        list(HISTORY_FEATURES),
    ]
    try:
        np.testing.assert_allclose(
            recorded.to_numpy(dtype=float),
            recomputed.to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-10,
        )
    except AssertionError as error:
        raise ValueError(
            "Recorded history features do not match strictly prior outcomes."
        ) from error


def shuffled_label_auc(
    observable: pd.DataFrame,
    oracle: pd.DataFrame,
    *,
    random_seed: int,
) -> float:
    """Fit shuffled training labels and score untouched validation labels."""

    merged = observable[["order_id", "split"]].merge(
        oracle[["order_id", "cod_rto"]],
        on="order_id",
        validate="one_to_one",
    )
    train_mask = merged["split"] == "train"
    validation_mask = merged["split"] == "validation"
    train_features = select_runtime_features(observable.loc[train_mask])
    validation_features = select_runtime_features(observable.loc[validation_mask])
    train_labels = merged.loc[train_mask, "cod_rto"].to_numpy()
    validation_labels = merged.loc[validation_mask, "cod_rto"].to_numpy()
    shuffled = np.random.default_rng(random_seed).permutation(train_labels)

    preprocessor = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    random_state=random_seed,
                ),
            ),
        ]
    )
    pipeline.fit(train_features, shuffled)
    probabilities = pipeline.predict_proba(validation_features)[:, 1]
    return float(roc_auc_score(validation_labels, probabilities))


def run_leakage_gates(
    observable_path: Path = OBSERVABLE_PATH,
    oracle_path: Path = ORACLE_PATH,
    report_path: Path = LEAKAGE_REPORT_PATH,
) -> dict[str, object]:
    """Run every leakage gate and persist a machine-readable report."""

    if not observable_path.exists() or not oracle_path.exists():
        raise FileNotFoundError("Generate observable and oracle artifacts first.")
    observable = pd.read_csv(observable_path, parse_dates=["ordered_at"])
    oracle = pd.read_csv(oracle_path)

    validate_feature_contract(observable)
    validate_physical_separation(observable, oracle)
    validate_prior_history(observable, oracle)
    shuffled_auc = shuffled_label_auc(
        observable,
        oracle,
        random_seed=DEFAULT_CONFIG.random_seed,
    )
    if shuffled_auc >= 0.60:
        raise ValueError(
            f"Shuffled-label validation ROC-AUC {shuffled_auc:.3f} is "
            "unexpectedly high."
        )

    report: dict[str, object] = {
        "passed": True,
        "gates": {
            "feature_allowlist_and_provenance": "passed",
            "oracle_physical_separation": "passed",
            "prior_history_recomputation": "passed",
            "shuffled_label_sanity": "passed",
        },
        "shuffled_label_validation_roc_auc": shuffled_auc,
        "oracle_denylist": sorted(ORACLE_DENYLIST),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = run_leakage_gates()
    print(
        "Leakage gates passed; shuffled-label ROC-AUC="
        f"{report['shuffled_label_validation_roc_auc']:.3f}"
    )


if __name__ == "__main__":
    main()

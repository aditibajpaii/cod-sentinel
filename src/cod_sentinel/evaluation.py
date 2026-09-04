"""Validation freezing and held-out simulator policy evaluation."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from cod_sentinel.calibration import expected_calibration_error
from cod_sentinel.configuration import (
    ARTIFACTS_DIR,
    RESULTS_DIR,
    PolicySettings,
    load_policy_settings,
)
from cod_sentinel.economics import realized_contribution
from cod_sentinel.generator import OBSERVABLE_PATH, ORACLE_PATH
from cod_sentinel.models import MODEL_BUNDLE_PATH, ModelBundle
from cod_sentinel.policy import REQUIRED_PROBABILITIES, decide_from_probabilities
from cod_sentinel.schemas import Action, OrderEconomics, RealizedOutcome
from cod_sentinel.versioning import (
    DATASET_VERSION,
    DGP_VERSION,
    ECONOMICS_VERSION,
    MODEL_VERSION,
    POLICY_VERSION,
)

FROZEN_POLICY_PATH = ARTIFACTS_DIR / "frozen_policy.json"
METRICS_PATH = RESULTS_DIR / "metrics.json"


def realized_action_contribution(
    action: Action,
    order_value: float,
    oracle_row: pd.Series,
    settings: PolicySettings,
) -> float:
    """Price the held-out potential outcome selected by a policy."""

    if action is Action.COD:
        outcome = RealizedOutcome(
            converted=True,
            delivered=not bool(oracle_row["cod_rto"]),
        )
    elif action is Action.OTP:
        converted = bool(oracle_row["otp_completed"])
        outcome = RealizedOutcome(
            converted=converted,
            delivered=converted and not bool(oracle_row["otp_rto_if_completed"]),
        )
    elif action is Action.PREPAID:
        converted = bool(oracle_row["prepaid_converted"])
        outcome = RealizedOutcome(
            converted=converted,
            delivered=converted
            and not bool(oracle_row["prepaid_failed_delivery"]),
        )
    else:
        raise ValueError(f"Unsupported evaluation action: {action!r}")
    return realized_contribution(
        action,
        OrderEconomics(order_value=float(order_value)),
        settings.merchant_economics,
        outcome,
    )


def _split_data(
    observable: pd.DataFrame,
    oracle: pd.DataFrame,
    split: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = observable.loc[observable["split"] == split].reset_index(drop=True)
    outcomes = features[["order_id"]].merge(
        oracle,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    if outcomes.isna().any().any():
        raise ValueError(f"Oracle outcomes are incomplete for split {split!r}.")
    return features, outcomes


def _realized_policy_values(
    actions: list[Action],
    observable: pd.DataFrame,
    oracle: pd.DataFrame,
    settings: PolicySettings,
) -> np.ndarray:
    return np.array(
        [
            realized_action_contribution(
                action,
                float(observable.iloc[index]["order_value"]),
                oracle.iloc[index],
                settings,
            )
            for index, action in enumerate(actions)
        ],
        dtype=float,
    )


def _risk_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    candidates = np.linspace(0.05, 0.95, 37)
    scored = [
        (
            float(f1_score(labels, probabilities >= threshold)),
            float(threshold),
        )
        for threshold in candidates
    ]
    best_f1, threshold = max(scored, key=lambda item: (item[0], -item[1]))
    return threshold, best_f1


def _threshold_policy(
    cod_probabilities: np.ndarray,
    threshold: float,
) -> list[Action]:
    return [
        Action.OTP if probability >= threshold else Action.COD
        for probability in cod_probabilities
    ]


def freeze_policy_on_validation(
    bundle: ModelBundle,
    observable: pd.DataFrame,
    oracle: pd.DataFrame,
    settings: PolicySettings,
    path: Path = FROZEN_POLICY_PATH,
) -> dict[str, object]:
    """Select all thresholds on validation and persist immutable settings."""

    validation, outcomes = _split_data(observable, oracle, "validation")
    predictions = bundle.predict_all(validation)
    labels = outcomes["cod_rto"].to_numpy(dtype=int)
    classification_threshold, validation_f1 = _risk_threshold(
        labels,
        predictions["cod_rto"],
    )

    economic_candidates: list[tuple[float, float]] = []
    for threshold in np.linspace(0.05, 0.95, 37):
        actions = _threshold_policy(predictions["cod_rto"], float(threshold))
        values = _realized_policy_values(
            actions,
            validation,
            outcomes,
            settings,
        )
        economic_candidates.append((float(values.mean()), float(threshold)))
    validation_contribution, economic_threshold = max(
        economic_candidates,
        key=lambda item: (item[0], -item[1]),
    )

    frozen: dict[str, object] = {
        "policy_version": POLICY_VERSION,
        "model_version": MODEL_VERSION,
        "economics_version": ECONOMICS_VERSION,
        "dataset_version": DATASET_VERSION,
        "dgp_version": DGP_VERSION,
        "config_version": settings.config_version,
        "risk_classification_threshold": classification_threshold,
        "threshold_policy_rto_cutoff": economic_threshold,
        "selection_split": "validation",
        "validation_f1": validation_f1,
        "validation_threshold_policy_contribution_per_order": (
            validation_contribution
        ),
        "merchant_economics": asdict(settings.merchant_economics),
        "tie_break_order": [action.value for action in settings.tie_break_order],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    return frozen


def _learned_actions(
    observable: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    settings: PolicySettings,
) -> tuple[list[Action], np.ndarray]:
    actions: list[Action] = []
    predicted_values: list[float] = []
    for index, row in observable.iterrows():
        values = {
            name: float(predictions[name][index])
            for name in REQUIRED_PROBABILITIES
        }
        decision = decide_from_probabilities(
            order_id=row["order_id"],
            order_value=row["order_value"],
            probability_values=values,
            settings=settings,
        )
        if decision.requires_review or decision.expected_contributions is None:
            raise ValueError("Frozen test inputs unexpectedly routed to REVIEW.")
        action = Action(decision.selected_action)
        actions.append(action)
        predicted_values.append(decision.expected_contributions[action.value])
    return actions, np.asarray(predicted_values, dtype=float)


def _oracle_actions_and_values(
    observable: pd.DataFrame,
    oracle: pd.DataFrame,
    settings: PolicySettings,
) -> tuple[list[Action], np.ndarray]:
    actions: list[Action] = []
    values: list[float] = []
    tie_rank = {action: index for index, action in enumerate(settings.tie_break_order)}
    for index, row in observable.iterrows():
        candidates = {
            action: realized_action_contribution(
                action,
                float(row["order_value"]),
                oracle.iloc[index],
                settings,
            )
            for action in Action
        }
        selected = max(
            Action,
            key=lambda action: (candidates[action], -tie_rank[action]),
        )
        actions.append(selected)
        values.append(candidates[selected])
    return actions, np.asarray(values, dtype=float)


def customer_cluster_bootstrap_interval(
    customer_ids: np.ndarray,
    differences: np.ndarray,
    *,
    random_seed: int,
    samples: int = 1_000,
) -> tuple[float, float]:
    """Bootstrap mean contribution difference by customer cluster."""

    if samples <= 0:
        raise ValueError("samples must be positive.")
    frame = pd.DataFrame(
        {"customer_id": customer_ids, "difference": differences}
    )
    clusters = frame.groupby("customer_id")["difference"].agg(["sum", "count"])
    if clusters.empty:
        raise ValueError("Cannot bootstrap an empty evaluation.")
    sums = clusters["sum"].to_numpy(dtype=float)
    counts = clusters["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_seed)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        estimates[index] = sums[selected].sum() / counts[selected].sum()
    low, high = np.quantile(estimates, [0.025, 0.975])
    return float(low), float(high)


def _policy_summary(
    name: str,
    actions: list[Action],
    values: np.ndarray,
    always_cod_values: np.ndarray,
    cod_success: np.ndarray,
    oracle: pd.DataFrame,
) -> dict[str, object]:
    action_values = np.array([action.value for action in actions])
    intervened = action_values != Action.COD.value
    false_positive = intervened & cod_success
    false_positive_loss = np.maximum(
        always_cod_values[false_positive] - values[false_positive],
        0.0,
    )
    otp_selected = action_values == Action.OTP.value
    prepaid_selected = action_values == Action.PREPAID.value
    return {
        "name": name,
        "total_realized_contribution": float(values.sum()),
        "realized_contribution_per_order": float(values.mean()),
        "improvement_vs_always_cod_total": float(
            (values - always_cod_values).sum()
        ),
        "action_distribution": {
            action.value: int((action_values == action.value).sum())
            for action in Action
        },
        "intervention_rate": float(intervened.mean()),
        "unnecessary_intervention_rate": float(false_positive.mean()),
        "false_positive_cost": float(false_positive_loss.sum()),
        "false_positive_orders": int(false_positive.sum()),
        "otp_completion_rate_when_selected": (
            float(oracle.loc[otp_selected, "otp_completed"].mean())
            if otp_selected.any()
            else None
        ),
        "prepaid_conversion_rate_when_selected": (
            float(oracle.loc[prepaid_selected, "prepaid_converted"].mean())
            if prepaid_selected.any()
            else None
        ),
    }


def _sensitivity_analysis(
    observable: pd.DataFrame,
    predictions: dict[str, np.ndarray],
    settings: PolicySettings,
) -> dict[str, list[dict[str, object]]]:
    scenarios: dict[str, list[tuple[float, PolicySettings, dict[str, np.ndarray]]]] = {}
    for field, values in {
        "prepaid_discount_rate": (0.0, 0.02, 0.05, 0.10),
        "reverse_shipping_cost": (40.0, 80.0, 120.0, 180.0),
        "inventory_damage_rate": (0.0, 0.10, 0.25, 0.50),
        "damaged_inventory_recovery_rate": (0.0, 0.25, 0.50, 0.90),
    }.items():
        scenarios[field] = [
            (
                value,
                replace(
                    settings,
                    merchant_economics=replace(
                        settings.merchant_economics,
                        **{field: value},
                    ),
                ),
                predictions,
            )
            for value in values
        ]

    scenarios["prepaid_conversion_multiplier"] = [
        (
            value,
            settings,
            {
                **predictions,
                "prepaid_conversion": np.clip(
                    predictions["prepaid_conversion"] * value,
                    0.0,
                    1.0,
                ),
            },
        )
        for value in (0.60, 0.80, 1.00, 1.20)
    ]
    scenarios["otp_completion_multiplier"] = [
        (
            value,
            settings,
            {
                **predictions,
                "otp_completion": np.clip(
                    predictions["otp_completion"] * value,
                    0.0,
                    1.0,
                ),
            },
        )
        for value in (0.60, 0.80, 1.00, 1.20)
    ]

    output: dict[str, list[dict[str, object]]] = {}
    for name, variants in scenarios.items():
        rows: list[dict[str, object]] = []
        for value, scenario_settings, scenario_predictions in variants:
            actions, expected = _learned_actions(
                observable,
                scenario_predictions,
                scenario_settings,
            )
            rows.append(
                {
                    "value": value,
                    "mean_selected_predicted_ev": float(expected.mean()),
                    "action_distribution": {
                        action.value: actions.count(action) for action in Action
                    },
                }
            )
        output[name] = rows
    return output


def evaluate_frozen_test(
    bundle: ModelBundle,
    observable: pd.DataFrame,
    oracle: pd.DataFrame,
    settings: PolicySettings,
    frozen: dict[str, object],
) -> dict[str, object]:
    """Evaluate a frozen policy against held-out realized potential outcomes."""

    test, outcomes = _split_data(observable, oracle, "test")
    predictions = bundle.predict_all(test)
    cod_labels = outcomes["cod_rto"].to_numpy(dtype=int)
    cod_probabilities = predictions["cod_rto"]
    classification_threshold = float(frozen["risk_classification_threshold"])
    classifications = cod_probabilities >= classification_threshold

    always_actions = [Action.COD] * len(test)
    always_otp_actions = [Action.OTP] * len(test)
    always_prepaid_actions = [Action.PREPAID] * len(test)
    threshold_actions = _threshold_policy(
        cod_probabilities,
        float(frozen["threshold_policy_rto_cutoff"]),
    )
    learned_actions, selected_predicted_ev = _learned_actions(
        test,
        predictions,
        settings,
    )
    oracle_actions, oracle_values = _oracle_actions_and_values(
        test,
        outcomes,
        settings,
    )
    always_values = _realized_policy_values(
        always_actions,
        test,
        outcomes,
        settings,
    )
    always_otp_values = _realized_policy_values(
        always_otp_actions,
        test,
        outcomes,
        settings,
    )
    always_prepaid_values = _realized_policy_values(
        always_prepaid_actions,
        test,
        outcomes,
        settings,
    )
    threshold_values = _realized_policy_values(
        threshold_actions,
        test,
        outcomes,
        settings,
    )
    learned_values = _realized_policy_values(
        learned_actions,
        test,
        outcomes,
        settings,
    )
    cod_success = cod_labels == 0
    improvement = learned_values - always_values
    confidence_interval = customer_cluster_bootstrap_interval(
        test["customer_id"].to_numpy(),
        improvement,
        random_seed=42,
    )

    policies = {
        "always_cod": _policy_summary(
            "Always COD",
            always_actions,
            always_values,
            always_values,
            cod_success,
            outcomes,
        ),
        "always_otp": _policy_summary(
            "Always OTP",
            always_otp_actions,
            always_otp_values,
            always_values,
            cod_success,
            outcomes,
        ),
        "always_prepaid": _policy_summary(
            "Always prepaid",
            always_prepaid_actions,
            always_prepaid_values,
            always_values,
            cod_success,
            outcomes,
        ),
        "risk_threshold": _policy_summary(
            "Validation-selected RTO threshold",
            threshold_actions,
            threshold_values,
            always_values,
            cod_success,
            outcomes,
        ),
        "cod_sentinel": _policy_summary(
            "COD Sentinel",
            learned_actions,
            learned_values,
            always_values,
            cod_success,
            outcomes,
        ),
    }
    policies["cod_sentinel"]["improvement_per_order_95pct_cluster_bootstrap"] = {
        "low": confidence_interval[0],
        "high": confidence_interval[1],
    }
    policies["cod_sentinel"]["mean_selected_predicted_ev"] = float(
        selected_predicted_ev.mean()
    )
    policies["cod_sentinel"]["mean_realized_simulator_contribution"] = float(
        learned_values.mean()
    )
    simple_baselines = {
        name: summary["realized_contribution_per_order"]
        for name, summary in policies.items()
        if name != "cod_sentinel"
    }
    best_simple_name = max(simple_baselines, key=simple_baselines.get)
    best_simple_per_order = float(simple_baselines[best_simple_name])
    policies["cod_sentinel"]["best_simple_baseline"] = best_simple_name
    policies["cod_sentinel"]["improvement_vs_best_simple_baseline_per_order"] = (
        float(learned_values.mean()) - best_simple_per_order
    )

    regret = oracle_values - learned_values
    if (regret < -1e-9).any():
        raise ValueError("Learned policy exceeded the per-order oracle upper bound.")
    calibration_bins: list[dict[str, object]] = []
    bin_ids = np.minimum(
        np.digitize(cod_probabilities, np.linspace(0.0, 1.0, 11)[1:-1]),
        9,
    )
    for bin_id in range(10):
        mask = bin_ids == bin_id
        if mask.any():
            calibration_bins.append(
                {
                    "bin": bin_id,
                    "count": int(mask.sum()),
                    "mean_predicted": float(cod_probabilities[mask].mean()),
                    "observed_rate": float(cod_labels[mask].mean()),
                }
            )

    return {
        "evidence_scope": "temporally held-out synthetic simulator",
        "test_orders": int(len(test)),
        "versions": {
            "dataset": DATASET_VERSION,
            "dgp": DGP_VERSION,
            "model": MODEL_VERSION,
            "economics": ECONOMICS_VERSION,
            "policy": POLICY_VERSION,
            "config": settings.config_version,
        },
        "risk_model": {
            "threshold": classification_threshold,
            "precision": float(precision_score(cod_labels, classifications)),
            "recall": float(recall_score(cod_labels, classifications)),
            "f1": float(f1_score(cod_labels, classifications)),
            "pr_auc": float(
                average_precision_score(cod_labels, cod_probabilities)
            ),
            "brier": float(brier_score_loss(cod_labels, cod_probabilities)),
            "ece": expected_calibration_error(cod_labels, cod_probabilities),
            "false_positive_count": int(
                ((classifications == 1) & (cod_labels == 0)).sum()
            ),
            "calibration_bins": calibration_bins,
        },
        "policies": policies,
        "oracle_upper_bound": {
            "total_realized_contribution": float(oracle_values.sum()),
            "realized_contribution_per_order": float(oracle_values.mean()),
            "action_distribution": {
                action.value: oracle_actions.count(action) for action in Action
            },
        },
        "oracle_regret": {
            "total": float(regret.sum()),
            "mean_per_order": float(regret.mean()),
            "median_per_order": float(np.median(regret)),
            "p95_per_order": float(np.quantile(regret, 0.95)),
        },
        "sensitivity": {
            "scope": "predicted-policy sensitivity; potential outcomes unchanged",
            "scenarios": _sensitivity_analysis(test, predictions, settings),
        },
    }


def evaluate_from_artifacts(
    observable_path: Path = OBSERVABLE_PATH,
    oracle_path: Path = ORACLE_PATH,
    bundle_path: Path = MODEL_BUNDLE_PATH,
    frozen_path: Path = FROZEN_POLICY_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, object]:
    """Freeze on validation, then evaluate the unchanged policy on test."""

    observable = pd.read_csv(observable_path, parse_dates=["ordered_at"])
    oracle = pd.read_csv(oracle_path)
    bundle = ModelBundle.load(bundle_path)
    settings = load_policy_settings()
    frozen = freeze_policy_on_validation(
        bundle,
        observable,
        oracle,
        settings,
        frozen_path,
    )
    metrics = evaluate_frozen_test(
        bundle,
        observable,
        oracle,
        settings,
        frozen,
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return metrics


def main() -> None:
    metrics = evaluate_from_artifacts()
    risk = metrics["risk_model"]
    policy = metrics["policies"]["cod_sentinel"]
    print(
        f"Test precision={risk['precision']:.3f}; "
        f"recall={risk['recall']:.3f}; PR-AUC={risk['pr_auc']:.3f}"
    )
    print(
        "COD Sentinel realized contribution/order="
        f"₹{policy['realized_contribution_per_order']:.2f}; "
        "improvement vs always-COD="
        f"₹{policy['improvement_vs_always_cod_total']:.2f}"
    )
    print(f"Saved held-out metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()

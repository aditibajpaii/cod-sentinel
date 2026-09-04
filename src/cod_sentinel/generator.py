"""Reproducible temporal synthetic world for COD Sentinel."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from cod_sentinel.configuration import ARTIFACTS_DIR, DEFAULT_CONFIG, ProjectConfig
from cod_sentinel.features import add_prior_history_features
from cod_sentinel.splits import assign_temporal_splits, persist_split_manifest
from cod_sentinel.versioning import DATASET_VERSION, DGP_VERSION

OBSERVABLE_PATH = ARTIFACTS_DIR / "observable_orders.csv"
ORACLE_PATH = ARTIFACTS_DIR / "oracle_outcomes.csv"
SPLIT_MANIFEST_PATH = ARTIFACTS_DIR / "splits.json"
DATASET_METADATA_PATH = ARTIFACTS_DIR / "dataset_metadata.json"

ARCHETYPES = (
    "customer_refusal",
    "unavailability",
    "address_failure",
    "serial_abuse",
    "legitimate_high_risk_looking",
)

_PROFILE_MEANS = {
    "customer_refusal": (0.52, 0.72, 0.78, 0.12, 0.78),
    "unavailability": (0.76, 0.38, 0.82, 0.08, 0.42),
    "address_failure": (0.82, 0.72, 0.34, 0.08, 0.34),
    "serial_abuse": (0.34, 0.62, 0.62, 0.88, 0.66),
    "legitimate_high_risk_looking": (0.94, 0.90, 0.90, 0.03, 0.18),
}


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _customer_profiles(
    rng: np.random.Generator,
    customer_count: int,
) -> pd.DataFrame:
    archetypes = rng.choice(
        ARCHETYPES,
        size=customer_count,
        p=(0.28, 0.20, 0.15, 0.12, 0.25),
    )
    latent = np.array([_PROFILE_MEANS[name] for name in archetypes], dtype=float)
    latent += rng.normal(0.0, 0.08, size=latent.shape)
    latent = np.clip(latent, 0.01, 0.99)

    base_aov = rng.lognormal(mean=np.log(1_250), sigma=0.55, size=customer_count)
    base_aov *= np.where(
        archetypes == "legitimate_high_risk_looking",
        1.45,
        1.0,
    )
    activity = rng.lognormal(mean=0.0, sigma=0.8, size=customer_count)
    activity /= activity.sum()

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{index:05d}" for index in range(customer_count)],
            "archetype": archetypes,
            "latent_intent": latent[:, 0],
            "latent_availability": latent[:, 1],
            "latent_address_quality": latent[:, 2],
            "latent_abuse": latent[:, 3],
            "latent_price_sensitivity": latent[:, 4],
            "base_aov": base_aov,
            "activity_weight": activity,
        }
    )


def _logged_outcomes(
    actions: np.ndarray,
    cod_rto: np.ndarray,
    otp_completed: np.ndarray,
    otp_rto: np.ndarray,
    prepaid_converted: np.ndarray,
    prepaid_failed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    converted = np.ones(len(actions), dtype=bool)
    delivered = ~cod_rto
    rto = cod_rto.copy()

    otp_mask = actions == "OTP"
    converted[otp_mask] = otp_completed[otp_mask]
    delivered[otp_mask] = otp_completed[otp_mask] & ~otp_rto[otp_mask]
    rto[otp_mask] = otp_completed[otp_mask] & otp_rto[otp_mask]

    prepaid_mask = actions == "PREPAID"
    converted[prepaid_mask] = prepaid_converted[prepaid_mask]
    delivered[prepaid_mask] = (
        prepaid_converted[prepaid_mask] & ~prepaid_failed[prepaid_mask]
    )
    rto[prepaid_mask] = (
        prepaid_converted[prepaid_mask] & prepaid_failed[prepaid_mask]
    )
    return converted, delivered, rto


def generate_synthetic_world(
    config: ProjectConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Generate observable orders and physically separate oracle outcomes."""

    config.validate()
    rng = np.random.default_rng(config.random_seed)
    customer_count = max(2_500, config.dataset_size // 5)
    customers = _customer_profiles(rng, customer_count)
    customer_indices = rng.choice(
        customer_count,
        size=config.dataset_size,
        p=customers["activity_weight"].to_numpy(),
    )
    selected = customers.iloc[customer_indices].reset_index(drop=True)

    base_time = pd.Timestamp("2025-01-01T00:00:00")
    ordered_at = base_time + pd.to_timedelta(
        np.arange(config.dataset_size) * 1_560
        + rng.integers(0, 300, size=config.dataset_size),
        unit="s",
    )
    month = ordered_at.month.to_numpy()
    order_hour = ordered_at.hour.to_numpy()
    day_of_week = ordered_at.dayofweek.to_numpy()
    festival = np.isin(month, [10, 11]).astype(int)
    night = ((order_hour < 7) | (order_hour >= 22)).astype(float)

    categories = rng.choice(
        ("fashion", "electronics", "home", "beauty", "grocery"),
        size=config.dataset_size,
        p=(0.28, 0.20, 0.20, 0.17, 0.15),
    )
    category_value_multiplier = pd.Series(categories).map(
        {
            "fashion": 1.00,
            "electronics": 1.75,
            "home": 1.15,
            "beauty": 0.65,
            "grocery": 0.45,
        }
    ).to_numpy()
    category_risk = pd.Series(categories).map(
        {
            "fashion": 0.25,
            "electronics": 0.15,
            "home": 0.00,
            "beauty": 0.10,
            "grocery": -0.15,
        }
    ).to_numpy()

    order_value = (
        selected["base_aov"].to_numpy()
        * category_value_multiplier
        * rng.lognormal(mean=0.0, sigma=0.32, size=config.dataset_size)
    )
    order_value = np.round(np.clip(order_value, 200.0, 12_000.0), 2)
    order_value_ratio = order_value / selected["base_aov"].to_numpy()

    pincode_index = rng.integers(0, 120, size=config.dataset_size)
    pincodes = np.array([f"ZONE-{value:03d}" for value in pincode_index])
    pincode_effects = rng.normal(0.0, 0.28, size=120)
    pincode_risk = pincode_effects[pincode_index]

    archetype = selected["archetype"].to_numpy()
    appearance_penalty = np.where(
        archetype == "legitimate_high_risk_looking",
        0.28,
        0.0,
    )
    address_quality_signal = np.clip(
        selected["latent_address_quality"].to_numpy()
        - appearance_penalty
        + rng.normal(0.0, 0.12, size=config.dataset_size),
        0.0,
        1.0,
    )
    phone_probability = np.clip(
        0.45
        + 0.45 * selected["latent_intent"].to_numpy()
        - 0.18 * selected["latent_abuse"].to_numpy()
        - 0.12
        * (archetype == "legitimate_high_risk_looking").astype(float),
        0.05,
        0.98,
    )
    phone_verified = rng.random(config.dataset_size) < phone_probability

    intent = selected["latent_intent"].to_numpy()
    availability = selected["latent_availability"].to_numpy()
    address_quality = selected["latent_address_quality"].to_numpy()
    abuse = selected["latent_abuse"].to_numpy()
    price_sensitivity = selected["latent_price_sensitivity"].to_numpy()

    cod_logit = (
        -2.70
        + 1.55 * (1.0 - intent)
        + 1.45 * (1.0 - availability)
        + 1.70 * (1.0 - address_quality)
        + 1.65 * abuse
        + 0.24 * night
        + 0.20 * np.maximum(order_value_ratio - 1.0, 0.0)
        + category_risk
        + pincode_risk
        + 0.42 * festival
        + rng.normal(0.0, 0.18, size=config.dataset_size)
    )
    p_cod_rto = _sigmoid(cod_logit)
    p_otp_completion = _sigmoid(
        1.25
        + 1.10 * intent
        - 1.45 * price_sensitivity
        + 0.35 * phone_verified.astype(float)
        - 0.22 * night
        - 0.10 * np.maximum(order_value_ratio - 1.0, 0.0)
    )
    p_otp_rto = _sigmoid(
        cod_logit
        - 0.65
        - 0.55 * (1.0 - intent)
        - 0.35 * (1.0 - availability)
    )
    p_prepaid_conversion = _sigmoid(
        1.20
        + 0.95 * intent
        - 1.85 * price_sensitivity
        + 0.20 * phone_verified.astype(float)
        - 0.20 * np.maximum(order_value_ratio - 1.0, 0.0)
        - 0.18 * festival
    )
    p_prepaid_failure = _sigmoid(cod_logit - 0.95 - 0.25 * intent)

    cod_rto = rng.random(config.dataset_size) < p_cod_rto
    otp_completed = rng.random(config.dataset_size) < p_otp_completion
    otp_rto = rng.random(config.dataset_size) < p_otp_rto
    prepaid_converted = rng.random(config.dataset_size) < p_prepaid_conversion
    prepaid_failed = rng.random(config.dataset_size) < p_prepaid_failure

    apparent_risk = (
        0.35 * (1.0 - address_quality_signal)
        + 0.25 * (~phone_verified).astype(float)
        + 0.15 * night
        + 0.10 * festival
        + 0.15 * np.clip(order_value_ratio - 1.0, 0.0, 1.0)
    )
    prepaid_share = np.clip(0.06 + 0.12 * apparent_risk, 0.04, 0.20)
    otp_share = np.clip(0.12 + 0.28 * apparent_risk, 0.10, 0.42)
    action_draw = rng.random(config.dataset_size)
    logged_action = np.where(
        action_draw < prepaid_share,
        "PREPAID",
        np.where(action_draw < prepaid_share + otp_share, "OTP", "COD"),
    )
    logged_converted, logged_delivered, logged_rto = _logged_outcomes(
        logged_action,
        cod_rto,
        otp_completed,
        otp_rto,
        prepaid_converted,
        prepaid_failed,
    )
    quick_resolution_minutes = rng.integers(5, 121, size=config.dataset_size)
    delivered_resolution_minutes = rng.integers(
        2 * 24 * 60,
        7 * 24 * 60 + 1,
        size=config.dataset_size,
    )
    rto_resolution_minutes = rng.integers(
        5 * 24 * 60,
        12 * 24 * 60 + 1,
        size=config.dataset_size,
    )
    resolution_minutes = np.where(
        ~logged_converted,
        quick_resolution_minutes,
        np.where(
            logged_rto,
            rto_resolution_minutes,
            delivered_resolution_minutes,
        ),
    )
    outcome_observed_at = ordered_at + pd.to_timedelta(
        resolution_minutes,
        unit="m",
    )

    raw = pd.DataFrame(
        {
            "order_id": [
                f"ORDER-{index:06d}" for index in range(config.dataset_size)
            ],
            "ordered_at": ordered_at,
            "customer_id": selected["customer_id"].to_numpy(),
            "category": categories,
            "pincode": pincodes,
            "order_value": order_value,
            "order_hour": order_hour,
            "day_of_week": day_of_week,
            "month": month,
            "is_weekend": (day_of_week >= 5).astype(int),
            "festival_period": festival,
            "phone_verified": phone_verified.astype(int),
            "address_quality_signal": np.round(address_quality_signal, 6),
            "outcome_observed_at": outcome_observed_at,
            "logged_action": logged_action,
            "logged_converted": logged_converted.astype(int),
            "logged_delivered": logged_delivered.astype(int),
            "logged_rto": logged_rto.astype(int),
        }
    )
    with_history = add_prior_history_features(raw)
    with_splits = assign_temporal_splits(with_history, config)

    internal_columns = {
        "logged_action",
        "logged_converted",
        "logged_delivered",
        "logged_rto",
        "outcome_observed_at",
    }
    observable = with_splits.drop(columns=sorted(internal_columns))

    oracle = pd.DataFrame(
        {
            "order_id": raw["order_id"],
            "archetype": archetype,
            "latent_intent": np.round(intent, 6),
            "latent_address_quality": np.round(address_quality, 6),
            "latent_availability": np.round(availability, 6),
            "latent_abuse": np.round(abuse, 6),
            "latent_price_sensitivity": np.round(price_sensitivity, 6),
            "oracle_p_cod_rto": np.round(p_cod_rto, 8),
            "oracle_p_otp_completion": np.round(p_otp_completion, 8),
            "oracle_p_otp_rto": np.round(p_otp_rto, 8),
            "oracle_p_prepaid_conversion": np.round(
                p_prepaid_conversion,
                8,
            ),
            "oracle_p_prepaid_failure": np.round(p_prepaid_failure, 8),
            "cod_rto": cod_rto.astype(int),
            "otp_completed": otp_completed.astype(int),
            "otp_rto_if_completed": otp_rto.astype(int),
            "prepaid_converted": prepaid_converted.astype(int),
            "prepaid_failed_delivery": prepaid_failed.astype(int),
            "logged_action": logged_action,
            "logged_converted": logged_converted.astype(int),
            "logged_delivered": logged_delivered.astype(int),
            "logged_rto": logged_rto.astype(int),
            "outcome_observed_at": outcome_observed_at,
        }
    )
    oracle = observable[["order_id"]].merge(
        oracle,
        on="order_id",
        how="left",
        validate="one_to_one",
    )

    metadata: dict[str, object] = {
        "dataset_version": DATASET_VERSION,
        "dgp_version": DGP_VERSION,
        "random_seed": config.random_seed,
        "dataset_size": config.dataset_size,
        "split_sizes": asdict(config.splits),
        "archetypes": list(ARCHETYPES),
        "rates": {
            "cod_rto": float(oracle["cod_rto"].mean()),
            "otp_completion": float(oracle["otp_completed"].mean()),
            "otp_rto_if_completed": float(
                oracle.loc[
                    oracle["otp_completed"] == 1,
                    "otp_rto_if_completed",
                ].mean()
            ),
            "prepaid_conversion": float(oracle["prepaid_converted"].mean()),
            "prepaid_failure_if_converted": float(
                oracle.loc[
                    oracle["prepaid_converted"] == 1,
                    "prepaid_failed_delivery",
                ].mean()
            ),
        },
    }
    return observable, oracle, metadata


def write_synthetic_world(
    config: ProjectConfig = DEFAULT_CONFIG,
    artifact_dir: Path = ARTIFACTS_DIR,
) -> tuple[Path, Path]:
    """Generate and persist separate observable and oracle artifacts."""

    observable, oracle, metadata = generate_synthetic_world(config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    observable_path = artifact_dir / OBSERVABLE_PATH.name
    oracle_path = artifact_dir / ORACLE_PATH.name
    observable.to_csv(observable_path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
    oracle.to_csv(oracle_path, index=False)
    persist_split_manifest(observable, artifact_dir / SPLIT_MANIFEST_PATH.name)
    (artifact_dir / DATASET_METADATA_PATH.name).write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return observable_path, oracle_path


def main() -> None:
    observable_path, oracle_path = write_synthetic_world()
    print(f"Wrote observable orders: {observable_path}")
    print(f"Wrote oracle outcomes: {oracle_path}")


if __name__ == "__main__":
    main()

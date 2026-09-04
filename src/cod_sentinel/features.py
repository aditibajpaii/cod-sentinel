"""Prior-only feature construction and runtime feature schema."""

import heapq
from dataclasses import dataclass

import pandas as pd

CATEGORICAL_FEATURES = ("category", "pincode")
NUMERIC_FEATURES = (
    "order_value",
    "order_hour",
    "day_of_week",
    "month",
    "is_weekend",
    "festival_period",
    "phone_verified",
    "address_quality_signal",
    "customer_prior_orders",
    "customer_prior_rto_rate",
    "customer_prior_delivery_rate",
    "customer_prior_prepaid_attempts",
    "customer_prior_prepaid_success_rate",
    "customer_order_value_ratio",
    "pincode_prior_orders",
    "pincode_prior_rto_rate",
)
RUNTIME_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
HISTORY_FEATURES = (
    "customer_prior_orders",
    "customer_prior_rto_rate",
    "customer_prior_delivery_rate",
    "customer_prior_prepaid_attempts",
    "customer_prior_prepaid_success_rate",
    "customer_order_value_ratio",
    "pincode_prior_orders",
    "pincode_prior_rto_rate",
)

ORACLE_DENYLIST = frozenset(
    {
        "archetype",
        "latent_intent",
        "latent_address_quality",
        "latent_availability",
        "latent_abuse",
        "latent_price_sensitivity",
        "oracle_p_cod_rto",
        "oracle_p_otp_completion",
        "oracle_p_otp_rto",
        "oracle_p_prepaid_conversion",
        "oracle_p_prepaid_failure",
        "cod_rto",
        "otp_completed",
        "otp_rto_if_completed",
        "prepaid_converted",
        "prepaid_failed_delivery",
        "realized_contribution",
        "outcome_observed_at",
        "logged_action",
        "logged_converted",
        "logged_delivered",
        "logged_rto",
    }
)

FEATURE_PROVENANCE = {
    "category": "current order",
    "pincode": "current order synthetic service zone",
    "order_value": "current order",
    "order_hour": "current order timestamp",
    "day_of_week": "current order timestamp",
    "month": "current order timestamp",
    "is_weekend": "current order timestamp",
    "festival_period": "current order timestamp",
    "phone_verified": "current order",
    "address_quality_signal": "current order deterministic signal",
    "customer_prior_orders": "strictly earlier logged orders",
    "customer_prior_rto_rate": "strictly earlier logged orders",
    "customer_prior_delivery_rate": "strictly earlier logged orders",
    "customer_prior_prepaid_attempts": "strictly earlier logged orders",
    "customer_prior_prepaid_success_rate": "strictly earlier logged orders",
    "customer_order_value_ratio": "current value / strictly prior customer AOV",
    "pincode_prior_orders": "strictly earlier logged orders",
    "pincode_prior_rto_rate": "strictly earlier logged orders",
}

_HISTORY_INPUT_COLUMNS = {
    "order_id",
    "ordered_at",
    "customer_id",
    "pincode",
    "order_value",
    "outcome_observed_at",
    "logged_action",
    "logged_converted",
    "logged_delivered",
    "logged_rto",
}


@dataclass
class _HistoryStats:
    orders: int = 0
    rtos: int = 0
    deliveries: int = 0
    order_value_sum: float = 0.0
    prepaid_attempts: int = 0
    prepaid_successes: int = 0


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def add_prior_history_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Add history features using only rows with an earlier timestamp.

    Rows sharing the same timestamp are featurized before any row in that
    timestamp group updates history, preserving the strictly-earlier contract.
    """

    missing = _HISTORY_INPUT_COLUMNS.difference(orders.columns)
    if missing:
        raise ValueError(f"Cannot build history features: {sorted(missing)}")

    result = orders.sort_values(
        ["ordered_at", "order_id"],
        kind="stable",
    ).reset_index(drop=True)
    result["ordered_at"] = pd.to_datetime(result["ordered_at"], errors="raise")
    result["outcome_observed_at"] = pd.to_datetime(
        result["outcome_observed_at"],
        errors="raise",
    )
    if result["ordered_at"].isna().any() or result["outcome_observed_at"].isna().any():
        raise ValueError("Order and outcome timestamps must be complete.")
    if (result["outcome_observed_at"] < result["ordered_at"]).any():
        raise ValueError("An outcome cannot be observed before its order.")

    customer_stats: dict[str, _HistoryStats] = {}
    pincode_stats: dict[str, _HistoryStats] = {}
    feature_rows: list[dict[str, float | int]] = [{} for _ in range(len(result))]
    pending: list[tuple[int, int, dict[str, object]]] = []
    sequence = 0

    for ordered_at, timestamp_rows in result.groupby("ordered_at", sort=True):
        indices = list(timestamp_rows.index)
        ordered_at_ns = pd.Timestamp(ordered_at).value

        while pending and pending[0][0] < ordered_at_ns:
            _, _, outcome = heapq.heappop(pending)
            customer = customer_stats[str(outcome["customer_id"])]
            pincode = pincode_stats[str(outcome["pincode"])]
            customer.rtos += int(outcome["logged_rto"])
            customer.deliveries += int(outcome["logged_delivered"])
            if outcome["logged_action"] == "PREPAID":
                customer.prepaid_successes += int(outcome["logged_converted"])
            pincode.rtos += int(outcome["logged_rto"])

        for index in indices:
            row = result.loc[index]
            customer = customer_stats.get(str(row["customer_id"]), _HistoryStats())
            pincode = pincode_stats.get(str(row["pincode"]), _HistoryStats())
            prior_aov = (
                customer.order_value_sum / customer.orders
                if customer.orders
                else float(row["order_value"])
            )
            feature_rows[index] = {
                "customer_prior_orders": customer.orders,
                "customer_prior_rto_rate": _rate(customer.rtos, customer.orders),
                "customer_prior_delivery_rate": _rate(
                    customer.deliveries,
                    customer.orders,
                ),
                "customer_prior_prepaid_attempts": customer.prepaid_attempts,
                "customer_prior_prepaid_success_rate": _rate(
                    customer.prepaid_successes,
                    customer.prepaid_attempts,
                ),
                "customer_order_value_ratio": float(row["order_value"]) / prior_aov,
                "pincode_prior_orders": pincode.orders,
                "pincode_prior_rto_rate": _rate(pincode.rtos, pincode.orders),
            }

        for index in indices:
            row = result.loc[index]
            customer_key = str(row["customer_id"])
            pincode_key = str(row["pincode"])
            customer = customer_stats.setdefault(customer_key, _HistoryStats())
            pincode = pincode_stats.setdefault(pincode_key, _HistoryStats())

            customer.orders += 1
            customer.order_value_sum += float(row["order_value"])
            if row["logged_action"] == "PREPAID":
                customer.prepaid_attempts += 1

            pincode.orders += 1
            heapq.heappush(
                pending,
                (
                    pd.Timestamp(row["outcome_observed_at"]).value,
                    sequence,
                    {
                        "customer_id": customer_key,
                        "pincode": pincode_key,
                        "logged_action": row["logged_action"],
                        "logged_converted": row["logged_converted"],
                        "logged_delivered": row["logged_delivered"],
                        "logged_rto": row["logged_rto"],
                    },
                ),
            )
            sequence += 1

    return pd.concat([result, pd.DataFrame(feature_rows)], axis=1)


def select_runtime_features(orders: pd.DataFrame) -> pd.DataFrame:
    """Select the explicit runtime allowlist and reject oracle contamination."""

    contaminated = ORACLE_DENYLIST.intersection(orders.columns)
    if contaminated:
        raise ValueError(
            f"Oracle columns reached runtime feature selection: "
            f"{sorted(contaminated)}"
        )
    missing = set(RUNTIME_FEATURES).difference(orders.columns)
    if missing:
        raise ValueError(f"Missing runtime features: {sorted(missing)}")
    return orders.loc[:, list(RUNTIME_FEATURES)].copy()

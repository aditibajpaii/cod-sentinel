import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from cod_sentinel.features import (
    RUNTIME_FEATURES,
    add_prior_history_features,
    select_runtime_features,
)


def _raw_history_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order_id": "ORDER-1",
                "ordered_at": pd.Timestamp("2025-01-01T10:00:00"),
                "customer_id": "CUSTOMER-1",
                "pincode": "ZONE-001",
                "order_value": 500.0,
                "outcome_observed_at": pd.Timestamp("2025-01-01T18:00:00"),
                "logged_action": "COD",
                "logged_converted": 1,
                "logged_delivered": 0,
                "logged_rto": 1,
            },
            {
                "order_id": "ORDER-2",
                "ordered_at": pd.Timestamp("2025-01-02T10:00:00"),
                "customer_id": "CUSTOMER-1",
                "pincode": "ZONE-001",
                "order_value": 1_000.0,
                "outcome_observed_at": pd.Timestamp("2025-01-04T10:00:00"),
                "logged_action": "PREPAID",
                "logged_converted": 1,
                "logged_delivered": 1,
                "logged_rto": 0,
            },
        ]
    )


def test_history_features_use_prior_rows_only() -> None:
    featured = add_prior_history_features(_raw_history_rows())

    first = featured.loc[featured["order_id"] == "ORDER-1"].iloc[0]
    second = featured.loc[featured["order_id"] == "ORDER-2"].iloc[0]
    assert first["customer_prior_orders"] == 0
    assert first["customer_prior_rto_rate"] == 0.0
    assert second["customer_prior_orders"] == 1
    assert second["customer_prior_rto_rate"] == 1.0
    assert second["customer_order_value_ratio"] == 2.0


def test_same_timestamp_rows_do_not_observe_each_other() -> None:
    rows = _raw_history_rows()
    rows.loc[1, "ordered_at"] = rows.loc[0, "ordered_at"]

    featured = add_prior_history_features(rows)

    assert featured["customer_prior_orders"].tolist() == [0, 0]
    assert featured["pincode_prior_orders"].tolist() == [0, 0]


def test_unresolved_prior_outcome_is_not_visible() -> None:
    rows = _raw_history_rows()
    rows.loc[0, "outcome_observed_at"] = pd.Timestamp("2025-01-03T10:00:00")

    featured = add_prior_history_features(rows)
    second = featured.loc[featured["order_id"] == "ORDER-2"].iloc[0]

    assert second["customer_prior_orders"] == 1
    assert second["customer_prior_rto_rate"] == 0.0


def test_appending_future_rows_does_not_change_historical_features() -> None:
    original_rows = _raw_history_rows()
    original = add_prior_history_features(original_rows)
    future = original_rows.iloc[[1]].copy()
    future["order_id"] = "ORDER-3"
    future["ordered_at"] = pd.to_datetime(["2025-12-31T10:00:00"])
    future["outcome_observed_at"] = pd.to_datetime(["2026-01-05T10:00:00"])
    expanded = add_prior_history_features(
        pd.concat([original_rows, future], ignore_index=True)
    )

    history_columns = [
        column
        for column in original.columns
        if column.startswith("customer_prior_")
        or column.startswith("pincode_prior_")
        or column == "customer_order_value_ratio"
    ]
    assert_frame_equal(
        original.set_index("order_id").loc[:, history_columns],
        expanded.set_index("order_id")
        .loc[original["order_id"], history_columns],
        check_dtype=False,
    )


def test_runtime_feature_selection_uses_exact_allowlist(synthetic_world) -> None:
    observable, _, _ = synthetic_world

    selected = select_runtime_features(observable)

    assert tuple(selected.columns) == RUNTIME_FEATURES


def test_runtime_feature_selection_rejects_oracle_contamination(
    synthetic_world,
) -> None:
    observable, oracle, _ = synthetic_world
    contaminated = observable.merge(
        oracle[["order_id", "latent_intent"]],
        on="order_id",
        validate="one_to_one",
    )

    with pytest.raises(ValueError, match="Oracle columns"):
        select_runtime_features(contaminated)

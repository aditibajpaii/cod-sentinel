import numpy as np
import pandas as pd
import pytest

from cod_sentinel.configuration import PolicySettings
from cod_sentinel.evaluation import (
    _oracle_actions_and_values,
    _realized_policy_values,
    customer_cluster_bootstrap_interval,
    realized_action_contribution,
)
from cod_sentinel.schemas import Action, MerchantEconomics


@pytest.fixture
def evaluation_settings() -> PolicySettings:
    return PolicySettings(
        merchant_economics=MerchantEconomics(
            gross_margin_rate=0.30,
            forward_shipping_cost=50.0,
            reverse_shipping_cost=70.0,
            packaging_cost=10.0,
            otp_verification_cost=3.0,
        ),
        tie_break_order=(Action.COD, Action.OTP, Action.PREPAID),
        config_version="evaluation-test-v1",
    )


def test_realized_evaluation_uses_selected_potential_outcome(
    evaluation_settings: PolicySettings,
) -> None:
    oracle = pd.Series(
        {
            "cod_rto": 1,
            "otp_completed": 1,
            "otp_rto_if_completed": 0,
            "prepaid_converted": 0,
            "prepaid_failed_delivery": 0,
        }
    )

    cod = realized_action_contribution(
        Action.COD,
        1_000.0,
        oracle,
        evaluation_settings,
    )
    otp = realized_action_contribution(
        Action.OTP,
        1_000.0,
        oracle,
        evaluation_settings,
    )
    prepaid = realized_action_contribution(
        Action.PREPAID,
        1_000.0,
        oracle,
        evaluation_settings,
    )

    assert cod == pytest.approx(-130.0)
    assert otp == pytest.approx(237.0)
    assert prepaid == pytest.approx(0.0)


def test_oracle_is_per_order_upper_bound(
    evaluation_settings: PolicySettings,
) -> None:
    observable = pd.DataFrame(
        {
            "order_id": ["A", "B"],
            "order_value": [1_000.0, 500.0],
        }
    )
    oracle = pd.DataFrame(
        {
            "order_id": ["A", "B"],
            "cod_rto": [1, 0],
            "otp_completed": [1, 0],
            "otp_rto_if_completed": [0, 0],
            "prepaid_converted": [0, 1],
            "prepaid_failed_delivery": [0, 0],
        }
    )
    _, oracle_values = _oracle_actions_and_values(
        observable,
        oracle,
        evaluation_settings,
    )

    for action in Action:
        fixed = _realized_policy_values(
            [action, action],
            observable,
            oracle,
            evaluation_settings,
        )
        assert np.all(oracle_values >= fixed)


def test_customer_cluster_bootstrap_is_deterministic() -> None:
    customer_ids = np.array(["A", "A", "B", "C", "C", "C"])
    differences = np.array([1.0, 2.0, -1.0, 3.0, 4.0, 5.0])

    first = customer_cluster_bootstrap_interval(
        customer_ids,
        differences,
        random_seed=42,
        samples=200,
    )
    second = customer_cluster_bootstrap_interval(
        customer_ids,
        differences,
        random_seed=42,
        samples=200,
    )

    assert first == second
    assert first[0] <= differences.mean() <= first[1]


def test_cluster_bootstrap_rejects_invalid_sample_count() -> None:
    with pytest.raises(ValueError, match="samples must be positive"):
        customer_cluster_bootstrap_interval(
            np.array(["A"]),
            np.array([1.0]),
            random_seed=42,
            samples=0,
        )

from dataclasses import replace
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cod_sentinel.economics import (
    cod_break_even_rto_probability,
    expected_contribution,
    max_profitable_prepaid_discount,
    realized_contribution,
)
from cod_sentinel.schemas import (
    Action,
    ActionProbabilities,
    MerchantEconomics,
    OrderEconomics,
    RealizedOutcome,
)


@pytest.fixture
def order() -> OrderEconomics:
    return OrderEconomics(order_value=1_000.0)


@pytest.fixture
def merchant() -> MerchantEconomics:
    return MerchantEconomics(
        gross_margin_rate=0.30,
        forward_shipping_cost=50.0,
        reverse_shipping_cost=70.0,
        packaging_cost=10.0,
        other_fulfillment_cost=5.0,
        cod_fee_rate=0.02,
        cod_fee_flat=5.0,
        otp_verification_cost=3.0,
        prepaid_fee_rate=0.02,
        prepaid_fee_flat=1.0,
        prepaid_discount_rate=0.05,
        prepaid_refund_cost=4.0,
        inventory_damage_rate=0.10,
        damaged_inventory_recovery_rate=0.20,
    )


@pytest.mark.parametrize(
    ("action", "outcome", "expected"),
    [
        (Action.COD, RealizedOutcome(True, True), 210.0),
        (Action.COD, RealizedOutcome(True, False), -191.0),
        (Action.OTP, RealizedOutcome(False, False), -3.0),
        (Action.OTP, RealizedOutcome(True, True), 207.0),
        (Action.OTP, RealizedOutcome(True, False), -194.0),
        (Action.PREPAID, RealizedOutcome(False, False), 0.0),
        (Action.PREPAID, RealizedOutcome(True, True), 165.0),
        (Action.PREPAID, RealizedOutcome(True, False), -215.0),
    ],
)
def test_realized_branches_share_consistent_cost_timing(
    action: Action,
    outcome: RealizedOutcome,
    expected: float,
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> None:
    assert realized_contribution(action, order, merchant, outcome) == pytest.approx(
        expected
    )


def test_expected_contribution_weights_realized_branches(
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> None:
    probabilities = ActionProbabilities(
        conversion_probability=0.80,
        delivery_probability=0.75,
    )

    expected = 0.20 * -3.0 + 0.80 * 0.75 * 207.0 + 0.80 * 0.25 * -194.0

    assert expected_contribution(
        Action.OTP,
        order,
        merchant,
        probabilities,
    ) == pytest.approx(expected)


def test_cod_requires_unit_conversion_probability(
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> None:
    with pytest.raises(ValueError, match="must be 1"):
        expected_contribution(
            Action.COD,
            order,
            merchant,
            ActionProbabilities(
                conversion_probability=0.99,
                delivery_probability=0.8,
            ),
        )


def test_invalid_action_is_rejected_instead_of_treated_as_prepaid(
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> None:
    with pytest.raises(ValueError, match="Unsupported economic action"):
        realized_contribution(
            cast(Action, "INVALID"),
            order,
            merchant,
            RealizedOutcome(converted=True, delivered=True),
        )


def test_cod_expected_contribution_falls_as_rto_risk_rises(
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> None:
    low_risk = expected_contribution(
        Action.COD,
        order,
        merchant,
        ActionProbabilities.for_cod(0.10),
    )
    high_risk = expected_contribution(
        Action.COD,
        order,
        merchant,
        ActionProbabilities.for_cod(0.80),
    )

    assert low_risk > high_risk


def test_break_even_probability_zeroes_shared_cod_model(
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> None:
    break_even = cod_break_even_rto_probability(order, merchant)

    assert break_even == pytest.approx(210.0 / 401.0)
    assert expected_contribution(
        Action.COD,
        order,
        merchant,
        ActionProbabilities.for_cod(break_even),
    ) == pytest.approx(0.0, abs=1e-10)


def test_break_even_reproduces_reviewed_numeric_examples() -> None:
    low_value = cod_break_even_rto_probability(
        OrderEconomics(order_value=400.0),
        MerchantEconomics(
            gross_margin_rate=0.30,
            forward_shipping_cost=60.0,
            reverse_shipping_cost=60.0,
            packaging_cost=0.0,
        ),
    )
    high_value = cod_break_even_rto_probability(
        OrderEconomics(order_value=5_000.0),
        MerchantEconomics(
            gross_margin_rate=0.10,
            forward_shipping_cost=100.0,
            reverse_shipping_cost=100.0,
            packaging_cost=0.0,
        ),
    )

    assert low_value == pytest.approx(1.0 / 3.0)
    assert high_value == pytest.approx(2.0 / 3.0)


def test_negative_break_even_marks_cod_never_profitable() -> None:
    break_even = cod_break_even_rto_probability(
        OrderEconomics(order_value=300.0),
        MerchantEconomics(
            gross_margin_rate=0.30,
            forward_shipping_cost=60.0,
            reverse_shipping_cost=60.0,
            packaging_cost=40.0,
        ),
    )

    assert break_even < 0.0


def test_break_even_rejects_perverse_delivered_economics() -> None:
    with pytest.raises(ValueError, match="must exceed RTO"):
        cod_break_even_rto_probability(
            OrderEconomics(order_value=100.0),
            MerchantEconomics(
                gross_margin_rate=0.10,
                forward_shipping_cost=0.0,
                reverse_shipping_cost=0.0,
                packaging_cost=0.0,
                cod_fee_flat=200.0,
            ),
        )


def test_prepaid_discount_uses_discounted_revenue_and_fee_base(
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> None:
    original = realized_contribution(
        Action.PREPAID,
        order,
        merchant,
        RealizedOutcome(converted=True, delivered=True),
    )
    larger_discount = realized_contribution(
        Action.PREPAID,
        order,
        replace(merchant, prepaid_discount_rate=0.10),
        RealizedOutcome(converted=True, delivered=True),
    )

    assert original - larger_discount == pytest.approx(49.0)


def test_damage_recovery_only_changes_failed_delivery_branch(
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> None:
    fully_recovered = replace(merchant, damaged_inventory_recovery_rate=1.0)

    failed_delta = realized_contribution(
        Action.COD,
        order,
        fully_recovered,
        RealizedOutcome(converted=True, delivered=False),
    ) - realized_contribution(
        Action.COD,
        order,
        merchant,
        RealizedOutcome(converted=True, delivered=False),
    )
    delivered_delta = realized_contribution(
        Action.COD,
        order,
        fully_recovered,
        RealizedOutcome(converted=True, delivered=True),
    ) - realized_contribution(
        Action.COD,
        order,
        merchant,
        RealizedOutcome(converted=True, delivered=True),
    )

    assert failed_delta == pytest.approx(56.0)
    assert delivered_delta == pytest.approx(0.0)


@given(
    conversion=st.floats(min_value=0.0, max_value=1.0),
    delivery=st.floats(min_value=0.0, max_value=1.0),
)
def test_expected_otp_is_convex_combination_of_realized_branches(
    conversion: float,
    delivery: float,
) -> None:
    order = OrderEconomics(order_value=1_000.0)
    merchant = MerchantEconomics(
        gross_margin_rate=0.30,
        forward_shipping_cost=50.0,
        reverse_shipping_cost=70.0,
        packaging_cost=10.0,
        otp_verification_cost=3.0,
    )
    branches = [
        realized_contribution(
            Action.OTP,
            order,
            merchant,
            RealizedOutcome(converted=False, delivered=False),
        ),
        realized_contribution(
            Action.OTP,
            order,
            merchant,
            RealizedOutcome(converted=True, delivered=True),
        ),
        realized_contribution(
            Action.OTP,
            order,
            merchant,
            RealizedOutcome(converted=True, delivered=False),
        ),
    ]

    expected = expected_contribution(
        Action.OTP,
        order,
        merchant,
        ActionProbabilities(conversion, delivery),
    )

    assert min(branches) <= expected <= max(branches)


def test_max_profitable_prepaid_discount_returns_zero_when_unprofitable(
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> None:
    prepaid_probs = ActionProbabilities(
        conversion_probability=0.10,
        delivery_probability=0.50,
    )
    fallback_probs = ActionProbabilities.for_cod(0.20)

    assert (
        max_profitable_prepaid_discount(
            order,
            merchant,
            prepaid_probs,
            Action.COD,
            fallback_probs,
        )
        == 0.0
    )


def test_max_profitable_prepaid_discount_respects_cap_and_floor(
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> None:
    prepaid_probs = ActionProbabilities(
        conversion_probability=0.90,
        delivery_probability=0.90,
    )
    fallback_probs = ActionProbabilities.for_cod(0.60)

    max_rate = max_profitable_prepaid_discount(
        order,
        merchant,
        prepaid_probs,
        Action.COD,
        fallback_probs,
        cap=0.05,
    )

    assert 0.0 <= max_rate <= 0.05
    adjusted = replace(merchant, prepaid_discount_rate=max_rate)
    assert expected_contribution(
        Action.PREPAID,
        order,
        adjusted,
        prepaid_probs,
    ) >= max(
        expected_contribution(Action.COD, order, merchant, fallback_probs),
        0.0,
    )

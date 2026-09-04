import math

import pytest

from cod_sentinel.schemas import (
    Action,
    ActionProbabilities,
    MerchantEconomics,
    OrderEconomics,
    RealizedOutcome,
)


def test_review_is_not_an_economic_action() -> None:
    assert {action.value for action in Action} == {"COD", "OTP", "PREPAID"}


@pytest.mark.parametrize("value", [0.0, -1.0, math.nan, math.inf])
def test_order_value_must_be_positive_and_finite(value: float) -> None:
    with pytest.raises(ValueError):
        OrderEconomics(order_value=value)


@pytest.mark.parametrize("value", [-0.01, 1.01, math.nan, math.inf])
def test_action_probabilities_must_be_valid(value: float) -> None:
    with pytest.raises(ValueError):
        ActionProbabilities(
            conversion_probability=value,
            delivery_probability=0.5,
        )


def test_cod_probability_factory_converts_rto_to_delivery() -> None:
    probabilities = ActionProbabilities.for_cod(rto_probability=0.27)

    assert probabilities.conversion_probability == 1.0
    assert probabilities.delivery_probability == pytest.approx(0.73)


def test_delivered_requires_conversion() -> None:
    with pytest.raises(ValueError, match="cannot be delivered"):
        RealizedOutcome(converted=False, delivered=True)


def test_merchant_rates_must_be_probabilities() -> None:
    with pytest.raises(ValueError, match="gross_margin_rate"):
        MerchantEconomics(
            gross_margin_rate=1.2,
            forward_shipping_cost=60.0,
            reverse_shipping_cost=80.0,
            packaging_cost=10.0,
        )


def test_merchant_costs_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="reverse_shipping_cost"):
        MerchantEconomics(
            gross_margin_rate=0.3,
            forward_shipping_cost=60.0,
            reverse_shipping_cost=-1.0,
            packaging_cost=10.0,
        )

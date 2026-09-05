"""Tests for post-agent economics: what the agent's outcome was worth."""

import pytest

from cod_sentinel.economics import max_profitable_prepaid_discount
from cod_sentinel.orchestrator.recovery import evaluate_recovery
from cod_sentinel.schemas import Action, ActionProbabilities, MerchantEconomics


@pytest.fixture
def merchant() -> MerchantEconomics:
    return MerchantEconomics(
        gross_margin_rate=0.30,
        forward_shipping_cost=60.0,
        reverse_shipping_cost=80.0,
        packaging_cost=10.0,
        otp_verification_cost=2.0,
    )


def _probabilities(cod_rto: float = 0.70) -> dict[str, float]:
    return {
        "cod_rto": cod_rto,
        "otp_completion": 0.80,
        "otp_rto": 0.20,
        "prepaid_conversion": 0.60,
        "prepaid_failure": 0.05,
    }


def test_conversion_to_prepaid_recovers_margin(merchant: MerchantEconomics) -> None:
    """A high-RTO COD order converted to prepaid must gain expected value."""

    outcome = evaluate_recovery(
        order_value=2_000.0,
        merchant=merchant,
        probabilities=_probabilities(cod_rto=0.70),
        baseline_action="COD",
        achieved_action="PREPAID",
        discount_rate=0.02,
    )

    assert outcome is not None
    assert outcome.baseline_action == "COD"
    assert outcome.achieved_action == "PREPAID"
    assert outcome.recovered_margin > 0.0
    assert outcome.recovered_margin == pytest.approx(
        outcome.achieved_ev - outcome.baseline_ev
    )


def test_no_conversion_recovers_exactly_nothing(merchant: MerchantEconomics) -> None:
    """When the agent fails, the order is worth what it was worth."""

    outcome = evaluate_recovery(
        order_value=2_000.0,
        merchant=merchant,
        probabilities=_probabilities(),
        baseline_action="OTP",
        achieved_action="OTP",
    )

    assert outcome is not None
    assert outcome.recovered_margin == pytest.approx(0.0)
    assert outcome.action_changed is False


def test_renegotiating_a_discount_is_not_recovery(
    merchant: MerchantEconomics,
) -> None:
    """An order the policy already routed to prepaid earns the agent nothing.

    Pricing the same action at a smaller negotiated discount would show a gain
    the agent did not create — the buyer was never offered the configured terms.
    """

    outcome = evaluate_recovery(
        order_value=2_000.0,
        merchant=merchant,
        probabilities=_probabilities(),
        baseline_action="PREPAID",
        achieved_action="PREPAID",
        discount_rate=0.001,  # far below the configured discount
    )

    assert outcome is not None
    assert outcome.action_changed is False
    assert outcome.recovered_margin == pytest.approx(0.0)
    assert outcome.discount_rate is None


def test_discount_at_the_ceiling_cannot_show_a_gain(
    merchant: MerchantEconomics,
) -> None:
    """The economics ceiling is the break-even point, by construction.

    A conversion at exactly max_profitable_prepaid_discount must not look
    profitable relative to the fallback — if it did, the ceiling would be wrong.
    """

    from cod_sentinel.schemas import OrderEconomics

    probabilities = _probabilities(cod_rto=0.70)
    order = OrderEconomics(order_value=2_000.0)
    prepaid = ActionProbabilities(
        conversion_probability=probabilities["prepaid_conversion"],
        delivery_probability=1.0 - probabilities["prepaid_failure"],
    )
    fallback = ActionProbabilities.for_cod(probabilities["cod_rto"])
    ceiling = max_profitable_prepaid_discount(
        order,
        merchant,
        prepaid,
        Action.COD,
        fallback,
        cap=0.95,
    )

    outcome = evaluate_recovery(
        order_value=2_000.0,
        merchant=merchant,
        probabilities=probabilities,
        baseline_action="COD",
        achieved_action="PREPAID",
        discount_rate=ceiling,
    )

    assert outcome is not None
    assert outcome.recovered_margin >= -1e-6
    assert outcome.recovered_margin < 1.0  # at break-even, not a windfall


def test_review_is_not_priced(merchant: MerchantEconomics) -> None:
    """REVIEW is not an economic action, so there is nothing to compare."""

    assert (
        evaluate_recovery(
            order_value=1_000.0,
            merchant=merchant,
            probabilities=_probabilities(),
            baseline_action="REVIEW",
            achieved_action="REVIEW",
        )
        is None
    )


def test_missing_probabilities_are_not_priced(merchant: MerchantEconomics) -> None:
    assert (
        evaluate_recovery(
            order_value=1_000.0,
            merchant=merchant,
            probabilities={},
            baseline_action="COD",
            achieved_action="PREPAID",
            discount_rate=0.02,
        )
        is None
    )

"""Shared state-transition economics for COD, OTP, and prepaid actions."""

from dataclasses import dataclass
from math import isclose

from cod_sentinel.schemas import (
    Action,
    ActionProbabilities,
    MerchantEconomics,
    OrderEconomics,
    RealizedOutcome,
)


@dataclass(frozen=True, slots=True)
class _ActionTerms:
    attempt_cost: float
    converted_common_cost: float
    delivered_revenue: float
    delivered_payment_cost: float
    failed_delivery_cost: float


def _action_terms(
    action: Action,
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> _ActionTerms:
    """Translate action-specific fees into common state-transition terms."""

    if not isinstance(action, Action):
        raise ValueError(f"Unsupported economic action: {action!r}.")

    value = order.order_value
    product_cost = value * (1.0 - merchant.gross_margin_rate)
    fulfillment_cost = (
        merchant.forward_shipping_cost
        + merchant.packaging_cost
        + merchant.other_fulfillment_cost
    )
    expected_inventory_loss = (
        product_cost
        * merchant.inventory_damage_rate
        * (1.0 - merchant.damaged_inventory_recovery_rate)
    )

    if action in (Action.COD, Action.OTP):
        return _ActionTerms(
            attempt_cost=(
                merchant.otp_verification_cost if action is Action.OTP else 0.0
            ),
            converted_common_cost=fulfillment_cost,
            delivered_revenue=value,
            delivered_payment_cost=(
                merchant.cod_fee_flat + merchant.cod_fee_rate * value
            ),
            failed_delivery_cost=(
                merchant.reverse_shipping_cost + expected_inventory_loss
            ),
        )

    discounted_revenue = value * (1.0 - merchant.prepaid_discount_rate)
    prepaid_processing_cost = (
        merchant.prepaid_fee_flat
        + merchant.prepaid_fee_rate * discounted_revenue
    )
    return _ActionTerms(
        attempt_cost=0.0,
        converted_common_cost=fulfillment_cost + prepaid_processing_cost,
        delivered_revenue=discounted_revenue,
        delivered_payment_cost=0.0,
        failed_delivery_cost=(
            merchant.reverse_shipping_cost
            + merchant.prepaid_refund_cost
            + expected_inventory_loss
        ),
    )


def _validate_action_probabilities(
    action: Action,
    probabilities: ActionProbabilities,
) -> None:
    if action is Action.COD and not isclose(
        probabilities.conversion_probability,
        1.0,
        abs_tol=1e-12,
    ):
        raise ValueError("COD conversion_probability must be 1.")


def realized_contribution(
    action: Action,
    order: OrderEconomics,
    merchant: MerchantEconomics,
    outcome: RealizedOutcome,
) -> float:
    """Return contribution for one realized action branch.

    Cost timing:

    - OTP verification is paid whenever OTP is attempted.
    - Fulfillment is paid only after action completion/conversion.
    - COD collection fees are paid only after successful delivery.
    - Prepaid processing fees are paid on conversion and are not refunded.
    - Failed shipped orders incur reverse shipping, expected inventory loss,
      and, for prepaid, the configured refund processing cost.
    """

    if action is Action.COD and not outcome.converted:
        raise ValueError("COD is already accepted and cannot be non-converted.")

    terms = _action_terms(action, order, merchant)
    contribution = -terms.attempt_cost

    if not outcome.converted:
        return contribution

    contribution -= terms.converted_common_cost
    if outcome.delivered:
        product_cost = order.order_value * (1.0 - merchant.gross_margin_rate)
        contribution += (
            terms.delivered_revenue
            - product_cost
            - terms.delivered_payment_cost
        )
    else:
        contribution -= terms.failed_delivery_cost

    return contribution


def expected_contribution(
    action: Action,
    order: OrderEconomics,
    merchant: MerchantEconomics,
    probabilities: ActionProbabilities,
) -> float:
    """Return probability-weighted contribution across all action branches."""

    _validate_action_probabilities(action, probabilities)

    conversion = probabilities.conversion_probability
    delivery = probabilities.delivery_probability
    expected = 0.0

    if conversion < 1.0:
        expected += (1.0 - conversion) * realized_contribution(
            action,
            order,
            merchant,
            RealizedOutcome(converted=False, delivered=False),
        )

    expected += conversion * delivery * realized_contribution(
        action,
        order,
        merchant,
        RealizedOutcome(converted=True, delivered=True),
    )
    expected += conversion * (1.0 - delivery) * realized_contribution(
        action,
        order,
        merchant,
        RealizedOutcome(converted=True, delivered=False),
    )
    return expected


def cod_break_even_rto_probability(
    order: OrderEconomics,
    merchant: MerchantEconomics,
) -> float:
    """Return the unbounded COD RTO probability where contribution is zero.

    A result at or below zero means delivered COD is already non-profitable.
    A result of one means only a certain RTO removes all expected contribution.
    The function rejects configurations where delivery is not economically
    preferable to RTO because a standard break-even-risk interpretation would
    be misleading.
    """

    delivered = realized_contribution(
        Action.COD,
        order,
        merchant,
        RealizedOutcome(converted=True, delivered=True),
    )
    rto = realized_contribution(
        Action.COD,
        order,
        merchant,
        RealizedOutcome(converted=True, delivered=False),
    )

    if delivered <= rto:
        raise ValueError(
            "COD delivered contribution must exceed RTO contribution to "
            "derive a break-even risk."
        )
    return delivered / (delivered - rto)

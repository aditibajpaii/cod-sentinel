"""Economics tool wrapping the deterministic decision engine."""

from dataclasses import asdict
from typing import Any, Mapping

from cod_sentinel.economics import max_profitable_prepaid_discount
from cod_sentinel.policy import DecisionEngine
from cod_sentinel.schemas import Action, ActionProbabilities, OrderEconomics


class EconomicsTool:
    """Run frozen policy economics and derive negotiation bounds."""

    def __init__(self, engine: DecisionEngine) -> None:
        self._engine = engine
        self._settings = engine.settings

    def run(self, order: Mapping[str, object]) -> dict[str, Any]:
        decision = self._engine.decide(order)
        if decision.requires_review or decision.probabilities is None:
            return {
                "decision": decision.to_dict(),
                "cod_rto": None,
                "unit_economics": None,
                "max_prepaid_discount_rate": 0.0,
                "fallback_action": "REVIEW",
            }

        probabilities = decision.probabilities
        order_economics = OrderEconomics(order_value=float(order["order_value"]))
        merchant = self._settings.merchant_economics
        selected = Action(decision.selected_action)
        prepaid_probs = ActionProbabilities(
            conversion_probability=probabilities["prepaid_conversion"],
            delivery_probability=1.0 - probabilities["prepaid_failure"],
        )
        fallback_probs = self._fallback_probabilities(selected, probabilities)
        max_discount = max_profitable_prepaid_discount(
            order_economics,
            merchant,
            prepaid_probs,
            selected,
            fallback_probs,
        )

        value = order_economics.order_value
        margin = value * merchant.gross_margin_rate
        cod_fee = merchant.cod_fee_flat + merchant.cod_fee_rate * value

        return {
            "decision": decision.to_dict(),
            "cod_rto": probabilities["cod_rto"],
            "unit_economics": {
                "order_value": value,
                "expected_margin": margin,
                "forward_shipping_cost": merchant.forward_shipping_cost,
                "reverse_shipping_cost": merchant.reverse_shipping_cost,
                "cod_handling_cost": cod_fee,
                "expected_contributions": decision.expected_contributions,
            },
            "max_prepaid_discount_rate": max_discount,
            "fallback_action": decision.selected_action,
        }

    @staticmethod
    def _fallback_probabilities(
        action: Action,
        probabilities: Mapping[str, float],
    ) -> ActionProbabilities:
        if action is Action.COD:
            return ActionProbabilities.for_cod(probabilities["cod_rto"])
        if action is Action.OTP:
            return ActionProbabilities(
                conversion_probability=probabilities["otp_completion"],
                delivery_probability=1.0 - probabilities["otp_rto"],
            )
        return ActionProbabilities(
            conversion_probability=probabilities["prepaid_conversion"],
            delivery_probability=1.0 - probabilities["prepaid_failure"],
        )

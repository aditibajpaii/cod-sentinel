"""What the agent's outcome did to the order's expected contribution.

This closes the loop between the recovery agent and the economics core: after
the agent acts, the order is re-priced under the action it actually achieved and
compared against the action the deterministic policy would have taken on its
own. The difference is the margin the agent recovered.

Every figure here is an **expected** contribution computed from calibrated model
probabilities at decision time. It is not realized profit, and it is not the
held-out evaluation in `results/metrics.json`, which scores frozen policies
against simulator potential outcomes. Keep the two separate.
"""

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping

from cod_sentinel.economics import expected_contribution
from cod_sentinel.schemas import (
    Action,
    ActionProbabilities,
    MerchantEconomics,
    OrderEconomics,
)


@dataclass(frozen=True, slots=True)
class RecoveryOutcome:
    """Expected contribution before and after the agent intervened."""

    baseline_action: str
    baseline_ev: float
    achieved_action: str
    achieved_ev: float
    recovered_margin: float
    action_changed: bool = False
    discount_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def action_probabilities(
    action: Action,
    probabilities: Mapping[str, float],
) -> ActionProbabilities:
    """Map calibrated outcome probabilities onto one action's branch pair."""

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


def evaluate_recovery(
    *,
    order_value: float,
    merchant: MerchantEconomics,
    probabilities: Mapping[str, float],
    baseline_action: str,
    achieved_action: str,
    discount_rate: float | None = None,
) -> RecoveryOutcome | None:
    """Re-price an order under the agent's achieved action.

    Returns ``None`` when either action is not an economic action (REVIEW), or
    when probabilities are unavailable — there is nothing meaningful to price.
    """

    try:
        baseline = Action(baseline_action)
        achieved = Action(achieved_action)
    except ValueError:
        return None
    if not probabilities:
        return None

    order = OrderEconomics(order_value=float(order_value))
    baseline_ev = expected_contribution(
        baseline,
        order,
        merchant,
        action_probabilities(baseline, probabilities),
    )

    # Recovery is only claimed when the agent changed the action. If the
    # deterministic policy was already going to take this action, re-pricing it
    # at a negotiated discount would flatter the agent with a gain it did not
    # produce — the counterfactual "the policy would have used the configured
    # discount" is a static assumption, not a per-order decision, and the buyer
    # was never offered those terms.
    action_changed = baseline is not achieved

    achieved_merchant = merchant
    if action_changed and achieved is Action.PREPAID and discount_rate is not None:
        achieved_merchant = replace(merchant, prepaid_discount_rate=float(discount_rate))
    achieved_ev = expected_contribution(
        achieved,
        order,
        achieved_merchant,
        action_probabilities(achieved, probabilities),
    )

    return RecoveryOutcome(
        baseline_action=baseline.value,
        baseline_ev=float(baseline_ev),
        achieved_action=achieved.value,
        achieved_ev=float(achieved_ev),
        recovered_margin=float(achieved_ev - baseline_ev),
        action_changed=action_changed,
        discount_rate=(
            float(discount_rate)
            if action_changed and discount_rate is not None
            else None
        ),
    )

"""Typed inputs for the shared economic state model."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite


def _require_finite(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite.")


def _require_non_negative(name: str, value: float) -> None:
    _require_finite(name, value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _require_probability(name: str, value: float) -> None:
    _require_finite(name, value)
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


class Action(str, Enum):
    """Actions that participate in economic optimization."""

    COD = "COD"
    OTP = "OTP"
    PREPAID = "PREPAID"


@dataclass(frozen=True, slots=True)
class OrderEconomics:
    """Order-specific inputs known when the policy runs."""

    order_value: float

    def __post_init__(self) -> None:
        _require_finite("order_value", self.order_value)
        if self.order_value <= 0:
            raise ValueError("order_value must be positive.")


@dataclass(frozen=True, slots=True)
class MerchantEconomics:
    """Merchant cost assumptions shared by every action.

    ``gross_margin_rate`` is defined at the undiscounted order value. Product
    cost is therefore ``order_value * (1 - gross_margin_rate)``.

    ``inventory_damage_rate`` is the probability that inventory from a failed
    delivery is damaged. ``damaged_inventory_recovery_rate`` is the fraction
    of product cost recovered even when that damage occurs.
    """

    gross_margin_rate: float
    forward_shipping_cost: float
    reverse_shipping_cost: float
    packaging_cost: float
    other_fulfillment_cost: float = 0.0
    cod_fee_rate: float = 0.0
    cod_fee_flat: float = 0.0
    otp_verification_cost: float = 0.0
    prepaid_fee_rate: float = 0.0
    prepaid_fee_flat: float = 0.0
    prepaid_discount_rate: float = 0.0
    prepaid_refund_cost: float = 0.0
    inventory_damage_rate: float = 0.0
    damaged_inventory_recovery_rate: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "gross_margin_rate",
            "cod_fee_rate",
            "prepaid_fee_rate",
            "prepaid_discount_rate",
            "inventory_damage_rate",
            "damaged_inventory_recovery_rate",
        ):
            _require_probability(name, getattr(self, name))

        for name in (
            "forward_shipping_cost",
            "reverse_shipping_cost",
            "packaging_cost",
            "other_fulfillment_cost",
            "cod_fee_flat",
            "otp_verification_cost",
            "prepaid_fee_flat",
            "prepaid_refund_cost",
        ):
            _require_non_negative(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class ActionProbabilities:
    """Conditional action outcomes available to the runtime policy.

    ``conversion_probability`` is the chance that the customer completes the
    action and the order ships. ``delivery_probability`` is conditional on
    conversion and represents successful delivery rather than RTO/return.
    """

    conversion_probability: float
    delivery_probability: float

    def __post_init__(self) -> None:
        _require_probability("conversion_probability", self.conversion_probability)
        _require_probability("delivery_probability", self.delivery_probability)

    @classmethod
    def for_cod(cls, rto_probability: float) -> "ActionProbabilities":
        _require_probability("rto_probability", rto_probability)
        return cls(
            conversion_probability=1.0,
            delivery_probability=1.0 - rto_probability,
        )


@dataclass(frozen=True, slots=True)
class RealizedOutcome:
    """One realized branch used by simulator evaluation."""

    converted: bool
    delivered: bool

    def __post_init__(self) -> None:
        if self.delivered and not self.converted:
            raise ValueError("A non-converted order cannot be delivered.")

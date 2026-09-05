"""Typed inputs and outputs for the agent orchestrator."""

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from cod_sentinel.features import ORACLE_DENYLIST, RUNTIME_FEATURES

PaymentMethod = Literal["COD", "PREPAID"]
NegotiationStatus = Literal["AGREED", "DECLINED", "UNREACHABLE", "IN_PROGRESS"]
FinalStatus = Literal[
    "PENDING_PAYMENT_CONFIRMATION",
    "FALLBACK_ACTION",
    "REVIEW",
    "NO_NEGOTIATION_NEEDED",
]


@dataclass(frozen=True, slots=True)
class CheckoutEvent:
    """Runtime checkout payload for the orchestrator."""

    order_id: str
    payment_method: PaymentMethod
    raw_address: str
    buyer_phone: str
    order_features: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.order_id.strip():
            raise ValueError("order_id must be non-empty.")
        if not self.raw_address.strip():
            raise ValueError("raw_address must be non-empty.")
        if not self.buyer_phone.strip():
            raise ValueError("buyer_phone must be non-empty.")
        for key in ORACLE_DENYLIST:
            if key in self.order_features:
                raise ValueError(f"Oracle field {key!r} is not allowed at runtime.")
        for key in self.order_features:
            if key not in RUNTIME_FEATURES and key != "order_id":
                raise ValueError(f"Unknown runtime feature {key!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "payment_method": self.payment_method,
            "raw_address": self.raw_address,
            "buyer_phone": self.buyer_phone,
            "order_features": dict(self.order_features),
        }


@dataclass(frozen=True, slots=True)
class AgentStep:
    """One auditable step in an orchestrator run."""

    step: int
    tool: str
    model: str | None
    input_summary: str
    output_summary: str
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AgentRunResult:
    """Terminal outcome of an orchestrator run."""

    order_id: str
    final_status: FinalStatus
    selected_action: str
    steps: list[AgentStep] = field(default_factory=list)
    payment_link_url: str | None = None
    payment_link_id: str | None = None
    agreed_discount_rate: float | None = None
    reason_codes: tuple[str, ...] = ()
    economics_decision_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "order_id": self.order_id,
            "final_status": self.final_status,
            "selected_action": self.selected_action,
            "steps": [step.to_dict() for step in self.steps],
            "payment_link_url": self.payment_link_url,
            "payment_link_id": self.payment_link_id,
            "agreed_discount_rate": self.agreed_discount_rate,
            "reason_codes": list(self.reason_codes),
            "economics_decision_id": self.economics_decision_id,
        }

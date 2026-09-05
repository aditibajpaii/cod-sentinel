"""Orchestrator runner sequencing specialist tools."""

from dataclasses import dataclass, field
from typing import Any

import httpx

from cod_sentinel.orchestrator.audit import append_step
from cod_sentinel.orchestrator.client import create_anthropic_client
from cod_sentinel.orchestrator.credentials import AgentCredentials, load_credentials
from cod_sentinel.orchestrator.schemas import AgentRunResult, AgentStep, CheckoutEvent
from cod_sentinel.orchestrator.tools.address_detective import (
    AddressDetectiveTool,
    address_needs_structuring,
)
from cod_sentinel.orchestrator.tools.call_e_negotiator import CallENegotiatorTool
from cod_sentinel.orchestrator.tools.dynamic_dealmaker import DynamicDealmakerTool
from cod_sentinel.orchestrator.tools.economics import EconomicsTool
from cod_sentinel.policy import DecisionEngine, REVIEW


@dataclass
class OrchestratorRunner:
    """Run the checkout orchestration flow with auditable steps."""

    engine: DecisionEngine
    credentials: AgentCredentials | None = None
    http_client: httpx.Client | None = None
    anthropic_client: Any | None = field(default=None, repr=False)
    _step_counter: int = field(default=0, init=False, repr=False)
    _steps: list[AgentStep] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.credentials is None:
            self.credentials = load_credentials()
        if self.anthropic_client is None and self.credentials.anthropic_api_key:
            self.anthropic_client = create_anthropic_client(
                self.credentials.anthropic_api_key
            )

    def run(
        self,
        event: CheckoutEvent,
        *,
        buyer_reply: str | None = None,
        negotiation_started: bool = False,
    ) -> AgentRunResult:
        self._step_counter = 0
        self._steps = []

        if not self.credentials.complete:
            return self._review_result(
                event,
                reason_codes=("AGENT_CREDENTIALS_UNAVAILABLE",),
            )

        order_row = {**event.order_features, "order_id": event.order_id}
        economics_tool = EconomicsTool(self.engine)
        economics = economics_tool.run(order_row)
        self._log_step(
            event.order_id,
            "economics_tool",
            None,
            f"order_id={event.order_id}",
            str(economics.get("fallback_action")),
            economics.get("decision", {}).get("reason_codes", ()),
        )

        decision = economics["decision"]
        if decision.get("selected_action") == REVIEW:
            return AgentRunResult(
                order_id=event.order_id,
                final_status="REVIEW",
                selected_action=REVIEW,
                steps=list(self._steps),
                reason_codes=tuple(decision.get("reason_codes", ())),
                economics_decision_id=decision.get("decision_id"),
            )

        fallback_action = economics["fallback_action"]
        cod_rto = float(economics["cod_rto"])
        max_discount = float(economics["max_prepaid_discount_rate"])
        order_value = float(economics["unit_economics"]["order_value"])

        if address_needs_structuring(event.raw_address):
            address_tool = AddressDetectiveTool(
                self.credentials,
                client=self.http_client,
                anthropic_client=self.anthropic_client,
            )
            address_result = address_tool.run(event.raw_address)
            self._log_step(
                event.order_id,
                "address_detective",
                "claude-haiku-4-5-20251001",
                event.raw_address[:120],
                str(address_result.get("verified")),
                (address_result.get("reason_code") or "ADDRESS_OK",),
            )
            if not address_result["verified"]:
                return AgentRunResult(
                    order_id=event.order_id,
                    final_status="REVIEW",
                    selected_action=REVIEW,
                    steps=list(self._steps),
                    reason_codes=("ADDRESS_UNVERIFIED",),
                    economics_decision_id=decision.get("decision_id"),
                )

        should_negotiate = (
            event.payment_method == "COD"
            and cod_rto >= self.credentials.cod_rto_threshold
            and max_discount > 0.0
        )
        if not should_negotiate:
            return AgentRunResult(
                order_id=event.order_id,
                final_status="NO_NEGOTIATION_NEEDED",
                selected_action=fallback_action,
                steps=list(self._steps),
                economics_decision_id=decision.get("decision_id"),
            )

        negotiator = CallENegotiatorTool(
            self.credentials,
            client=self.http_client,
            anthropic_client=self.anthropic_client,
        )
        if negotiation_started and buyer_reply:
            neg_result = negotiator.continue_conversation(
                buyer_phone=event.buyer_phone,
                buyer_reply=buyer_reply,
                order_id=event.order_id,
                order_value=order_value,
                max_prepaid_discount_rate=max_discount,
            )
        else:
            neg_result = negotiator.start(
                buyer_phone=event.buyer_phone,
                order_id=event.order_id,
                order_value=order_value,
                max_prepaid_discount_rate=max_discount,
                cod_rto=cod_rto,
            )

        self._log_step(
            event.order_id,
            "call_e_negotiator",
            "claude-sonnet-4-6",
            f"phone={event.buyer_phone}",
            neg_result.get("status", "UNKNOWN"),
            (neg_result.get("reason_code") or "NEGOTIATION",),
        )

        status = neg_result.get("status")
        if status == "IN_PROGRESS":
            return AgentRunResult(
                order_id=event.order_id,
                final_status="FALLBACK_ACTION",
                selected_action=fallback_action,
                steps=list(self._steps),
                reason_codes=("NEGOTIATION_IN_PROGRESS",),
                economics_decision_id=decision.get("decision_id"),
            )

        if status != "AGREED":
            return AgentRunResult(
                order_id=event.order_id,
                final_status="FALLBACK_ACTION",
                selected_action=fallback_action,
                steps=list(self._steps),
                reason_codes=(status or "NEGOTIATION_FAILED",),
                economics_decision_id=decision.get("decision_id"),
            )

        agreed_discount = float(neg_result["agreed_discount_rate"])
        dealmaker = DynamicDealmakerTool(
            self.credentials,
            client=self.http_client,
            anthropic_client=self.anthropic_client,
        )
        deal_result = dealmaker.run(
            order_id=event.order_id,
            order_value=order_value,
            agreed_discount_rate=agreed_discount,
            max_prepaid_discount_rate=max_discount,
            buyer_phone=event.buyer_phone,
        )
        self._log_step(
            event.order_id,
            "dynamic_dealmaker",
            "claude-haiku-4-5-20251001",
            f"discount={agreed_discount:.4f}",
            deal_result.get("status", "UNKNOWN"),
            (deal_result.get("reason_code") or "PAYMENT_LINK",),
        )

        if deal_result.get("status") != "CREATED":
            return AgentRunResult(
                order_id=event.order_id,
                final_status="FALLBACK_ACTION",
                selected_action=fallback_action,
                steps=list(self._steps),
                reason_codes=(deal_result.get("reason_code") or "PAYMENT_LINK_FAILED",),
                economics_decision_id=decision.get("decision_id"),
            )

        return AgentRunResult(
            order_id=event.order_id,
            final_status="PENDING_PAYMENT_CONFIRMATION",
            selected_action="PREPAID",
            steps=list(self._steps),
            payment_link_url=deal_result.get("payment_link_url"),
            payment_link_id=deal_result.get("payment_link_id"),
            agreed_discount_rate=agreed_discount,
            economics_decision_id=decision.get("decision_id"),
        )

    def _log_step(
        self,
        order_id: str,
        tool: str,
        model: str | None,
        input_summary: str,
        output_summary: str,
        reason_codes: Any = (),
    ) -> None:
        self._step_counter += 1
        codes = tuple(reason_codes) if reason_codes else ()
        step = AgentStep(
            step=self._step_counter,
            tool=tool,
            model=model,
            input_summary=input_summary,
            output_summary=output_summary,
            reason_codes=codes,
        )
        self._steps.append(step)
        append_step(
            order_id=order_id,
            step=step.step,
            tool=tool,
            model=model,
            input_summary=input_summary,
            output_summary=output_summary,
            reason_codes=codes,
        )

    def _review_result(
        self,
        event: CheckoutEvent,
        *,
        reason_codes: tuple[str, ...],
    ) -> AgentRunResult:
        return AgentRunResult(
            order_id=event.order_id,
            final_status="REVIEW",
            selected_action=REVIEW,
            steps=list(self._steps),
            reason_codes=reason_codes,
        )

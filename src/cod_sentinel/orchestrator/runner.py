"""LLM-orchestrated recovery loop over the deterministic economics core.

Economics is computed before the loop starts and is never gated on a model
decision, so no sequencing choice can produce an unpriced discount. Within that
boundary the orchestrator model decides which recovery tools to call and in what
order, bounded by a hard tool-call cap.
"""

import json
from dataclasses import dataclass, field
from typing import Any

import httpx

from cod_sentinel.orchestrator.audit import append_step
from cod_sentinel.orchestrator.client import (
    TOOL_DEFINITIONS,
    create_anthropic_client,
)
from cod_sentinel.orchestrator.credentials import AgentCredentials, load_credentials
from cod_sentinel.orchestrator.models import ORCHESTRATOR_MODEL
from cod_sentinel.orchestrator.prompts import ORCHESTRATOR_SYSTEM_PROMPT
from cod_sentinel.orchestrator.recovery import evaluate_recovery
from cod_sentinel.orchestrator.schemas import AgentRunResult, AgentStep, CheckoutEvent
from cod_sentinel.orchestrator.tools.address_detective import AddressDetectiveTool
from cod_sentinel.orchestrator.tools.call_e_negotiator import CallENegotiatorTool
from cod_sentinel.orchestrator.tools.dynamic_dealmaker import DynamicDealmakerTool
from cod_sentinel.orchestrator.tools.economics import EconomicsTool
from cod_sentinel.policy import REVIEW, DecisionEngine

MAX_TOOL_CALLS = 5


@dataclass
class _RunContext:
    """Mutable state accumulated across one orchestration run."""

    event: CheckoutEvent
    economics: dict[str, Any]
    probabilities: dict[str, float]
    order_value: float
    cod_rto: float
    max_discount: float
    fallback_action: str
    buyer_reply: str | None
    negotiation_started: bool
    address_tool: AddressDetectiveTool
    negotiator: CallENegotiatorTool
    dealmaker: DynamicDealmakerTool
    tools_used: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    address_verified: bool | None = None
    negotiation_status: str | None = None
    agreed_discount_rate: float | None = None
    payment_link_url: str | None = None
    payment_link_id: str | None = None


@dataclass
class OrchestratorRunner:
    """Run the checkout recovery flow as an auditable agent loop."""

    engine: DecisionEngine
    credentials: AgentCredentials | None = None
    http_client: httpx.Client | None = None
    anthropic_client: Any | None = field(default=None, repr=False)
    max_tool_calls: int = MAX_TOOL_CALLS
    _step_counter: int = field(default=0, init=False, repr=False)
    _steps: list[AgentStep] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.credentials is None:
            self.credentials = load_credentials()
        if self.anthropic_client is None and self.credentials.anthropic_api_key:
            self.anthropic_client = create_anthropic_client(
                self.credentials.anthropic_api_key
            )

    # ---------------------------------------------------------------- run

    def run(
        self,
        event: CheckoutEvent,
        *,
        buyer_reply: str | None = None,
        negotiation_started: bool = False,
    ) -> AgentRunResult:
        self._step_counter = 0
        self._steps = []

        if not self.credentials.complete or self.anthropic_client is None:
            return AgentRunResult(
                order_id=event.order_id,
                final_status="REVIEW",
                selected_action=REVIEW,
                steps=list(self._steps),
                reason_codes=("AGENT_CREDENTIALS_UNAVAILABLE",),
            )

        # Economics runs first, always, outside the model's control.
        order_row = {**event.order_features, "order_id": event.order_id}
        economics = EconomicsTool(self.engine).run(order_row)
        decision = economics["decision"]
        self._log_step(
            event.order_id,
            "economics_tool",
            None,
            f"order_id={event.order_id}",
            str(economics.get("fallback_action")),
            decision.get("reason_codes", ()),
        )
        if decision.get("selected_action") == REVIEW:
            return AgentRunResult(
                order_id=event.order_id,
                final_status="REVIEW",
                selected_action=REVIEW,
                steps=list(self._steps),
                reason_codes=tuple(decision.get("reason_codes", ())),
                economics_decision_id=decision.get("decision_id"),
            )

        context = _RunContext(
            event=event,
            economics=economics,
            probabilities=dict(decision.get("probabilities") or {}),
            order_value=float(economics["unit_economics"]["order_value"]),
            cod_rto=float(economics["cod_rto"]),
            max_discount=float(economics["max_prepaid_discount_rate"]),
            fallback_action=str(economics["fallback_action"]),
            buyer_reply=buyer_reply,
            negotiation_started=negotiation_started,
            address_tool=AddressDetectiveTool(
                self.credentials,
                client=self.http_client,
                anthropic_client=self.anthropic_client,
            ),
            negotiator=CallENegotiatorTool(
                self.credentials,
                client=self.http_client,
                anthropic_client=self.anthropic_client,
            ),
            dealmaker=DynamicDealmakerTool(
                self.credentials,
                client=self.http_client,
                anthropic_client=self.anthropic_client,
            ),
        )
        return self._loop(context, decision.get("decision_id"))

    # --------------------------------------------------------------- loop

    def _loop(
        self,
        context: _RunContext,
        decision_id: str | None,
    ) -> AgentRunResult:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": json.dumps(self._opening_context(context))}
        ]

        for _ in range(self.max_tool_calls):
            response = self.anthropic_client.messages.create(
                model=ORCHESTRATOR_MODEL,
                max_tokens=4096,
                system=ORCHESTRATOR_SYSTEM_PROMPT,
                tools=TOOL_DEFINITIONS,
                messages=messages,
            )
            # The whole content list goes back, not just the text.
            messages.append({"role": "assistant", "content": response.content})
            reasoning = self._first_text(response.content)

            if getattr(response, "stop_reason", None) != "tool_use":
                break

            results: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                name = str(block.name)
                tool_input = dict(block.input or {})
                try:
                    output = self._dispatch(name, tool_input, context)
                except Exception as error:  # fails closed, never a traceback
                    code = f"{name.upper()}_FAILED"
                    context.reason_codes.append(code)
                    output = {
                        "status": "TOOL_ERROR",
                        "reason_code": code,
                        "detail": type(error).__name__,
                    }
                self._log_step(
                    context.event.order_id,
                    name,
                    self._model_for(name),
                    json.dumps(tool_input, default=str)[:200],
                    self._summarize(output),
                    (output.get("reason_code"),) if output.get("reason_code") else (),
                    discount_offered=self._discount_in(name, tool_input, output),
                    reasoning=reasoning,
                )
                results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(output, default=str),
                    }
                )

            if not results:
                break
            # Every tool_result for this turn goes back in ONE user message.
            messages.append({"role": "user", "content": results})
        else:
            context.reason_codes.append("MAX_TOOL_CALLS_EXCEEDED")
            return self._finalize(context, decision_id, forced=True)

        return self._finalize(context, decision_id)

    def _opening_context(self, context: _RunContext) -> dict[str, Any]:
        return {
            "order_id": context.event.order_id,
            "payment_method": context.event.payment_method,
            "raw_address": context.event.raw_address,
            "buyer_phone": context.event.buyer_phone,
            "buyer_reply": context.buyer_reply,
            "negotiation_started": context.negotiation_started,
            "economics": {
                "cod_rto": context.cod_rto,
                "cod_rto_threshold": self.credentials.cod_rto_threshold,
                "max_prepaid_discount_rate": context.max_discount,
                "fallback_action": context.fallback_action,
                "unit_economics": context.economics["unit_economics"],
            },
        }

    # ----------------------------------------------------------- dispatch

    def _dispatch(
        self,
        name: str,
        tool_input: dict[str, Any],
        context: _RunContext,
    ) -> dict[str, Any]:
        context.tools_used.append(name)

        if name == "economics_tool":
            # Served from the pre-computed result; never recomputed on demand.
            return {
                "cod_rto": context.cod_rto,
                "max_prepaid_discount_rate": context.max_discount,
                "fallback_action": context.fallback_action,
                "unit_economics": context.economics["unit_economics"],
            }

        if name == "address_detective":
            raw = str(tool_input.get("raw_address") or context.event.raw_address)
            result = context.address_tool.run(raw)
            context.address_verified = bool(result["verified"])
            if not context.address_verified:
                context.reason_codes.append("ADDRESS_UNVERIFIED")
            return {
                "verified": result["verified"],
                "structured": result["structured"],
                "reason_code": result.get("reason_code"),
            }

        if name == "call_e_negotiator":
            action = str(tool_input.get("action") or "start")
            reply = tool_input.get("buyer_reply") or context.buyer_reply
            if action == "continue" and reply:
                result = context.negotiator.continue_conversation(
                    buyer_phone=context.event.buyer_phone,
                    buyer_reply=str(reply),
                    order_id=context.event.order_id,
                    order_value=context.order_value,
                    max_prepaid_discount_rate=context.max_discount,
                )
            else:
                result = context.negotiator.start(
                    buyer_phone=context.event.buyer_phone,
                    order_id=context.event.order_id,
                    order_value=context.order_value,
                    max_prepaid_discount_rate=context.max_discount,
                    cod_rto=context.cod_rto,
                )
            context.negotiation_status = result.get("status")
            agreed = result.get("agreed_discount_rate")
            if agreed is not None:
                context.agreed_discount_rate = float(agreed)
            return result

        if name == "dynamic_dealmaker":
            raw_rate = tool_input.get("agreed_discount_rate")
            rate = float(
                raw_rate if raw_rate is not None else (context.agreed_discount_rate or 0.0)
            )
            result = context.dealmaker.run(
                order_id=context.event.order_id,
                order_value=context.order_value,
                agreed_discount_rate=rate,
                # The ceiling always comes from economics, never from the model.
                max_prepaid_discount_rate=context.max_discount,
                buyer_phone=context.event.buyer_phone,
            )
            if result.get("status") == "CREATED":
                context.payment_link_url = result.get("payment_link_url")
                context.payment_link_id = result.get("payment_link_id")
                context.agreed_discount_rate = rate
            elif result.get("reason_code"):
                context.reason_codes.append(str(result["reason_code"]))
            return result

        raise ValueError(f"Unknown tool {name!r}.")

    # ----------------------------------------------------------- finalize

    def _finalize(
        self,
        context: _RunContext,
        decision_id: str | None,
        *,
        forced: bool = False,
    ) -> AgentRunResult:
        reason_codes = tuple(dict.fromkeys(context.reason_codes))

        if context.payment_link_url:
            recovery = self._price_recovery(context, "PREPAID")
            return AgentRunResult(
                order_id=context.event.order_id,
                final_status="PENDING_PAYMENT_CONFIRMATION",
                selected_action="PREPAID",
                steps=list(self._steps),
                payment_link_url=context.payment_link_url,
                payment_link_id=context.payment_link_id,
                agreed_discount_rate=context.agreed_discount_rate,
                reason_codes=reason_codes,
                economics_decision_id=decision_id,
                recovery=recovery,
            )

        if context.address_verified is False:
            return AgentRunResult(
                order_id=context.event.order_id,
                final_status="REVIEW",
                selected_action=REVIEW,
                steps=list(self._steps),
                reason_codes=reason_codes or ("ADDRESS_UNVERIFIED",),
                economics_decision_id=decision_id,
            )

        # No conversion: the order is worth what the deterministic policy already
        # made it worth, so the agent recovered nothing. Recording that zero
        # explicitly is the point — it keeps the agent honest about its value.
        unchanged = self._price_recovery(context, context.fallback_action)

        negotiated = context.negotiation_status is not None
        if forced or negotiated or reason_codes:
            return AgentRunResult(
                order_id=context.event.order_id,
                final_status="FALLBACK_ACTION",
                selected_action=context.fallback_action,
                steps=list(self._steps),
                reason_codes=reason_codes
                or ((context.negotiation_status,) if negotiated else ()),
                economics_decision_id=decision_id,
                recovery=unchanged,
            )

        return AgentRunResult(
            order_id=context.event.order_id,
            final_status="NO_NEGOTIATION_NEEDED",
            selected_action=context.fallback_action,
            steps=list(self._steps),
            economics_decision_id=decision_id,
            recovery=unchanged,
        )

    def _price_recovery(
        self,
        context: _RunContext,
        achieved_action: str,
    ) -> dict[str, Any] | None:
        """Re-price the order under what the agent actually achieved."""

        outcome = evaluate_recovery(
            order_value=context.order_value,
            merchant=self.engine.settings.merchant_economics,
            probabilities=context.probabilities,
            baseline_action=context.fallback_action,
            achieved_action=achieved_action,
            discount_rate=(
                context.agreed_discount_rate if achieved_action == "PREPAID" else None
            ),
        )
        return outcome.to_dict() if outcome else None

    # -------------------------------------------------------------- utils

    @staticmethod
    def _summarize(output: dict[str, Any]) -> str:
        """One-word outcome for the audit trail.

        ``verified`` is a bool, so it must be checked for presence rather than
        truthiness — otherwise a failed verification reads as success.
        """

        if output.get("status"):
            return str(output["status"])
        if "verified" in output:
            return "VERIFIED" if output["verified"] else "UNVERIFIED"
        return "OK"

    @staticmethod
    def _first_text(content: Any) -> str | None:
        for block in content or []:
            if getattr(block, "type", None) == "text":
                text = str(getattr(block, "text", "") or "").strip()
                if text:
                    return text[:500]
        return None

    @staticmethod
    def _model_for(tool: str) -> str | None:
        from cod_sentinel.orchestrator.models import (
            ADDRESS_MODEL,
            DEALMAKER_MODEL,
            NEGOTIATOR_MODEL,
        )

        return {
            "address_detective": ADDRESS_MODEL,
            "call_e_negotiator": NEGOTIATOR_MODEL,
            "dynamic_dealmaker": DEALMAKER_MODEL,
        }.get(tool)

    @staticmethod
    def _discount_in(
        tool: str,
        tool_input: dict[str, Any],
        output: dict[str, Any],
    ) -> float | None:
        for source in (output, tool_input):
            value = source.get("agreed_discount_rate")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _log_step(
        self,
        order_id: str,
        tool: str,
        model: str | None,
        input_summary: str,
        output_summary: str,
        reason_codes: Any = (),
        *,
        discount_offered: float | None = None,
        reasoning: str | None = None,
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
            discount_offered=discount_offered,
            reasoning=reasoning,
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
            discount_offered=discount_offered,
            reasoning=reasoning,
        )

"""Hinglish WhatsApp negotiation via Twilio."""

import json
from typing import Any

import httpx

from cod_sentinel.orchestrator.credentials import AgentCredentials
from cod_sentinel.orchestrator.models import NEGOTIATOR_MODEL
from cod_sentinel.orchestrator.prompts import CALL_E_NEGOTIATOR_SYSTEM_PROMPT


class CallENegotiatorTool:
    """Offer margin-bounded prepaid conversion over WhatsApp."""

    MAX_TURNS = 3

    def __init__(
        self,
        credentials: AgentCredentials,
        *,
        client: httpx.Client | None = None,
        anthropic_client: Any | None = None,
    ) -> None:
        self._credentials = credentials
        self._http = client or httpx.Client(timeout=30.0)
        self._anthropic = anthropic_client
        self._turn_count = 0
        self._history: list[dict[str, str]] = []

    def reset(self) -> None:
        self._turn_count = 0
        self._history = []

    def start(
        self,
        *,
        buyer_phone: str,
        order_id: str,
        order_value: float,
        max_prepaid_discount_rate: float,
        cod_rto: float,
    ) -> dict[str, Any]:
        self.reset()
        context = (
            f"Order {order_id}, value ₹{order_value:.0f}, COD RTO risk {cod_rto:.2f}. "
            f"Max discount rate: {max_prepaid_discount_rate:.4f}."
        )
        message = self._draft_message(context, buyer_reply=None)
        send_result = self._send_whatsapp(buyer_phone, message)
        self._history.append({"role": "assistant", "content": message})
        return {
            "status": "IN_PROGRESS",
            "message_sent": message,
            "twilio": send_result,
            "turn": self._turn_count,
        }

    def continue_conversation(
        self,
        *,
        buyer_phone: str,
        buyer_reply: str,
        order_id: str,
        order_value: float,
        max_prepaid_discount_rate: float,
    ) -> dict[str, Any]:
        self._turn_count += 1
        self._history.append({"role": "user", "content": buyer_reply})
        if self._turn_count > self.MAX_TURNS:
            return {"status": "UNREACHABLE", "agreed_discount_rate": None}

        context = (
            f"Order {order_id}, value ₹{order_value:.0f}. "
            f"Max discount rate: {max_prepaid_discount_rate:.4f}. "
            f"Buyer said: {buyer_reply!r}"
        )
        parsed = self._parse_reply(context, buyer_reply)
        status = parsed["status"]
        if status == "AGREED":
            agreed = float(parsed.get("agreed_discount_rate", 0.0))
            if agreed > max_prepaid_discount_rate:
                return {
                    "status": "DECLINED",
                    "agreed_discount_rate": None,
                    "reason_code": "DISCOUNT_EXCEEDS_MARGIN",
                }
            return {"status": "AGREED", "agreed_discount_rate": agreed}
        if status == "DECLINED":
            return {"status": "DECLINED", "agreed_discount_rate": None}

        message = self._draft_message(context, buyer_reply=buyer_reply)
        send_result = self._send_whatsapp(buyer_phone, message)
        self._history.append({"role": "assistant", "content": message})
        return {
            "status": "IN_PROGRESS",
            "message_sent": message,
            "twilio": send_result,
            "turn": self._turn_count,
        }

    def _draft_message(self, context: str, buyer_reply: str | None) -> str:
        if self._anthropic is None:
            return (
                "Namaste! Aapka order confirm karne ke liye — agar aap prepaid "
                "payment karte hain toh hum chhota sa discount de sakte hain. "
                "Interested?"
            )
        prompt = context if buyer_reply is None else f"{context}\nDraft next message."
        response = self._anthropic.messages.create(
            model=NEGOTIATOR_MODEL,
            max_tokens=256,
            system=CALL_E_NEGOTIATOR_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    def _parse_reply(self, context: str, buyer_reply: str) -> dict[str, Any]:
        if self._anthropic is None:
            lowered = buyer_reply.lower()
            if any(word in lowered for word in ("yes", "haan", "ok", "theek")):
                return {"status": "AGREED", "agreed_discount_rate": 0.02}
            if any(word in lowered for word in ("no", "nahi", "decline")):
                return {"status": "DECLINED"}
            return {"status": "UNCLEAR"}

        response = self._anthropic.messages.create(
            model=NEGOTIATOR_MODEL,
            max_tokens=128,
            system=(
                CALL_E_NEGOTIATOR_SYSTEM_PROMPT
                + "\nRespond with JSON: {\"status\": ..., \"agreed_discount_rate\": ...}"
            ),
            messages=[{"role": "user", "content": context}],
        )
        return json.loads(response.content[0].text.strip())

    def _send_whatsapp(self, buyer_phone: str, body: str) -> dict[str, Any]:
        url = (
            f"https://api.twilio.com/2010-04-01/Accounts/"
            f"{self._credentials.twilio_account_sid}/Messages.json"
        )
        to_number = buyer_phone if buyer_phone.startswith("whatsapp:") else f"whatsapp:{buyer_phone}"
        response = self._http.post(
            url,
            data={
                "From": self._credentials.twilio_whatsapp_from,
                "To": to_number,
                "Body": body,
            },
            auth=(
                self._credentials.twilio_account_sid,
                self._credentials.twilio_auth_token,
            ),
        )
        response.raise_for_status()
        payload = response.json()
        return {"sid": payload.get("sid"), "status": payload.get("status")}

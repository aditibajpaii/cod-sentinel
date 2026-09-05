"""Razorpay payment link creation for prepaid conversion."""

import json
from typing import Any

import httpx

from cod_sentinel.orchestrator.credentials import AgentCredentials
from cod_sentinel.orchestrator.models import DEALMAKER_MODEL
from cod_sentinel.orchestrator.prompts import DYNAMIC_DEALMAKER_SYSTEM_PROMPT


class DynamicDealmakerTool:
    """Create Razorpay payment links with margin validation."""

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

    def run(
        self,
        *,
        order_id: str,
        order_value: float,
        agreed_discount_rate: float,
        max_prepaid_discount_rate: float,
        buyer_phone: str,
        customer_name: str | None = None,
    ) -> dict[str, Any]:
        if agreed_discount_rate > max_prepaid_discount_rate:
            return {
                "status": "REJECTED",
                "reason_code": "DISCOUNT_EXCEEDS_MARGIN",
                "payment_link_url": None,
                "payment_link_id": None,
            }

        payload = self._build_payload(
            order_id=order_id,
            order_value=order_value,
            agreed_discount_rate=agreed_discount_rate,
            buyer_phone=buyer_phone,
            customer_name=customer_name,
        )
        response = self._http.post(
            "https://api.razorpay.com/v1/payment_links",
            json=payload,
            auth=(
                self._credentials.razorpay_key_id,
                self._credentials.razorpay_key_secret,
            ),
        )
        response.raise_for_status()
        body = response.json()
        return {
            "status": "CREATED",
            "payment_link_url": body.get("short_url"),
            "payment_link_id": body.get("id"),
            "amount_paise": payload["amount"],
            "reference_id": order_id,
        }

    def _build_payload(
        self,
        *,
        order_id: str,
        order_value: float,
        agreed_discount_rate: float,
        buyer_phone: str,
        customer_name: str | None,
    ) -> dict[str, Any]:
        if self._anthropic is not None:
            prompt = json.dumps(
                {
                    "order_id": order_id,
                    "order_value": order_value,
                    "agreed_discount_rate": agreed_discount_rate,
                    "buyer_phone": buyer_phone,
                    "customer_name": customer_name,
                }
            )
            response = self._anthropic.messages.create(
                model=DEALMAKER_MODEL,
                max_tokens=256,
                system=DYNAMIC_DEALMAKER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            payload = json.loads(response.content[0].text.strip())
            return {
                "amount": int(payload["amount_paise"]),
                "currency": "INR",
                "description": payload["description"],
                "reference_id": payload.get("reference_id", order_id),
                "customer": {
                    "name": payload.get("customer_name") or customer_name or "Buyer",
                    "contact": payload.get("customer_contact", buyer_phone),
                },
            }

        amount_paise = round(order_value * (1.0 - agreed_discount_rate) * 100)
        return {
            "amount": amount_paise,
            "currency": "INR",
            "description": f"Prepaid conversion for order {order_id}",
            "reference_id": order_id,
            "customer": {
                "name": customer_name or "Buyer",
                "contact": buyer_phone,
            },
        }

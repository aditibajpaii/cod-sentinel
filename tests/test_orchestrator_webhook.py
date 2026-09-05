"""Tests for the agent webhook endpoint."""

from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from cod_sentinel.orchestrator.schemas import AgentRunResult, AgentStep
from cod_sentinel.orchestrator.webhook import (
    WEBHOOK_PATH,
    create_app,
    to_response,
)


def _payload(order_id: str = "ORDER-017000") -> dict:
    """A Razorpay Magic Checkout-shaped order webhook body."""

    return {
        "event": "order.paid",
        "payload": {
            "order": {
                "entity": {
                    "id": order_id,
                    "amount": 120000,
                    "currency": "INR",
                    "method": "cod",
                }
            },
            "customer": {
                "contact": "+919876543210",
                "shipping_address": {
                    "line1": "Flat 12, Karol Bagh",
                    "city": "New Delhi",
                    "state": "Delhi",
                    "zipcode": "110005",
                },
            },
        },
    }


def _runner_returning(result: AgentRunResult) -> MagicMock:
    runner = MagicMock()
    runner.run.return_value = result
    return runner


@pytest.mark.parametrize(
    ("final_status", "selected_action", "expected_action"),
    [
        ("PENDING_PAYMENT_CONFIRMATION", "PREPAID", "CONVERT_TO_PREPAID_VIA_LINK"),
        ("NO_NEGOTIATION_NEEDED", "COD", "DISPATCH"),
        ("FALLBACK_ACTION", "OTP", "DISPATCH"),
        ("REVIEW", "REVIEW", "REVIEW"),
    ],
)
def test_terminal_status_maps_to_action(
    final_status: str,
    selected_action: str,
    expected_action: str,
) -> None:
    result = AgentRunResult(
        order_id="ORDER-017000",
        final_status=final_status,
        selected_action=selected_action,
        steps=[
            AgentStep(
                step=1,
                tool="economics_tool",
                model=None,
                input_summary="order_id=ORDER-017000",
                output_summary=selected_action,
            )
        ],
    )
    client = TestClient(create_app(runner=_runner_returning(result)))

    response = client.post(WEBHOOK_PATH, json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert body["action"] == expected_action
    assert body["final_status"] == final_status
    assert body["steps"][0]["tool"] == "economics_tool"


def test_payment_link_is_returned_for_prepaid_conversion() -> None:
    result = AgentRunResult(
        order_id="ORDER-017000",
        final_status="PENDING_PAYMENT_CONFIRMATION",
        selected_action="PREPAID",
        payment_link_url="https://rzp.io/i/SIMULATED-ORDER-017000",
        agreed_discount_rate=0.02,
    )
    client = TestClient(create_app(runner=_runner_returning(result)))

    body = client.post(WEBHOOK_PATH, json=_payload()).json()

    assert body["action"] == "CONVERT_TO_PREPAID_VIA_LINK"
    assert body["payment_link_url"].endswith("SIMULATED-ORDER-017000")
    assert body["agreed_discount_rate"] == 0.02


def test_missing_order_id_is_a_client_error() -> None:
    client = TestClient(create_app(runner=MagicMock()))

    response = client.post(WEBHOOK_PATH, json={"payload": {}})

    assert response.status_code == 400


def test_unknown_status_falls_back_to_review() -> None:
    result = AgentRunResult(
        order_id="ORDER-1",
        final_status="SOMETHING_NEW",  # type: ignore[arg-type]
        selected_action="COD",
    )

    assert to_response(result)["action"] == "REVIEW"

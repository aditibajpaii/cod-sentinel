"""Anthropic client factory and orchestrator tool schemas."""

from typing import Any

from cod_sentinel.orchestrator.models import ORCHESTRATOR_MODEL

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "economics_tool",
        "description": (
            "Compute COD-RTO risk, expected contributions, and max profitable "
            "prepaid discount. Must be called first for every checkout."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
            },
            "required": ["order_id"],
        },
    },
    {
        "name": "address_detective",
        "description": (
            "Structure and geocode an incomplete shipping address before dispatch."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "raw_address": {"type": "string"},
            },
            "required": ["raw_address"],
        },
    },
    {
        "name": "call_e_negotiator",
        "description": (
            "Start or continue a Hinglish WhatsApp prepaid conversion negotiation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["start", "continue"]},
                "buyer_reply": {"type": "string"},
            },
            "required": ["action"],
        },
    },
    {
        "name": "dynamic_dealmaker",
        "description": "Create a Razorpay payment link after buyer agrees to prepaid.",
        "input_schema": {
            "type": "object",
            "properties": {
                "agreed_discount_rate": {"type": "number"},
            },
            "required": ["agreed_discount_rate"],
        },
    },
]


def create_anthropic_client(api_key: str) -> Any:
    """Return an Anthropic client when the optional dependency is installed."""

    try:
        import anthropic
    except ImportError as error:
        raise ImportError(
            "Install agent extras: pip install -e '.[agent]'"
        ) from error
    return anthropic.Anthropic(api_key=api_key)


def orchestrator_model() -> str:
    return ORCHESTRATOR_MODEL

"""HTTP webhook exposing the orchestrator to a checkout event.

FastAPI is imported lazily so the core install and the non-agent test suite
never require it.
"""

from pathlib import Path
from typing import Any

import pandas as pd

from cod_sentinel.configuration import load_policy_settings
from cod_sentinel.features import RUNTIME_FEATURES
from cod_sentinel.generator import OBSERVABLE_PATH
from cod_sentinel.models import MODEL_BUNDLE_PATH, ModelBundle
from cod_sentinel.orchestrator.runner import OrchestratorRunner
from cod_sentinel.orchestrator.schemas import AgentRunResult, CheckoutEvent
from cod_sentinel.policy import DecisionEngine

WEBHOOK_PATH = "/api/order-webhook"

ACTION_BY_STATUS = {
    "PENDING_PAYMENT_CONFIRMATION": "CONVERT_TO_PREPAID_VIA_LINK",
    "NO_NEGOTIATION_NEEDED": "DISPATCH",
    "FALLBACK_ACTION": "DISPATCH",
    "REVIEW": "REVIEW",
}


def parse_checkout_event(
    payload: dict[str, Any],
    *,
    observable_path: Path = OBSERVABLE_PATH,
) -> CheckoutEvent:
    """Build a CheckoutEvent from a Magic Checkout-shaped webhook payload.

    Runtime features are not carried on the wire; a merchant would read them
    from its own feature store. Here that store is the observable artifact,
    joined on order id. A caller may pass ``order_features`` directly instead.
    """

    entity = (
        payload.get("payload", {})
        .get("order", {})
        .get("entity", {})
    )
    customer = payload.get("payload", {}).get("customer", {})

    order_id = str(entity.get("id") or payload.get("order_id") or "").strip()
    if not order_id:
        raise ValueError("Payload is missing an order id.")

    method = str(entity.get("method") or payload.get("payment_method") or "cod")
    payment_method = "COD" if method.lower() == "cod" else "PREPAID"

    shipping = customer.get("shipping_address") or {}
    raw_address = str(
        payload.get("raw_address")
        or shipping.get("line1")
        or ", ".join(
            str(part)
            for part in (
                shipping.get("line1"),
                shipping.get("line2"),
                shipping.get("city"),
                shipping.get("state"),
                shipping.get("zipcode"),
            )
            if part
        )
        or ""
    ).strip()
    buyer_phone = str(
        customer.get("contact") or payload.get("buyer_phone") or ""
    ).strip()

    features = payload.get("order_features")
    if not features:
        features = _lookup_features(order_id, observable_path)

    return CheckoutEvent(
        order_id=order_id,
        payment_method=payment_method,
        raw_address=raw_address,
        buyer_phone=buyer_phone,
        order_features=features,
    )


def _lookup_features(order_id: str, observable_path: Path) -> dict[str, Any]:
    if not observable_path.exists():
        raise FileNotFoundError(
            f"Observable orders not found: {observable_path}. Run `make generate`."
        )
    orders = pd.read_csv(observable_path)
    matches = orders.loc[orders["order_id"] == order_id]
    if matches.empty:
        raise ValueError(f"Order {order_id!r} not found in the feature store.")
    row = matches.iloc[0].to_dict()
    return {name: row[name] for name in RUNTIME_FEATURES if name in row}


def to_response(result: AgentRunResult) -> dict[str, Any]:
    """Map a terminal orchestrator status onto an actionable instruction."""

    return {
        "order_id": result.order_id,
        "action": ACTION_BY_STATUS.get(result.final_status, "REVIEW"),
        "final_status": result.final_status,
        "selected_action": result.selected_action,
        "payment_link_url": result.payment_link_url,
        "agreed_discount_rate": result.agreed_discount_rate,
        "reason_codes": list(result.reason_codes),
        "decision_id": result.economics_decision_id,
        "steps": [step.to_dict() for step in result.steps],
    }


def _default_runner() -> OrchestratorRunner:
    engine = DecisionEngine(
        bundle=ModelBundle.load(MODEL_BUNDLE_PATH),
        settings=load_policy_settings(),
    )
    return OrchestratorRunner(engine=engine)


def create_app(runner: OrchestratorRunner | None = None):
    """Return a FastAPI app exposing the orchestrator webhook."""

    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as error:  # pragma: no cover - depends on environment
        raise SystemExit(
            "The webhook needs the agent extras: pip install -e '.[agent]'"
        ) from error

    app = FastAPI(title="COD Sentinel agent webhook")
    resolved = runner

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(WEBHOOK_PATH)
    def order_webhook(payload: dict[str, Any]) -> dict[str, Any]:
        nonlocal resolved
        try:
            event = parse_checkout_event(payload)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        if resolved is None:
            resolved = _default_runner()
        return to_response(
            resolved.run(event, buyer_reply=payload.get("buyer_reply"))
        )

    return app


def main() -> None:  # pragma: no cover - process entrypoint
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":  # pragma: no cover
    main()

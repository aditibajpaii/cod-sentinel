"""CLI entrypoint for the agent orchestrator."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from cod_sentinel.configuration import load_policy_settings
from cod_sentinel.features import RUNTIME_FEATURES
from cod_sentinel.generator import OBSERVABLE_PATH
from cod_sentinel.models import MODEL_BUNDLE_PATH, ModelBundle
from cod_sentinel.orchestrator.runner import OrchestratorRunner
from cod_sentinel.orchestrator.schemas import CheckoutEvent
from cod_sentinel.policy import DecisionEngine


def _load_order(order_id: str, observable_path: Path) -> dict[str, object]:
    if not observable_path.exists():
        raise FileNotFoundError(
            f"Observable orders not found: {observable_path}. Run `make generate`."
        )
    orders = pd.read_csv(observable_path)
    matches = orders.loc[orders["order_id"] == order_id]
    if matches.empty:
        raise ValueError(f"Order {order_id!r} not found in observable artifact.")
    row = matches.iloc[0].to_dict()
    return {name: row[name] for name in RUNTIME_FEATURES if name in row}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run COD Sentinel agent orchestrator.")
    parser.add_argument("--order-id", required=True, help="Synthetic order id.")
    parser.add_argument("--address", required=True, help="Raw shipping address text.")
    parser.add_argument("--phone", required=True, help="Buyer phone in E.164 format.")
    parser.add_argument(
        "--payment-method",
        choices=("COD", "PREPAID"),
        default="COD",
        help="Checkout payment method.",
    )
    parser.add_argument(
        "--buyer-reply",
        default=None,
        help="Optional buyer reply to continue an in-progress negotiation.",
    )
    parser.add_argument(
        "--negotiation-started",
        action="store_true",
        help="Set when continuing an existing negotiation.",
    )
    parser.add_argument(
        "--observable-path",
        type=Path,
        default=OBSERVABLE_PATH,
        help="Path to observable_orders.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    order_features = _load_order(args.order_id, args.observable_path)
    event = CheckoutEvent(
        order_id=args.order_id,
        payment_method=args.payment_method,
        raw_address=args.address,
        buyer_phone=args.phone,
        order_features=order_features,
    )
    engine = DecisionEngine(
        bundle=ModelBundle.load(MODEL_BUNDLE_PATH),
        settings=load_policy_settings(),
    )
    runner = OrchestratorRunner(engine=engine)
    result = runner.run(
        event,
        buyer_reply=args.buyer_reply,
        negotiation_started=args.negotiation_started,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

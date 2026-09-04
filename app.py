"""Read-mostly Streamlit demo for frozen COD Sentinel artifacts."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

from cod_sentinel.configuration import load_policy_settings
from cod_sentinel.economics import cod_break_even_rto_probability
from cod_sentinel.evaluation import METRICS_PATH
from cod_sentinel.generator import OBSERVABLE_PATH
from cod_sentinel.models import MODEL_BUNDLE_PATH, ModelBundle
from cod_sentinel.policy import DecisionEngine
from cod_sentinel.schemas import OrderEconomics


def load_metrics(path: Path = METRICS_PATH) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation metrics not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_observable_orders(path: Path = OBSERVABLE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Observable orders not found: {path}")
    return pd.read_csv(path, parse_dates=["ordered_at"])


@st.cache_resource
def load_engine() -> DecisionEngine:
    return DecisionEngine(
        bundle=ModelBundle.load(MODEL_BUNDLE_PATH),
        settings=load_policy_settings(),
    )


@st.cache_data
def cached_metrics() -> dict[str, object]:
    return load_metrics()


@st.cache_data
def cached_orders() -> pd.DataFrame:
    return load_observable_orders()


def _economics_tab(engine: DecisionEngine) -> None:
    st.subheader("Merchant economics")
    st.caption(
        "Break-even values are derived from the same state-transition engine "
        "used by the policy."
    )
    merchant = engine.settings.merchant_economics
    margin = st.slider(
        "Gross margin rate",
        min_value=0.05,
        max_value=0.80,
        value=float(merchant.gross_margin_rate),
        step=0.01,
    )
    forward = st.slider(
        "Forward shipping cost (₹)",
        min_value=0.0,
        max_value=250.0,
        value=float(merchant.forward_shipping_cost),
        step=5.0,
    )
    reverse = st.slider(
        "Reverse shipping cost (₹)",
        min_value=0.0,
        max_value=300.0,
        value=float(merchant.reverse_shipping_cost),
        step=5.0,
    )
    damage = st.slider(
        "Inventory damage probability",
        min_value=0.0,
        max_value=0.80,
        value=float(merchant.inventory_damage_rate),
        step=0.01,
    )
    scenario = replace(
        merchant,
        gross_margin_rate=margin,
        forward_shipping_cost=forward,
        reverse_shipping_cost=reverse,
        inventory_damage_rate=damage,
    )
    values = range(200, 10_001, 200)
    curve = pd.DataFrame(
        {
            "Order value (₹)": values,
            "Break-even COD RTO probability": [
                cod_break_even_rto_probability(
                    OrderEconomics(order_value=float(value)),
                    scenario,
                )
                for value in values
            ],
        }
    ).set_index("Order value (₹)")
    st.line_chart(curve)
    minimum_profitable = (
        scenario.forward_shipping_cost
        + scenario.packaging_cost
        + scenario.other_fulfillment_cost
        + scenario.cod_fee_flat
    ) / max(
        scenario.gross_margin_rate - scenario.cod_fee_rate,
        1e-9,
    )
    st.info(
        "Under these assumptions, delivered COD becomes contribution-positive "
        f"above approximately ₹{minimum_profitable:,.0f}."
    )


def _live_decision_tab(engine: DecisionEngine, orders: pd.DataFrame) -> None:
    st.subheader("Live synthetic order")
    test_orders = orders.loc[orders["split"] == "test"].reset_index(drop=True)
    selected_id = st.selectbox(
        "Synthetic order",
        test_orders["order_id"].tolist(),
    )
    row = test_orders.loc[test_orders["order_id"] == selected_id].iloc[0].to_dict()
    row["order_value"] = st.number_input(
        "Order value (₹)",
        min_value=1.0,
        value=float(row["order_value"]),
        step=100.0,
    )
    row["address_quality_signal"] = st.slider(
        "Deterministic address-quality signal",
        min_value=0.0,
        max_value=1.0,
        value=float(row["address_quality_signal"]),
        step=0.01,
    )
    row["customer_prior_rto_rate"] = st.slider(
        "Prior customer RTO rate",
        min_value=0.0,
        max_value=1.0,
        value=float(row["customer_prior_rto_rate"]),
        step=0.01,
    )

    decision = engine.decide(row)
    if decision.requires_review:
        st.error(f"REVIEW — {', '.join(decision.reason_codes)}")
        return
    st.metric("Predicted COD RTO risk", f"{decision.probabilities['cod_rto']:.1%}")
    columns = st.columns(3)
    for column, action in zip(columns, ("COD", "OTP", "PREPAID")):
        column.metric(
            action,
            f"₹{decision.expected_contributions[action]:,.2f}",
        )
    st.success(f"Recommended action: {decision.selected_action}")
    st.write("Reason codes:", ", ".join(decision.reason_codes))
    st.caption(
        f"Decision {decision.decision_id} · model "
        f"{decision.versions['model']} · policy {decision.versions['policy']}"
    )


def _evaluation_tab(metrics: dict[str, object]) -> None:
    st.subheader("Frozen held-out evaluation")
    st.warning(
        "All values below are from a temporally held-out synthetic simulator, "
        "not real merchant outcomes."
    )
    risk = metrics["risk_model"]
    columns = st.columns(4)
    columns[0].metric("Precision", f"{risk['precision']:.3f}")
    columns[1].metric("Recall", f"{risk['recall']:.3f}")
    columns[2].metric("PR-AUC", f"{risk['pr_auc']:.3f}")
    columns[3].metric("Brier", f"{risk['brier']:.3f}")

    policies = metrics["policies"]
    comparison = pd.DataFrame(
        [
            {
                "Policy": value["name"],
                "Contribution/order (₹)": value[
                    "realized_contribution_per_order"
                ],
                "Total vs always-COD (₹)": value[
                    "improvement_vs_always_cod_total"
                ],
                "Intervention rate": value["intervention_rate"],
                "False-positive cost (₹)": value["false_positive_cost"],
            }
            for value in policies.values()
        ]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    cod_sentinel = policies["cod_sentinel"]
    delta = cod_sentinel["improvement_vs_best_simple_baseline_per_order"]
    if delta < 0:
        st.error(
            "Adverse result: COD Sentinel trails the best simple baseline by "
            f"₹{abs(delta):,.2f} per order in this simulator."
        )
    else:
        st.success(
            "COD Sentinel exceeds the best simple baseline by "
            f"₹{delta:,.2f} per order in this simulator."
        )

    calibration = pd.DataFrame(risk["calibration_bins"]).set_index("mean_predicted")
    st.caption("COD risk reliability")
    st.line_chart(calibration[["observed_rate"]])

    st.caption("Predicted-policy sensitivity; potential outcomes are unchanged")
    scenarios = metrics["sensitivity"]["scenarios"]
    scenario_name = st.selectbox("Sensitivity variable", sorted(scenarios))
    scenario_rows = pd.DataFrame(scenarios[scenario_name])
    st.dataframe(scenario_rows, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="COD Sentinel", layout="wide")
    st.title("COD Sentinel (RTO-X)")
    st.caption("Risk-to-action economic decisioning for COD orders")
    try:
        engine = load_engine()
        metrics = cached_metrics()
        orders = cached_orders()
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as error:
        st.error(
            "Required frozen artifacts are unavailable. Run `make generate`, "
            "`make leakage`, `make train`, and `make evaluate`."
        )
        st.code(str(error))
        st.stop()

    economics_tab, decision_tab, evaluation_tab = st.tabs(
        ["Economics", "Live Decision", "Held-Out Evaluation"]
    )
    with economics_tab:
        _economics_tab(engine)
    with decision_tab:
        _live_decision_tab(engine, orders)
    with evaluation_tab:
        _evaluation_tab(metrics)


if __name__ == "__main__":
    main()

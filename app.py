"""Read-mostly Streamlit demo for frozen COD Sentinel artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from cod_sentinel.configuration import load_policy_settings
from cod_sentinel.economics import cod_break_even_rto_probability
from cod_sentinel.evaluation import METRICS_PATH
from cod_sentinel.features import RUNTIME_FEATURES
from cod_sentinel.generator import OBSERVABLE_PATH
from cod_sentinel.models import MODEL_BUNDLE_PATH, ModelBundle
from cod_sentinel.policy import DecisionEngine
from cod_sentinel.schemas import MerchantEconomics, OrderEconomics
from cod_sentinel.versioning import DATASET_VERSION, MODEL_VERSION, POLICY_VERSION

# Product palette: ink on paper, one money-green accent. No purple theme.
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,650&family=Manrope:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Manrope', sans-serif;
}

.stApp {
  background:
    radial-gradient(1200px 500px at 10% -10%, #e8f3ee 0%, transparent 55%),
    linear-gradient(180deg, #f7f4ef 0%, #f3efe8 100%);
  color: #1a1a17;
}

[data-testid="stHeader"] {
  background: transparent;
}

.block-container {
  padding-top: 1.6rem !important;
  padding-bottom: 3rem !important;
  max-width: 1080px !important;
}

.brand-mark {
  font-family: 'Fraunces', Georgia, serif;
  font-size: clamp(2.4rem, 5vw, 3.4rem);
  font-weight: 650;
  letter-spacing: -0.03em;
  line-height: 0.95;
  color: #141411;
  margin: 0 0 0.55rem 0;
}

.brand-sub {
  font-size: 1.05rem;
  line-height: 1.45;
  color: #4a4a43;
  max-width: 34rem;
  margin: 0 0 1.4rem 0;
}

.pill {
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #0f6b4c;
  background: #dff3ea;
  border: 1px solid #b9e2d1;
  border-radius: 999px;
  padding: 0.28rem 0.7rem;
  margin-bottom: 0.85rem;
}

.section-kicker {
  font-size: 0.75rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #6b6b62;
  margin: 0.2rem 0 0.35rem 0;
}

.section-title {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.55rem;
  font-weight: 650;
  letter-spacing: -0.02em;
  margin: 0 0 0.35rem 0;
  color: #141411;
}

.section-copy {
  color: #55554c;
  font-size: 0.95rem;
  line-height: 1.45;
  margin: 0 0 1.1rem 0;
  max-width: 40rem;
}

.callout {
  border: 1px solid #d9d3c7;
  background: #fffdf8;
  border-radius: 14px;
  padding: 1rem 1.15rem;
  margin: 0.85rem 0 1.2rem 0;
}

.callout strong {
  color: #141411;
}

.callout.warn {
  border-color: #e2b8a0;
  background: #fff4ec;
}

.callout.good {
  border-color: #b9e2d1;
  background: #f1faf6;
}

.callout.bad {
  border-color: #e5b4b0;
  background: #fff1f0;
}

.action-card {
  border: 1px solid #d9d3c7;
  background: #fffdf8;
  border-radius: 14px;
  padding: 0.95rem 1rem;
  min-height: 5.5rem;
}

.action-card.winner {
  border-color: #0f6b4c;
  box-shadow: inset 0 0 0 1px #0f6b4c;
  background: #f1faf6;
}

.action-label {
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #6b6b62;
}

.action-value {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.55rem;
  font-weight: 650;
  letter-spacing: -0.02em;
  margin-top: 0.25rem;
  color: #141411;
}

.metric-strip {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0.75rem;
  margin: 0.4rem 0 1.1rem 0;
}

.metric-card {
  border: 1px solid #d9d3c7;
  background: #fffdf8;
  border-radius: 12px;
  padding: 0.8rem 0.9rem;
}

.metric-card .label {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #6b6b62;
}

.metric-card .value {
  font-family: 'Fraunces', Georgia, serif;
  font-size: 1.35rem;
  font-weight: 650;
  margin-top: 0.2rem;
  color: #141411;
}

div[data-testid="stTabs"] button {
  font-family: 'Manrope', sans-serif;
  font-weight: 600;
}

div[data-baseweb="tab-highlight"] {
  background-color: #0f6b4c !important;
}

[data-testid="stMetricValue"] {
  font-family: 'Fraunces', Georgia, serif;
}

@media (max-width: 800px) {
  .metric-strip {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
"""


def load_metrics(path: Path = METRICS_PATH) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation metrics not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for name in ("risk_model", "policies", "sensitivity", "versions"):
        if name not in payload:
            raise ValueError(f"Evaluation metrics missing section: {name}")
    expected_versions = {
        "dataset": DATASET_VERSION,
        "model": MODEL_VERSION,
        "policy": POLICY_VERSION,
    }
    for name, expected in expected_versions.items():
        if payload["versions"].get(name) != expected:
            raise ValueError(f"Evaluation metrics version mismatch for {name}.")
    return payload


def load_observable_orders(path: Path = OBSERVABLE_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Observable orders not found: {path}")
    orders = pd.read_csv(path, parse_dates=["ordered_at"])
    required = {"order_id", "ordered_at", "split", *RUNTIME_FEATURES}
    missing = required.difference(orders.columns)
    if missing:
        raise ValueError(f"Observable artifact missing columns: {sorted(missing)}")
    if orders.loc[orders["split"] == "test"].empty:
        raise ValueError("Observable artifact contains no test orders.")
    return orders


def minimum_profitable_cod_value(merchant: MerchantEconomics) -> float:
    """Order value where delivered COD contribution crosses zero."""

    denominator = merchant.gross_margin_rate - merchant.cod_fee_rate
    if denominator <= 0:
        return float("inf")
    return (
        merchant.forward_shipping_cost
        + merchant.packaging_cost
        + merchant.other_fulfillment_cost
        + merchant.cod_fee_flat
    ) / denominator


def break_even_curve(
    merchant: MerchantEconomics,
    *,
    min_value: int = 200,
    max_value: int = 10_000,
    step: int = 100,
) -> pd.DataFrame:
    """Return break-even points clipped for display on [0, 1]."""

    rows: list[dict[str, float | bool]] = []
    for value in range(min_value, max_value + 1, step):
        raw = cod_break_even_rto_probability(
            OrderEconomics(order_value=float(value)),
            merchant,
        )
        rows.append(
            {
                "order_value": float(value),
                "break_even_raw": float(raw),
                "break_even_display": float(min(max(raw, 0.0), 1.0)),
                "cod_never_pays": bool(raw < 0.0),
            }
        )
    return pd.DataFrame(rows)


def break_even_chart(curve: pd.DataFrame, floor_value: float) -> alt.Chart:
    """Build a readable break-even chart with a fixed probability domain."""

    base = alt.Chart(curve).encode(
        x=alt.X(
            "order_value:Q",
            title="Order value (₹)",
            scale=alt.Scale(domain=[200, 10_000]),
            axis=alt.Axis(format=",.0f", grid=True, tickCount=8),
        )
    )
    line = base.mark_line(
        color="#0f6b4c",
        strokeWidth=3,
    ).encode(
        y=alt.Y(
            "break_even_display:Q",
            title="Break-even COD RTO probability",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(format=".0%"),
        ),
        tooltip=[
            alt.Tooltip("order_value:Q", title="Order value", format=",.0f"),
            alt.Tooltip(
                "break_even_raw:Q",
                title="Break-even RTO",
                format=".1%",
            ),
        ],
    )
    floor = (
        alt.Chart(pd.DataFrame({"order_value": [floor_value]}))
        .mark_rule(color="#b45309", strokeDash=[5, 4], strokeWidth=1.5)
        .encode(x="order_value:Q")
    )
    return (
        (line + floor)
        .properties(height=340)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor="#55554c",
            titleColor="#2f2f2a",
            gridColor="#e7e1d6",
            domainColor="#d9d3c7",
        )
    )


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
    st.markdown('<p class="section-kicker">Merchant economics</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">When does COD stop paying?</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-copy">There is no universal RTO threshold. The '
        "curve below is derived from the same contribution engine the policy "
        "uses — fixed logistics, margin, and damage — not a hand-tuned cutoff."
        "</p>",
        unsafe_allow_html=True,
    )

    merchant = engine.settings.merchant_economics
    controls = st.columns(4)
    with controls[0]:
        margin = st.slider(
            "Gross margin rate",
            min_value=0.05,
            max_value=0.80,
            value=float(merchant.gross_margin_rate),
            step=0.01,
            help="Contribution rate before logistics and payment fees.",
        )
    with controls[1]:
        forward = st.slider(
            "Forward shipping (₹)",
            min_value=0.0,
            max_value=250.0,
            value=float(merchant.forward_shipping_cost),
            step=5.0,
        )
    with controls[2]:
        reverse = st.slider(
            "Reverse shipping (₹)",
            min_value=0.0,
            max_value=300.0,
            value=float(merchant.reverse_shipping_cost),
            step=5.0,
        )
    with controls[3]:
        damage = st.slider(
            "Inventory damage rate",
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
    floor = minimum_profitable_cod_value(scenario)
    curve = break_even_curve(scenario)
    st.altair_chart(break_even_chart(curve, floor), use_container_width=True)

    if floor == float("inf"):
        message = (
            "With these fees, delivered COD never becomes contribution-positive."
        )
        tone = "warn"
    else:
        message = (
            f"Below about <strong>₹{floor:,.0f}</strong>, delivered COD loses "
            "money at any RTO probability. Above that floor, the break-even "
            "risk rises and saturates — higher order value buys more risk "
            "tolerance when margin and damage rates stay fixed."
        )
        tone = "good"
    st.markdown(f'<div class="callout {tone}">{message}</div>', unsafe_allow_html=True)


def _live_decision_tab(engine: DecisionEngine, orders: pd.DataFrame) -> None:
    st.markdown('<p class="section-kicker">Live decision</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">Risk is not the decision</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-copy">Pick a synthetic held-out order. The models '
        "estimate action outcomes; the economic engine chooses COD, OTP, or "
        "prepaid. No simulator truth enters this screen.</p>",
        unsafe_allow_html=True,
    )

    test_orders = orders.loc[orders["split"] == "test"].reset_index(drop=True)
    left, right = st.columns([1.15, 1.0], gap="large")
    with left:
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
            "Address-quality signal",
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
    with right:
        if decision.requires_review:
            st.markdown(
                '<div class="callout warn"><strong>REVIEW</strong><br>'
                f"{', '.join(decision.reason_codes)}</div>",
                unsafe_allow_html=True,
            )
            return

        assert decision.probabilities is not None
        assert decision.expected_contributions is not None
        st.metric(
            "Predicted COD RTO risk",
            f"{decision.probabilities['cod_rto']:.1%}",
        )
        cards = st.columns(3)
        for column, action in zip(cards, ("COD", "OTP", "PREPAID")):
            winner = action == decision.selected_action
            with column:
                st.markdown(
                    f"""
                    <div class="action-card {'winner' if winner else ''}">
                      <div class="action-label">{action}</div>
                      <div class="action-value">
                        ₹{decision.expected_contributions[action]:,.0f}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        st.markdown(
            f'<div class="callout good"><strong>Recommend {decision.selected_action}</strong>'
            f"<br>{' · '.join(decision.reason_codes)}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Decision {decision.decision_id} · model "
            f"{decision.versions['model']} · policy {decision.versions['policy']}"
        )


def _evaluation_tab(metrics: dict[str, object]) -> None:
    st.markdown('<p class="section-kicker">Held-out evaluation</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-title">Honest synthetic evidence</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="section-copy">Frozen on validation, scored once on the '
        "temporal test split. Contribution is realized from held-out potential "
        "outcomes — not predicted EV.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="callout warn"><strong>Synthetic simulator only.</strong> '
        "These numbers are not merchant savings claims.</div>",
        unsafe_allow_html=True,
    )

    risk = metrics["risk_model"]
    st.markdown(
        f"""
        <div class="metric-strip">
          <div class="metric-card"><div class="label">Precision</div>
          <div class="value">{risk['precision']:.3f}</div></div>
          <div class="metric-card"><div class="label">Recall</div>
          <div class="value">{risk['recall']:.3f}</div></div>
          <div class="metric-card"><div class="label">PR-AUC</div>
          <div class="value">{risk['pr_auc']:.3f}</div></div>
          <div class="metric-card"><div class="label">Brier</div>
          <div class="value">{risk['brier']:.3f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    policies = metrics["policies"]
    comparison = pd.DataFrame(
        [
            {
                "Policy": value["name"],
                "Contribution / order (₹)": round(
                    float(value["realized_contribution_per_order"]),
                    2,
                ),
                "Vs always COD (₹)": round(
                    float(value["improvement_vs_always_cod_total"]),
                    0,
                ),
                "Intervention rate": round(float(value["intervention_rate"]), 3),
                "False-positive cost (₹)": round(
                    float(value["false_positive_cost"]),
                    0,
                ),
            }
            for value in policies.values()
        ]
    )
    st.dataframe(comparison, hide_index=True, use_container_width=True)

    cod_sentinel = policies["cod_sentinel"]
    delta = float(cod_sentinel["improvement_vs_best_simple_baseline_per_order"])
    if delta < 0:
        st.markdown(
            '<div class="callout bad"><strong>Adverse result.</strong> '
            f"COD Sentinel trails the best simple baseline by "
            f"₹{abs(delta):,.2f} per order in this simulator.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="callout good"><strong>Wins the baseline.</strong> '
            f"COD Sentinel beats the best simple baseline by "
            f"₹{delta:,.2f} per order in this simulator.</div>",
            unsafe_allow_html=True,
        )

    calibration = pd.DataFrame(risk["calibration_bins"])
    reliability = (
        alt.Chart(calibration)
        .mark_line(point=True, color="#0f6b4c", strokeWidth=2.5)
        .encode(
            x=alt.X("mean_predicted:Q", title="Predicted COD RTO", axis=alt.Axis(format=".0%")),
            y=alt.Y("observed_rate:Q", title="Observed rate", axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("mean_predicted:Q", format=".1%"),
                alt.Tooltip("observed_rate:Q", format=".1%"),
                alt.Tooltip("count:Q"),
            ],
        )
        .properties(height=260, title="COD risk reliability")
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(reliability, use_container_width=True)

    st.caption("Predicted-policy sensitivity; potential outcomes are unchanged.")
    scenarios = metrics["sensitivity"]["scenarios"]
    scenario_name = st.selectbox("Sensitivity variable", sorted(scenarios))
    scenario_rows = pd.DataFrame(scenarios[scenario_name])
    st.dataframe(scenario_rows, hide_index=True, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="COD Sentinel",
        page_icon="◈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<div class="pill">Razorpay Buildathon · Risk Manager</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="brand-mark">COD Sentinel</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="brand-sub">Risk is not the decision. An economic layer that '
        "turns calibrated COD risk into COD, OTP, or prepaid — then measures "
        "whether that choice actually paid.</p>",
        unsafe_allow_html=True,
    )

    try:
        engine = load_engine()
        metrics = cached_metrics()
        orders = cached_orders()
    except Exception as error:
        st.markdown(
            '<div class="callout warn"><strong>Artifacts unavailable.</strong> '
            "Run <code>make pipeline</code> then reopen the app.</div>",
            unsafe_allow_html=True,
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

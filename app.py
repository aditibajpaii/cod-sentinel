"""Read-mostly Streamlit demo for frozen COD Sentinel artifacts."""

from __future__ import annotations

import json
from dataclasses import replace
from html import escape
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from cod_sentinel.configuration import load_policy_settings
from cod_sentinel.economics import cod_break_even_rto_probability
from cod_sentinel.evaluation import METRICS_PATH
from cod_sentinel.features import RUNTIME_FEATURES
from cod_sentinel.generator import OBSERVABLE_PATH
from cod_sentinel.leakage import LEAKAGE_REPORT_PATH
from cod_sentinel.models import MODEL_BUNDLE_PATH, ModelBundle
from cod_sentinel.policy import Decision, DecisionEngine
from cod_sentinel.schemas import MerchantEconomics, OrderEconomics
from cod_sentinel.versioning import DATASET_VERSION, MODEL_VERSION, POLICY_VERSION

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
  --canvas: #0a0a0b;
  --panel: #12141a;
  --hairline: #1c1e22;
  --ink: #eceae4;
  --muted: #8b8a84;
  --dim: #5c5b56;
  --accent: #c9a46a;
  --allow: #6fbf86;
  --verify: #d4b56a;
  --fail: #c47a6a;
}

html, body, [class*="css"], .stApp, .stMarkdown, p, span, label, div {
  font-family: "IBM Plex Sans", system-ui, sans-serif;
}

.stApp { background: var(--canvas); color: var(--ink); }

[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stAppToolbar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stHeaderActionElements"],
.stHeadingActionElements, #MainMenu, footer, .stDeployButton,
header[data-testid="stHeader"] {
  display: none !important; visibility: hidden !important; height: 0 !important;
}

.block-container {
  padding-top: 2.4rem !important;
  padding-bottom: 3.6rem !important;
  max-width: 1040px !important;
}

.brand-kicker, .section-kicker {
  font-size: 0.68rem; font-weight: 500; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--accent); margin: 0 0 0.7rem 0;
}
.section-kicker { margin: 0.5rem 0 0.6rem 0; }

.brand-mark {
  font-size: clamp(2.5rem, 5.5vw, 3.8rem); font-weight: 600;
  letter-spacing: -0.04em; line-height: 0.94; color: var(--ink); margin: 0 0 0.75rem 0;
}
.brand-sub, .section-copy {
  font-size: 0.98rem; line-height: 1.5; color: var(--muted);
  max-width: 36rem; margin: 0 0 1.6rem 0;
}
.section-title {
  font-size: clamp(1.45rem, 2.6vw, 1.85rem); font-weight: 600;
  letter-spacing: -0.03em; margin: 0 0 0.45rem 0; color: var(--ink);
}

.hero-proof {
  border-top: 1px solid var(--hairline);
  border-bottom: 1px solid var(--hairline);
  padding: 1.35rem 0 1.15rem;
  margin: 0 0 0.55rem 0;
}
.hero-proof-top {
  display: flex; flex-wrap: wrap; align-items: baseline;
  justify-content: space-between; gap: 0.75rem 1.2rem;
}
.hero-proof-value {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: clamp(2.4rem, 5vw, 3.4rem); font-weight: 600;
  letter-spacing: -0.04em; line-height: 1; color: var(--ink);
}
.hero-proof-caption {
  margin-top: 0.45rem; font-size: 0.78rem; color: var(--dim);
}
.hero-baselines {
  display: flex; flex-wrap: wrap; gap: 0.35rem 1.4rem;
  margin: 0.85rem 0 1.6rem 0; font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.76rem; color: var(--muted);
}
.hero-baselines .sep { color: var(--dim); }

.flow-rail {
  display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 0.65rem;
  padding: 0.65rem 0 1.5rem; margin-bottom: 0.25rem;
  border-bottom: 1px solid var(--hairline);
  font-size: 0.82rem; color: var(--muted);
}
.flow-step em {
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-style: normal;
  font-weight: 600; color: var(--accent); letter-spacing: 0.06em; margin-right: 0.35rem;
}
.flow-arrow { color: var(--dim); font-family: "IBM Plex Mono", ui-monospace, monospace; }

.pill {
  display: inline-flex; align-items: center; gap: 0.35rem;
  border: 1px solid var(--hairline); border-radius: 999px;
  padding: 0.14rem 0.52rem 0.14rem 0.42rem;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.64rem; font-weight: 500; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--muted); background: transparent; white-space: nowrap;
}
.pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--dim); }
.pill-allow, .pill-pass { color: var(--allow); }
.pill-allow .dot, .pill-pass .dot { background: var(--allow); }
.pill-verify { color: var(--verify); }
.pill-verify .dot { background: var(--verify); }
.pill-fail { color: var(--fail); }
.pill-fail .dot { background: var(--fail); }
.pill-neutral { color: var(--muted); }
.pill-neutral .dot { background: var(--dim); }
.pill-row { display: flex; flex-wrap: wrap; gap: 0.4rem; margin: 0.5rem 0 1rem 0; }

.stat-value {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 1.15rem; font-weight: 600; letter-spacing: -0.03em; color: var(--ink);
}
.stat-caption { margin-top: 0.3rem; font-size: 0.72rem; color: var(--dim); }
.stat-row.four {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1rem; margin: 0.5rem 0 1.2rem 0;
}

.certificate {
  border: 1px solid var(--hairline); background: var(--panel);
  border-radius: 2px; margin: 1rem 0 0.35rem 0; overflow: hidden;
}
.cert-header {
  display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center;
  gap: 0.5rem; padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--hairline);
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.72rem; color: var(--muted);
}
.cert-header strong { color: var(--ink); font-weight: 500; }
.cert-context {
  padding: 0.45rem 1rem 0.55rem; border-bottom: 1px solid var(--hairline);
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.7rem; color: var(--dim);
}
.cert-row {
  display: grid; grid-template-columns: 1fr auto; gap: 1rem;
  padding: 0.38rem 1rem;
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.8rem;
}
.cert-row .k { color: var(--dim); }
.cert-row .v { color: var(--ink); text-align: right; }
.cert-row.selected .k, .cert-row.selected .v { color: var(--accent); }
.cert-total {
  display: grid; grid-template-columns: 1fr auto; gap: 1rem;
  padding: 0.65rem 1rem; margin-top: 0.15rem;
  border-top: 1px solid var(--hairline);
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.88rem; font-weight: 600;
}
.cert-total .k { color: var(--accent); text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.72rem; padding-top: 0.15rem; }
.cert-total .v { color: var(--accent); text-align: right; }
.cert-meta {
  padding: 0.5rem 1rem 0.75rem; font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.68rem; color: var(--dim); line-height: 1.5;
}

.outcome-grid {
  display: grid; grid-template-columns: 1fr 1fr; gap: 0.85rem;
  margin: 0.5rem 0 0.75rem 0;
}
.outcome-card {
  border: 1px solid var(--hairline); background: var(--panel);
  border-radius: 2px; padding: 1rem 1.05rem 0.95rem;
}
.outcome-card.best { border-color: #2a3a2e; }
.outcome-card.trails { border-color: #3a2a22; }
.outcome-label {
  font-size: 0.64rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--dim);
}
.outcome-value {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: clamp(1.6rem, 3vw, 2.1rem); font-weight: 600;
  letter-spacing: -0.03em; margin: 0.35rem 0 0.5rem 0; color: var(--ink);
}
.outcome-note { font-size: 0.76rem; color: var(--muted); margin: 0.35rem 0 0; }

.panel {
  border: 1px solid var(--hairline); background: var(--panel);
  border-radius: 2px; padding: 0.35rem 0.5rem 0.15rem; margin: 0.5rem 0 0.85rem 0;
}
.callout {
  border: 1px solid var(--hairline); background: transparent; border-radius: 2px;
  padding: 0.85rem 1rem; margin: 0.6rem 0 1.1rem 0; color: var(--muted); font-size: 0.9rem;
}
.callout strong { color: var(--ink); font-weight: 600; }

table.console {
  width: 100%; border-collapse: collapse;
  font-family: "IBM Plex Mono", ui-monospace, monospace; font-size: 0.72rem;
  margin: 0.25rem 0 1.1rem 0;
}
table.console th, table.console td {
  border-bottom: 1px solid var(--hairline); padding: 0.38rem 0.5rem;
  text-align: left; color: var(--ink); vertical-align: middle;
}
table.console th {
  font-family: "IBM Plex Sans", system-ui, sans-serif; font-size: 0.62rem;
  font-weight: 500; letter-spacing: 0.08em; text-transform: uppercase; color: var(--dim);
}
table.console td.num, table.console th.num { text-align: right; }
table.console tbody tr:nth-child(even) { background: #0e1014; }
table.console td.status { text-align: right; white-space: nowrap; }

.trust-strip {
  display: flex; flex-wrap: wrap; gap: 0.55rem 1.1rem; margin-top: 2.2rem;
  padding-top: 1rem; border-top: 1px solid var(--hairline);
  color: var(--dim); font-size: 0.72rem;
}

div[data-testid="stTabs"] button {
  font-family: "IBM Plex Sans", system-ui, sans-serif !important;
  font-size: 0.78rem !important; font-weight: 500 !important;
  letter-spacing: 0.06em; color: var(--muted) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] { color: var(--ink) !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--accent) !important; }
div[data-testid="stTabs"] [data-baseweb="tab-border"] { background-color: var(--hairline) !important; }

[data-testid="stSlider"] label, [data-testid="stNumberInput"] label,
[data-testid="stSelectbox"] label { color: var(--muted) !important; font-size: 0.82rem !important; }
[data-testid="stCaption"] {
  color: var(--dim) !important; font-family: "IBM Plex Mono", ui-monospace, monospace !important;
}

[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
  background: var(--panel) !important; border-color: var(--hairline) !important; color: var(--ink) !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
  background: var(--accent) !important; box-shadow: none !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[data-testid="stThumbValue"] {
  color: var(--accent) !important; font-family: "IBM Plex Mono", ui-monospace, monospace !important;
}

.stAltairChart, .vega-embed { background: var(--panel) !important; }
.stAltairChart details, .vega-actions, [data-testid="stElementToolbar"],
[data-testid="stChartToolbar"], button[title="Show data"],
button[title="Download as PNG"], button[title="Copy Vega-Lite spec"],
button[title="Fullscreen"] {
  display: none !important;
}

@media (max-width: 720px) {
  .outcome-grid, .stat-row.four { grid-template-columns: 1fr; }
}
</style>
"""

_LEDGER_INPUT_ORDER = (
    "order_value", "address_quality_signal", "customer_prior_rto_rate",
    "category", "pincode", "order_hour", "day_of_week", "month",
    "is_weekend", "festival_period", "phone_verified", "customer_prior_orders",
    "customer_prior_delivery_rate", "customer_prior_prepaid_attempts",
    "customer_prior_prepaid_success_rate", "customer_order_value_ratio",
    "pincode_prior_orders", "pincode_prior_rto_rate",
)

_CERT_CONTEXT_FIELDS = (
    "order_value", "address_quality_signal", "customer_prior_rto_rate",
    "category", "pincode",
)

_GATE_LABELS = {
    "feature_allowlist_and_provenance": "feature allowlist",
    "oracle_physical_separation": "oracle separation",
    "prior_history_recomputation": "prior history",
    "shuffled_label_sanity": "shuffled labels",
}


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
    denominator = merchant.gross_margin_rate - merchant.cod_fee_rate
    if denominator <= 0:
        return float("inf")
    return (
        merchant.forward_shipping_cost + merchant.packaging_cost
        + merchant.other_fulfillment_cost + merchant.cod_fee_flat
    ) / denominator


def break_even_curve(
    merchant: MerchantEconomics,
    *,
    min_value: int = 200,
    max_value: int = 10_000,
    step: int = 100,
) -> pd.DataFrame:
    rows: list[dict[str, float | bool]] = []
    for value in range(min_value, max_value + 1, step):
        raw = cod_break_even_rto_probability(
            OrderEconomics(order_value=float(value)), merchant,
        )
        rows.append({
            "order_value": float(value),
            "break_even_raw": float(raw),
            "break_even_display": float(min(max(raw, 0.0), 1.0)),
            "cod_never_pays": bool(raw < 0.0),
        })
    return pd.DataFrame(rows)


def _dark_chart(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0, fill="#12141a")
        .configure(background="#12141a")
        .configure_axis(
            labelColor="#8b8a84", titleColor="#c8c6bf", gridColor="#1c1e22",
            domainColor="#1c1e22", tickColor="#1c1e22",
            labelFont="IBM Plex Sans", titleFont="IBM Plex Sans",
        )
        .configure_title(color="#eceae4", font="IBM Plex Sans", fontSize=13, fontWeight=500)
        .configure_legend(labelColor="#8b8a84", titleColor="#c8c6bf")
    )


def break_even_chart(curve: pd.DataFrame, floor_value: float) -> alt.Chart:
    base = alt.Chart(curve).encode(
        x=alt.X("order_value:Q", title="Order value (₹)",
                scale=alt.Scale(domain=[200, 10_000]),
                axis=alt.Axis(format=",.0f", grid=True, tickCount=8))
    )
    line = base.mark_line(color="#c9a46a", strokeWidth=2.5).encode(
        y=alt.Y("break_even_display:Q", title="Break-even COD RTO probability",
                scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
        tooltip=[
            alt.Tooltip("order_value:Q", title="Order value", format=",.0f"),
            alt.Tooltip("break_even_raw:Q", title="Break-even RTO", format=".1%"),
        ],
    )
    floor = (
        alt.Chart(pd.DataFrame({"order_value": [floor_value]}))
        .mark_rule(color="#c9a46a", strokeDash=[5, 4], strokeWidth=1.5)
        .encode(x="order_value:Q")
    )
    return _dark_chart((line + floor).properties(height=340))


def reliability_chart(calibration: pd.DataFrame) -> alt.Chart:
    chart = (
        alt.Chart(calibration)
        .mark_line(
            point=alt.OverlayMarkDef(filled=True, color="#c9a46a", size=45),
            color="#c9a46a", strokeWidth=2,
        )
        .encode(
            x=alt.X("mean_predicted:Q", title="Predicted COD RTO",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            y=alt.Y("observed_rate:Q", title="Observed rate",
                    scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
            tooltip=[
                alt.Tooltip("mean_predicted:Q", format=".1%"),
                alt.Tooltip("observed_rate:Q", format=".1%"),
                alt.Tooltip("count:Q"),
            ],
        )
        .properties(height=260, title="COD risk reliability")
    )
    return _dark_chart(chart)


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


def _inr(value: float, *, decimals: int = 2) -> str:
    # Sign belongs outside the symbol: -₹120.50, not ₹-120.50.
    sign = "-" if value < 0 else ""
    return f"{sign}₹{abs(value):,.{decimals}f}"


def _status_pill(label: str, kind: str) -> str:
    safe_kind = kind if kind in {"allow", "verify", "fail", "pass", "neutral"} else "neutral"
    return (
        f'<span class="pill pill-{safe_kind}">'
        f'<span class="dot"></span>{escape(label)}</span>'
    )


def _stat_callout(value: str, caption: str) -> str:
    return (
        '<div class="stat">'
        f'<div class="stat-value">{escape(value)}</div>'
        f'<div class="stat-caption">{escape(caption)}</div>'
        "</div>"
    )


def _action_kind(action: str) -> str:
    if action == "COD":
        return "allow"
    if action in {"OTP", "REVIEW"}:
        return "verify"
    return "neutral"


def _format_feature(name: str, value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if name == "order_value":
        return _inr(float(value))
    if name.endswith(("_rate", "_signal", "_ratio")):
        return f"{float(value):.2f}"
    if name in {"is_weekend", "festival_period", "phone_verified"}:
        return str(int(float(value)))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:.2f}"
    return str(value)


def _cert_row(label: str, value: str, *, selected: bool = False) -> str:
    klass = "cert-row selected" if selected else "cert-row"
    return (
        f'<div class="{klass}"><span class="k">{escape(label)}</span>'
        f'<span class="v">{escape(value)}</span></div>'
    )


def _hero_strip(metrics: dict[str, object]) -> str:
    policies = metrics["policies"]
    always_cod = float(policies["always_cod"]["realized_contribution_per_order"])
    sentinel = float(policies["cod_sentinel"]["realized_contribution_per_order"])
    always_otp = float(policies["always_otp"]["realized_contribution_per_order"])
    delta = float(
        policies["cod_sentinel"]["improvement_vs_best_simple_baseline_per_order"]
    )
    status = _status_pill("trails", "fail") if delta < 0 else _status_pill("wins", "pass")
    delta_text = (
        f"Trails best baseline by {_inr(abs(delta))} / order"
        if delta < 0
        else f"Beats best baseline by {_inr(delta)} / order"
    )
    return (
        '<section class="hero-proof">'
        '<div class="hero-proof-top">'
        f'<div class="hero-proof-value">{_inr(sentinel)}</div>'
        f'<div>{status}</div>'
        "</div>"
        '<p class="hero-proof-caption">Sentinel · ₹ / order · frozen temporal test</p>'
        f'<p class="hero-proof-caption">{escape(delta_text)}</p>'
        "</section>"
        '<div class="hero-baselines">'
        f'<span>Always COD {_inr(always_cod)}</span>'
        '<span class="sep">·</span>'
        f'<span>Always OTP {_inr(always_otp)}</span>'
        "</div>"
    )


def _decision_certificate(decision: Decision, row: dict[str, object]) -> str:
    context_bits = [
        f"{name} {_format_feature(name, row[name])}"
        for name in _CERT_CONTEXT_FIELDS
        if name in row
    ]
    context = " · ".join(context_bits)
    header = (
        '<div class="cert-header">'
        f"<strong>{escape(str(decision.order_id))}</strong>"
        "<span>synthetic test</span>"
        f"{_status_pill('verified', 'pass')}"
        "</div>"
    )
    if decision.requires_review:
        body = _cert_row("Action", "REVIEW", selected=True)
        total = (
            '<div class="cert-total">'
            '<span class="k">Review</span><span class="v">REVIEW</span></div>'
        )
    else:
        assert decision.probabilities is not None
        assert decision.expected_contributions is not None
        body = _cert_row("Predicted COD RTO", f"{decision.probabilities['cod_rto']:.1%}")
        for action in ("COD", "OTP", "PREPAID"):
            body += _cert_row(
                f"EV {action}",
                _inr(decision.expected_contributions[action]),
                selected=action == decision.selected_action,
            )
        total = (
            '<div class="cert-total">'
            f'<span class="k">Recommend {escape(decision.selected_action)}</span>'
            f'<span class="v">{_inr(decision.expected_contributions[decision.selected_action])}</span>'
            "</div>"
        )
    meta = (
        f"decision {escape(decision.decision_id)} · "
        f"model {escape(decision.versions['model'])} · "
        f"policy {escape(decision.versions['policy'])}"
    )
    tags = "".join(_status_pill(code, "neutral") for code in decision.reason_codes)
    action_pill = _status_pill(decision.selected_action, _action_kind(decision.selected_action))
    return (
        f'<aside class="certificate">{header}'
        f'<div class="cert-context">{escape(context)}</div>'
        f"{body}{total}"
        f'<div class="cert-meta">{meta}</div></aside>'
        f'<div class="pill-row">{action_pill}{tags}</div>'
    )


def _all_inputs_table(row: dict[str, object]) -> str:
    rows = [
        [name, _format_feature(name, row[name])]
        for name in _LEDGER_INPUT_ORDER
        if name in row
    ]
    return _console_table([("Feature", False), ("Value", False)], rows)


def _outcome_cards(metrics: dict[str, object]) -> str:
    policies = metrics["policies"]
    sentinel = float(policies["cod_sentinel"]["realized_contribution_per_order"])
    always_otp = float(policies["always_otp"]["realized_contribution_per_order"])
    delta = float(policies["cod_sentinel"]["improvement_vs_best_simple_baseline_per_order"])
    trails_class = "trails" if delta < 0 else ""
    best_class = "best"
    note = (
        f"Trails best baseline by {_inr(abs(delta))} / order · synthetic held-out test"
        if delta < 0
        else f"Beats best baseline by {_inr(delta)} / order · synthetic held-out test"
    )
    return (
        f'<div class="outcome-grid">'
        f'<div class="outcome-card {trails_class}">'
        f'<div class="outcome-label">COD Sentinel</div>'
        f'<div class="outcome-value">{_inr(sentinel)}</div>'
        f'{_status_pill("trails" if delta < 0 else "wins", "fail" if delta < 0 else "pass")}'
        f'<p class="outcome-note">₹ / order · realized contribution</p></div>'
        f'<div class="outcome-card {best_class}">'
        f'<div class="outcome-label">Always OTP · best simple baseline</div>'
        f'<div class="outcome-value">{_inr(always_otp)}</div>'
        f'{_status_pill("best", "pass")}'
        f'<p class="outcome-note">₹ / order · realized contribution</p></div>'
        f"</div>"
        f'<p class="outcome-note">{escape(note)}</p>'
    )


def _console_table(
    headers: list[tuple[str, bool]],
    rows: list[list[str]],
    *,
    status: list[str] | None = None,
) -> str:
    head = "".join(
        f'<th class="{"num" if numeric else ""}">{escape(title)}</th>'
        for title, numeric in headers
    )
    if status is not None:
        head += '<th class="num">Status</th>'
    body_rows = []
    for index, cells in enumerate(rows):
        tds = []
        for cell, (_, numeric) in zip(cells, headers, strict=True):
            klass = ' class="num"' if numeric else ""
            tds.append(f"<td{klass}>{escape(cell)}</td>")
        if status is not None:
            tds.append(f'<td class="status">{status[index]}</td>')
        body_rows.append(f"<tr>{''.join(tds)}</tr>")
    return (
        '<table class="console"><thead><tr>'
        f"{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"
    )


def _policy_status_pills(policies: dict[str, dict[str, object]]) -> list[str]:
    best_key = str(policies["cod_sentinel"].get("best_simple_baseline", ""))
    delta = float(policies["cod_sentinel"]["improvement_vs_best_simple_baseline_per_order"])
    pills: list[str] = []
    for key in policies:
        if key == "always_cod":
            pills.append(_status_pill("baseline", "neutral"))
        elif key == best_key:
            pills.append(_status_pill("best", "pass"))
        elif key == "cod_sentinel" and delta < 0:
            pills.append(_status_pill("trails", "fail"))
        elif key == "cod_sentinel":
            pills.append(_status_pill("wins", "pass"))
        else:
            pills.append(_status_pill("—", "neutral"))
    return pills


def _leakage_pills(path: Path = LEAKAGE_REPORT_PATH) -> str:
    if not path.exists():
        return f'<div class="pill-row">{_status_pill("leakage unavailable", "verify")}</div>'
    report = json.loads(path.read_text(encoding="utf-8"))
    gates = report.get("gates", {})
    chips = [
        _status_pill(label, "pass" if str(gates.get(key, "")).lower() == "passed" else "fail")
        for key, label in _GATE_LABELS.items()
    ]
    prefix = _status_pill("leakage", "neutral")
    return f'<div class="pill-row">{prefix}{"".join(chips)}</div>'


def _trust_strip() -> str:
    badges = (
        "synthetic simulator only", "no conformal coverage guarantee",
        "REVIEW never touches oracle data", "pipeline, not an agent",
    )
    return f'<div class="trust-strip">{"".join(f"<span>{escape(b)}</span>" for b in badges)}</div>'


def _flow_rail() -> str:
    steps = ("Features", "Score", "Calibrate", "Decide")
    pieces = []
    for index, label in enumerate(steps, start=1):
        if index > 1:
            pieces.append('<span class="flow-arrow">→</span>')
        pieces.append(f'<span class="flow-step"><em>0{index}</em>{escape(label)}</span>')
    return f'<nav class="flow-rail">{"".join(pieces)}</nav>'


def _economics_tab(engine: DecisionEngine) -> None:
    st.markdown('<p class="section-kicker">01 / Economics</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">When does COD stop paying?</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">No universal RTO threshold. The curve uses the same '
        "contribution engine as the policy — not a hand-tuned cutoff.</p>",
        unsafe_allow_html=True,
    )
    merchant = engine.settings.merchant_economics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        margin = st.slider("Gross margin rate", 0.05, 0.80, float(merchant.gross_margin_rate), 0.01)
    with c2:
        forward = st.slider("Forward shipping (₹)", 0.0, 250.0, float(merchant.forward_shipping_cost), 5.0)
    with c3:
        reverse = st.slider("Reverse shipping (₹)", 0.0, 300.0, float(merchant.reverse_shipping_cost), 5.0)
    with c4:
        damage = st.slider("Inventory damage rate", 0.0, 0.80, float(merchant.inventory_damage_rate), 0.01)
    scenario = replace(
        merchant, gross_margin_rate=margin, forward_shipping_cost=forward,
        reverse_shipping_cost=reverse, inventory_damage_rate=damage,
    )
    floor = minimum_profitable_cod_value(scenario)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.altair_chart(break_even_chart(break_even_curve(scenario), floor), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if floor == float("inf"):
        msg = "Delivered COD never becomes contribution-positive at these fees. Live slider result."
    else:
        msg = (
            f"Below <strong>₹{floor:,.0f}</strong>, delivered COD loses at any RTO probability. "
            "Live slider result, not a frozen evaluation metric."
        )
    st.markdown(f'<div class="callout">{msg}</div>', unsafe_allow_html=True)


def _live_decision_tab(engine: DecisionEngine, orders: pd.DataFrame) -> None:
    st.markdown('<p class="section-kicker">02 / Decision</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">One held-out order</p>', unsafe_allow_html=True)
    test_orders = orders.loc[orders["split"] == "test"].reset_index(drop=True)
    c1, c2 = st.columns(2)
    with c1:
        selected_id = st.selectbox("Synthetic order", test_orders["order_id"].tolist())
    row = test_orders.loc[test_orders["order_id"] == selected_id].iloc[0].to_dict()
    with c2:
        row["order_value"] = st.number_input("Order value (₹)", 1.0, value=float(row["order_value"]), step=100.0)
    c3, c4 = st.columns(2)
    with c3:
        row["address_quality_signal"] = st.slider(
            "Address-quality signal", 0.0, 1.0, float(row["address_quality_signal"]), 0.01)
    with c4:
        row["customer_prior_rto_rate"] = st.slider(
            "Prior customer RTO rate", 0.0, 1.0, float(row["customer_prior_rto_rate"]), 0.01)
    decision = engine.decide(row)
    st.markdown(_decision_certificate(decision, row), unsafe_allow_html=True)
    with st.expander("All runtime inputs"):
        st.markdown(_all_inputs_table(row), unsafe_allow_html=True)


def _agent_credential_strip(credentials: object) -> str:
    """Render per-service mode: the agent runs simulated without service keys."""

    missing = list(getattr(credentials, "missing", []))
    modes = dict(getattr(credentials, "modes", {}))

    pills = [
        _status_pill(
            "Anthropic",
            "fail" if "ANTHROPIC_API_KEY" in missing else "pass",
        )
    ]
    for label, service in (
        ("Geocode", "geocode"),
        ("WhatsApp", "whatsapp"),
        ("Razorpay", "razorpay"),
    ):
        mode = modes.get(service, "SIMULATED")
        pills.append(_status_pill(f"{label} · {mode.title()}", "pass"))
    return f'<div class="pill-row">{"".join(pills)}</div>'


def _address_panel(parsed: dict) -> str:
    """Render the address detective's parse, score, and flagged issues."""

    if parsed.get("error"):
        return f'<div class="callout warn">{escape(str(parsed["error"]))}</div>'

    structured = parsed.get("structured") or {}
    score = float(parsed.get("deliverability_score") or 0.0)
    issues = parsed.get("issues") or []
    tone = "good" if score >= 0.6 else "warn" if score >= 0.3 else "bad"

    rows = "".join(
        _cert_row(label, str(structured.get(key) or "—"))
        for label, key in (
            ("Line 1", "line1"),
            ("Line 2", "line2"),
            ("City", "city"),
            ("State", "state"),
            ("Pincode", "pincode"),
        )
    )
    issue_text = (
        ", ".join(escape(str(issue)) for issue in issues) if issues else "none"
    )
    return (
        f'<div class="callout {tone}">'
        f'<div class="action-label">Deliverability confidence</div>'
        f'<div class="action-value">{score:.0%}</div>'
        f"<strong>Issues:</strong> {issue_text}"
        f"{rows}</div>"
    )


def _recovery_strip(recovery: dict | None) -> str:
    """Show the order re-priced under what the agent achieved.

    These are expected contributions at decision time, not realized profit and
    not the frozen held-out evaluation in tab 03.
    """

    if not recovery:
        return (
            '<div class="callout">No economic recovery to price — the order '
            "did not reach an economic action.</div>"
        )

    baseline = float(recovery["baseline_ev"])
    achieved = float(recovery["achieved_ev"])
    recovered = float(recovery["recovered_margin"])
    discount = recovery.get("discount_rate")
    tone = "good" if recovered > 0.005 else "warn" if recovered < -0.005 else ""
    verdict = (
        f"Agent recovered {_inr(recovered)}"
        if recovered > 0.005
        else (
            f"Agent cost {_inr(abs(recovered))}"
            if recovered < -0.005
            else "No change — the deterministic decision already stood"
        )
    )
    discount_note = (
        f" at a {discount:.1%} prepaid discount" if discount else ""
    )
    return (
        f'<div class="callout {tone}">'
        f'<div class="action-label">Expected contribution, this order</div>'
        f'<div class="action-value">{escape(recovery["baseline_action"])} '
        f"{_inr(baseline)} &rarr; {escape(recovery['achieved_action'])} "
        f"{_inr(achieved)}</div>"
        f"<strong>{escape(verdict)}</strong>{escape(discount_note)}."
        "<br><span class=\"action-label\">Expected value at decision time from "
        "calibrated probabilities — not realized profit, and not the frozen "
        "held-out evaluation in tab 03.</span>"
        "</div>"
    )


def _agent_timeline(steps: list[object]) -> str:
    if not steps:
        return '<div class="callout">No agent steps recorded yet.</div>'
    rows = []
    for step in steps:
        data = step.to_dict() if hasattr(step, "to_dict") else step
        tool = escape(str(data.get("tool", "—")))
        model = escape(str(data.get("model") or "—"))
        output = escape(str(data.get("output_summary", "")))
        rows.append(
            f'<tr><td>{data.get("step")}</td><td>{tool}</td>'
            f"<td>{model}</td><td>{output}</td></tr>"
        )
    return (
        '<table class="console-table"><thead><tr>'
        "<th>Step</th><th>Tool</th><th>Model</th><th>Output</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def _agent_tab(engine: DecisionEngine, orders: pd.DataFrame) -> None:
    st.markdown('<p class="section-kicker">04 / Agent</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Stretch orchestrator overlay</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Optional live-agent path for address validation, '
        "WhatsApp prepaid negotiation, and Razorpay payment links. "
        "Does not affect frozen tab 03 evidence.</p>",
        unsafe_allow_html=True,
    )
    try:
        from cod_sentinel.orchestrator.credentials import load_credentials
        from cod_sentinel.orchestrator.runner import OrchestratorRunner
        from cod_sentinel.orchestrator.schemas import CheckoutEvent
    except ImportError:
        st.markdown(
            '<div class="callout"><strong>Agent extras not installed.</strong> '
            "Run <code>make install-agent</code> then set credentials in <code>.env</code>.</div>",
            unsafe_allow_html=True,
        )
        return

    credentials = load_credentials()
    st.markdown(_agent_credential_strip(credentials), unsafe_allow_html=True)
    if credentials.missing:
        st.markdown(
            '<div class="callout">Missing credentials: '
            f"{escape(', '.join(credentials.missing))}. "
            "Copy <code>.env.example</code> to <code>.env</code> and add them.</div>",
            unsafe_allow_html=True,
        )
    else:
        simulated = [
            name.title()
            for name, mode in credentials.modes.items()
            if mode == "SIMULATED"
        ]
        if simulated:
            st.markdown(
                '<div class="callout">Running with simulated backends for '
                f"<strong>{escape(', '.join(simulated))}</strong>. "
                "Decisions, economics, and margin bounds are real; only the "
                "external calls are stubbed. Set <code>LIVE_MODE=1</code> (or a "
                "per-service flag) with credentials to go live.</div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        '<p class="section-title">Address Detective</p>'
        '<p class="section-copy">Paste a real Indian address. The agent parses '
        "landmarks into structured fields, cross-checks the pincode, and scores "
        "deliverability — no synthetic signal.</p>",
        unsafe_allow_html=True,
    )
    probe_address = st.text_input(
        "Messy address",
        value="C/O Ramesh, behind SBI ATM, Ward No 4, Churu",
        key="agent_probe_address",
    )
    if st.button("Parse address", key="agent_parse"):
        from cod_sentinel.orchestrator.tools.address_detective import (
            AddressDetectiveTool,
        )

        try:
            parsed = AddressDetectiveTool(credentials).run(probe_address)
        except Exception as error:  # noqa: BLE001 - surfaced to the operator
            parsed = {"error": f"{type(error).__name__}: {error}"}
        st.session_state["agent_parsed_address"] = parsed

    parsed = st.session_state.get("agent_parsed_address")
    if parsed:
        st.markdown(_address_panel(parsed), unsafe_allow_html=True)

    st.divider()

    test_orders = orders.loc[orders["split"] == "test"].reset_index(drop=True)
    c1, c2 = st.columns(2)
    with c1:
        selected_id = st.selectbox("Synthetic order", test_orders["order_id"].tolist(), key="agent_order")
    with c2:
        payment_method = st.selectbox("Payment method", ("COD", "PREPAID"), key="agent_payment")
    row = test_orders.loc[test_orders["order_id"] == selected_id].iloc[0].to_dict()
    order_features = {name: row[name] for name in RUNTIME_FEATURES if name in row}
    raw_address = st.text_area(
        "Raw shipping address",
        value="Flat 12, near Metro Station, Karol Bagh, New Delhi 110005",
        key="agent_address",
    )
    buyer_phone = st.text_input("Buyer phone (E.164)", value="+919876543210", key="agent_phone")
    buyer_reply = st.text_input(
        "Buyer reply (for continuing negotiation)",
        value="",
        key="agent_buyer_reply",
    )
    negotiation_started = st.session_state.get("agent_negotiation_started", False)

    if st.button("Run orchestrator", type="primary"):
        event = CheckoutEvent(
            order_id=str(selected_id),
            payment_method=payment_method,
            raw_address=raw_address,
            buyer_phone=buyer_phone,
            order_features=order_features,
        )
        runner = OrchestratorRunner(engine=engine, credentials=credentials)
        result = runner.run(
            event,
            buyer_reply=buyer_reply or None,
            negotiation_started=negotiation_started,
        )
        st.session_state["agent_last_result"] = result.to_dict()
        if "NEGOTIATION_IN_PROGRESS" in result.reason_codes:
            st.session_state["agent_negotiation_started"] = True
        else:
            st.session_state["agent_negotiation_started"] = False

    result_payload = st.session_state.get("agent_last_result")
    if result_payload:
        st.markdown(
            f'<div class="callout"><strong>{escape(result_payload["final_status"])}</strong> '
            f"→ {escape(result_payload['selected_action'])}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            _recovery_strip(result_payload.get("recovery")),
            unsafe_allow_html=True,
        )
        if result_payload.get("payment_link_url"):
            st.markdown(f"Payment link: `{result_payload['payment_link_url']}`")
        steps = result_payload.get("steps", [])
        st.markdown(_agent_timeline(steps), unsafe_allow_html=True)


def _evaluation_tab(metrics: dict[str, object]) -> None:
    st.markdown('<p class="section-kicker">03 / Evidence</p>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Honest synthetic evidence</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="section-copy">Frozen on validation, scored once on the temporal test split. '
        "Realized contribution from held-out potential outcomes — not predicted EV.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(_outcome_cards(metrics), unsafe_allow_html=True)
    risk = metrics["risk_model"]
    st.markdown(
        '<div class="stat-row four">'
        + _stat_callout(f"{risk['precision']:.3f}", "Precision · frozen test")
        + _stat_callout(f"{risk['recall']:.3f}", "Recall · frozen test")
        + _stat_callout(f"{risk['pr_auc']:.3f}", "PR-AUC · frozen test")
        + _stat_callout(f"{risk['brier']:.3f}", "Brier · frozen test")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<p class="section-kicker">Leakage gates</p>', unsafe_allow_html=True)
    st.markdown(_leakage_pills(), unsafe_allow_html=True)
    policies = metrics["policies"]
    comparison_rows = [
        [
            str(v["name"]),
            f"{float(v['realized_contribution_per_order']):.2f}",
            f"{float(v['improvement_vs_always_cod_total']):.0f}",
            f"{float(v['intervention_rate']):.3f}",
            f"{float(v['false_positive_cost']):.0f}",
        ]
        for v in policies.values()
    ]
    st.markdown(
        _console_table(
            [("Policy", False), ("₹ / order", True), ("vs always COD", True),
             ("Intervention", True), ("FP cost", True)],
            comparison_rows,
            status=_policy_status_pills(policies),
        ),
        unsafe_allow_html=True,
    )
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.altair_chart(reliability_chart(pd.DataFrame(risk["calibration_bins"])), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Predicted-policy sensitivity; potential outcomes are unchanged.")
    scenario_name = st.selectbox("Sensitivity variable", sorted(metrics["sensitivity"]["scenarios"]))
    scenario_rows = [
        [
            f"{float(item['value']):g}",
            f"{float(item['mean_selected_predicted_ev']):.2f}",
            " · ".join(f"{a} {int(c)}" for a, c in item.get("action_distribution", {}).items()),
        ]
        for item in metrics["sensitivity"]["scenarios"][scenario_name]
    ]
    st.markdown(
        _console_table(
            [("Value", True), ("Mean selected predicted EV", True), ("Action distribution", False)],
            scenario_rows,
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="COD Sentinel", page_icon="◈", layout="wide",
        initial_sidebar_state="collapsed",
        menu_items={"Get Help": None, "Report a bug": None, "About": None},
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown('<p class="brand-kicker">Sentinel</p>', unsafe_allow_html=True)
    st.markdown('<h1 class="brand-mark">Risk is not the decision.</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="brand-sub">An economic layer that turns calibrated COD risk into COD, '
        "OTP, or prepaid — then measures whether that choice actually paid.</p>",
        unsafe_allow_html=True,
    )
    try:
        engine = load_engine()
        metrics = cached_metrics()
        orders = cached_orders()
    except Exception as error:
        st.markdown(
            '<div class="callout"><strong>Artifacts unavailable.</strong> '
            "Run <code>make pipeline</code> then reopen the app.</div>",
            unsafe_allow_html=True,
        )
        st.code(str(error))
        st.stop()
    st.markdown(_hero_strip(metrics), unsafe_allow_html=True)
    st.markdown(_flow_rail(), unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["01 Economics", "02 Decision", "03 Evidence", "04 Agent"])
    with tab1:
        _economics_tab(engine)
    with tab2:
        _live_decision_tab(engine, orders)
    with tab3:
        _evaluation_tab(metrics)
    with tab4:
        _agent_tab(engine, orders)
    st.markdown(_trust_strip(), unsafe_allow_html=True)


if __name__ == "__main__":
    main()

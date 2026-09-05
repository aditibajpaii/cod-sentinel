"""Static evidence plots rendered from frozen evaluation metrics.

This module reads `results/metrics.json` only. It never loads a model,
regenerates data, or recomputes an evaluation, so rendering plots cannot
change a published number.
"""

import json
from pathlib import Path

from cod_sentinel.configuration import RESULTS_DIR
from cod_sentinel.evaluation import METRICS_PATH

RELIABILITY_PATH = RESULTS_DIR / "reliability_cod_rto.png"
CONTRIBUTION_PATH = RESULTS_DIR / "policy_contribution.png"
ACTION_MIX_PATH = RESULTS_DIR / "action_distribution.png"

PAPER = "#fffdf8"
INK = "#141411"
MUTED = "#6b6b62"
GREEN = "#0f6b4c"
AMBER = "#b45309"
GRID = "#e7e1d6"
EDGE = "#d9d3c7"
NEUTRAL = "#cdc6b8"

ACTION_COLORS = {"COD": MUTED, "OTP": GREEN, "PREPAID": AMBER}


def _pyplot():
    """Import matplotlib with a headless backend and project styling."""

    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - depends on environment
        raise SystemExit(
            "Rendering plots needs the optional viz extra:\n"
            '    python -m pip install -e ".[viz]"'
        ) from error

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 9.5,
            "text.color": INK,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": EDGE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 170,
        }
    )
    return plt


def _title(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=18)
    ax.text(
        0.0,
        1.035,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color=MUTED,
    )


def _rupees(value: float) -> str:
    return f"₹{value:,.2f}"


def load_metrics(path: Path = METRICS_PATH) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(
            f"Evaluation metrics not found: {path}. Run `make evaluate` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _ordered_policies(metrics: dict[str, object]) -> list[tuple[str, dict]]:
    """Return policies ascending by realized contribution per order."""

    policies: dict[str, dict] = metrics["policies"]
    return sorted(
        policies.items(),
        key=lambda item: item[1]["realized_contribution_per_order"],
    )


def render_reliability(
    metrics: dict[str, object],
    path: Path = RELIABILITY_PATH,
) -> Path:
    """Plot predicted vs observed COD RTO, with marker area by bin count."""

    plt = _pyplot()
    risk = metrics["risk_model"]
    bins = risk["calibration_bins"]
    predicted = [float(row["mean_predicted"]) for row in bins]
    observed = [float(row["observed_rate"]) for row in bins]
    counts = [int(row["count"]) for row in bins]
    largest = max(counts)

    figure, ax = plt.subplots(figsize=(6.6, 4.8))
    ax.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle=(0, (4, 4)),
        color=MUTED,
        linewidth=1.1,
        label="Perfect calibration",
        zorder=1,
    )
    # Connect only well-populated bins. A line through n=1 bins exaggerates
    # sampling noise into apparent miscalibration.
    dense = [
        (x, y) for x, y, count in zip(predicted, observed, counts) if count >= 20
    ]
    ax.plot(
        [point[0] for point in dense],
        [point[1] for point in dense],
        color=GREEN,
        linewidth=1.3,
        alpha=0.5,
        zorder=2,
    )
    ax.scatter(
        predicted,
        observed,
        s=[30.0 + 320.0 * (count / largest) for count in counts],
        color=GREEN,
        edgecolor=PAPER,
        linewidth=1.2,
        label="Test bin (area ∝ orders)",
        zorder=3,
    )

    for x, y, count in zip(predicted, observed, counts):
        if count <= 5:
            ax.annotate(
                f"n={count}",
                (x, y),
                textcoords="offset points",
                xytext=(9, -13) if y > 0.9 else (9, 5),
                fontsize=8,
                color=AMBER,
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Predicted COD RTO probability")
    ax.set_ylabel("Observed COD RTO rate")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5, labelcolor=MUTED)
    _title(
        ax,
        "COD RTO reliability, held-out test split",
        f"Brier {float(risk['brier']):.3f} · ECE {float(risk['ece']):.3f} · "
        f"{int(metrics['test_orders']):,} synthetic orders · sparse bins labelled",
    )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def render_policy_contribution(
    metrics: dict[str, object],
    path: Path = CONTRIBUTION_PATH,
) -> Path:
    """Plot realized contribution per order for every evaluated policy."""

    plt = _pyplot()
    rows = _ordered_policies(metrics)
    names = [row[1]["name"] for row in rows]
    values = [float(row[1]["realized_contribution_per_order"]) for row in rows]
    best_index = max(range(len(values)), key=lambda index: values[index])
    colors = []
    for index, (key, _) in enumerate(rows):
        if key == "cod_sentinel":
            colors.append(GREEN)
        elif index == best_index:
            colors.append(AMBER)
        else:
            colors.append(NEUTRAL)

    figure, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.barh(names, values, color=colors, height=0.62, zorder=3)
    ax.bar_label(
        bars,
        labels=[_rupees(value) for value in values],
        padding=6,
        fontsize=9,
        color=INK,
        fontweight="bold",
    )

    sentinel = metrics["policies"]["cod_sentinel"]
    gap = float(sentinel["improvement_vs_best_simple_baseline_per_order"])
    ax.set_xlim(0.0, max(values) * 1.22)
    ax.set_xlabel("Realized contribution per order")
    ax.xaxis.set_major_formatter(lambda value, _: f"₹{value:,.0f}")
    ax.grid(axis="y", visible=False)
    _title(
        ax,
        "Realized contribution per order, held-out test split",
        f"COD Sentinel trails the best simple baseline by {_rupees(abs(gap))} "
        "per order. Reported, not hidden.",
    )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def render_action_mix(
    metrics: dict[str, object],
    path: Path = ACTION_MIX_PATH,
) -> Path:
    """Plot the share of COD, OTP, and prepaid chosen by each policy."""

    plt = _pyplot()
    rows = _ordered_policies(metrics)
    names = [row[1]["name"] for row in rows]

    figure, ax = plt.subplots(figsize=(7.4, 4.4))
    left = [0.0] * len(rows)
    for action, color in ACTION_COLORS.items():
        shares = []
        for _, policy in rows:
            distribution = policy["action_distribution"]
            total = sum(distribution.values()) or 1
            shares.append(distribution.get(action, 0) / total)
        bars = ax.barh(
            names,
            shares,
            left=left,
            color=color,
            height=0.62,
            label=action,
            zorder=3,
        )
        ax.bar_label(
            bars,
            labels=[f"{share:.0%}" if share >= 0.08 else "" for share in shares],
            label_type="center",
            fontsize=8.5,
            color=PAPER,
            fontweight="bold",
        )
        left = [current + share for current, share in zip(left, shares)]

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Share of held-out orders")
    ax.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.grid(axis="y", visible=False)
    ax.legend(
        frameon=False,
        fontsize=8.5,
        labelcolor=MUTED,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
    )
    _title(
        ax,
        "Action mix by policy",
        "Only COD Sentinel uses all three actions; the winning threshold "
        "policy is nearly always OTP.",
    )

    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def render_all(metrics_path: Path = METRICS_PATH) -> list[Path]:
    metrics = load_metrics(metrics_path)
    return [
        render_reliability(metrics),
        render_policy_contribution(metrics),
        render_action_mix(metrics),
    ]


def main() -> None:
    for path in render_all():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

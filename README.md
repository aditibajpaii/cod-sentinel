# COD Sentinel (RTO-X)

**Razorpay Buildathon · Risk Manager**

**Risk tells you what might happen. COD Sentinel decides what to do about it.**
It estimates COD RTO risk and action-specific outcomes, prices the expected
contribution of COD, OTP, and prepaid against each other, and evaluates the
frozen policy against temporally held-out synthetic potential outcomes — a
supervised decision engine, not a risk score.

> Risk prediction is only the first step. The merchant decision is what to do
> with that risk.

## The result, up front

On our temporally held-out synthetic simulator of 3,000 future orders,
realized contribution per order:

| Policy | Contribution / order |
| --- | --- |
| Always COD (baseline) | ₹152.78 |
| **COD Sentinel** | **₹218.49** |
| Always OTP (best simple baseline) | ₹242.31 |

COD Sentinel beats always-COD by **+₹65.71/order** (95% customer-cluster
bootstrap interval ₹48.78–₹82.74) — and **trails the strongest simple
baseline by ₹23.81/order**, with an entirely negative paired interval.

That gap is not hidden. It shows that adding action models and an optimizer
does not automatically create a better policy: this simulator's OTP is broadly
effective, and the action-outcome models are not accurate enough to beat a
near-universal OTP policy. See [FAILURES.md](FAILURES.md) for the diagnosis
and what a v2 would need. The COD RTO model itself: precision **0.411**,
recall **0.902**, PR-AUC **0.505**, Brier score **0.227**.

All empirical language is scoped as:

> On our temporally held-out synthetic simulator...

The simulator is not evidence of real-world causal impact or merchant savings.
Oracle and potential-outcome data will remain evaluation-only and unavailable
to the runtime policy.

## Architecture

```text
Observable orders
  → prior-only temporal features
  → five supervised outcome models
  → validation-selected calibration
  → shared economic state model
  → deterministic COD / OTP / PREPAID policy

Separate oracle outcomes
  → held-out realized contribution
  → simple baselines and oracle regret
```

REVIEW is a safety fallback and never participates in economic maximization.
The outcome models are supervised simulator models, not causal estimators.

## Clean-clone workflow

Python 3.11–3.13 is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install
make test
make smoke
make generate
make leakage
make train
make evaluate
make app
```

Or run the non-interactive pipeline with `make pipeline`.
`requirements.lock` records the exact environment used for the frozen result;
run `python -m pip install -r requirements.lock` before `make install` when
exact dependency reproduction is needed.

The core project does not require credentials. Optional integration variables
are documented in `.env.example`.

## Data and evaluation protocol

- 20,000 chronological synthetic orders.
- 13,000 train / 2,000 calibration / 2,000 validation / 3,000 test.
- Five customer archetypes and repeated customers.
- Runtime and oracle data are written to separate artifacts.
- Historical features are computed from strictly earlier logged orders.
- Models and calibrators are selected on validation.
- The temporal test split is used only for frozen evaluation.
- Policy contribution is scored from the selected held-out potential outcome,
  never from predicted EV.
- Confidence intervals resample customer clusters.

## Repository structure

```text
src/cod_sentinel/   Simulator, features, leakage, models, economics, policy
tests/              Unit, property, leakage, artifact, and evaluation tests
configs/            Versioned merchant-economic assumptions
artifacts/          Reproducible generated data/models (ignored by default)
results/            Frozen evaluation evidence used by the demo
app.py              Read-mostly Streamlit application
```

The dependency-ordered implementation plan is in [BUILD_PLAN.md](BUILD_PLAN.md).
Architecture boundaries are summarized in
[ARCHITECTURE.md](ARCHITECTURE.md). Assumptions and rejected approaches are in
[DECISIONS.md](DECISIONS.md), [LIMITATIONS.md](LIMITATIONS.md), and
[FAILURES.md](FAILURES.md). The exact five-minute walkthrough is in
[PITCH.md](PITCH.md).

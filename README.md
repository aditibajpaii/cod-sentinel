# COD Sentinel (RTO-X)

**Razorpay AI Buildathon · Track 02 · AI Risk Manager**

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

![Realized contribution per order for all five evaluated policies on the
held-out test split. COD Sentinel reaches ₹218.49 and always OTP reaches
₹242.31.](results/policy_contribution.png)

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

### Why a pipeline, not an agent

Every decision must be reproducible from frozen artifacts: the same order and
the same model bundle must produce the same action and the same decision ID.
That property is what makes the held-out evaluation and the per-decision
record meaningful. An agent loop would introduce nondeterminism between the
score and the action for no gain in decision quality — choosing among COD,
OTP, and prepaid is a three-way expected-value comparison with a closed-form
answer, not a planning problem.

No language model sits on the runtime path, and that is a scope decision
rather than an omission. The place a language model would earn its keep is
address deliverability: real delivery addresses are unstructured text where
deterministic parsing is brittle, and a narrow, schema-validated call that
fails open to the deterministic features would reach signal this feature set
cannot. This simulator emits a scalar `address_quality_signal` rather than
free-text addresses, so an address model here would be decoration measured
against nothing. It belongs in a version evaluated on real addresses, with an
ablation reporting what it adds.

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

The core project does not require credentials. Optional agent integrations
(Anthropic, Google Maps, Twilio, Razorpay) are documented in `.env.example`.

### Optional agent overlay (Tab 04)

After `make install-agent`, configure `.env` and open tab **04 Agent** in the
Streamlit app (or run `make agent-demo ORDER_ID=... ADDRESS="..." PHONE=...+91...`).
This stretch layer sequences address validation, Hinglish WhatsApp prepaid
negotiation, and Razorpay payment links. It does not change frozen tab 03
evidence or the deterministic tab 02 decision path.

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

### Committed evidence

`results/metrics.json` holds every frozen number. The plots below are rendered
from that file alone — no model is loaded and no evaluation is recomputed, so
regenerating them cannot move a published result. Rebuild them with
`make plots` after `python -m pip install -e ".[viz]"`.

![Reliability of the COD RTO model on the held-out test split. Marker area is
proportional to the number of orders in each bin; sparse bins are
labelled.](results/reliability_cod_rto.png)

![Share of COD, OTP, and prepaid actions chosen by each policy. COD Sentinel
is the only policy that uses all three.](results/action_distribution.png)

Marker area on the reliability plot is proportional to bin count: the bins
carrying nearly all 3,000 test orders track the diagonal closely, while the
visibly noisy points hold one, two, and four orders.

## Repository structure

```text
src/cod_sentinel/   Simulator, features, leakage, models, economics, policy
tests/              Unit, property, leakage, artifact, and evaluation tests
configs/            Versioned merchant-economic assumptions
artifacts/          Reproducible generated data/models (ignored by default)
results/            Frozen evaluation evidence used by the demo
app.py              Read-mostly Streamlit application
```

Decisions, economics, and evaluation live in `src/cod_sentinel/`; `ledger.py`
records decisions and `plots.py` renders committed evidence. Neither is part of
the core pipeline.

### Decision ledger and replay

Each decision carries its own audit record — the decision ID, the
counterfactual expected contribution of every action, reason codes, and the
model, calibration, economics, and policy versions that produced it.
`make ledger-sample` writes 25 held-out decisions to
`artifacts/decision_ledger_sample.jsonl` as hash-chained, append-only records,
so any edit to a past record invalidates every record after it. Because a
decision ID is a hash over the inputs, economics, and versions, replaying a
record either reproduces its ID exactly or reports whether the difference is
version drift or a corrupted log.

Records store only the runtime feature allowlist, so oracle columns cannot
reach the log. The ledger is a sink: no runtime decision reads it, `decide`
never writes to it, and it is excluded from `make pipeline`.

The dependency-ordered implementation plan is in [BUILD_PLAN.md](BUILD_PLAN.md).
Architecture boundaries are summarized in
[ARCHITECTURE.md](ARCHITECTURE.md). Assumptions and rejected approaches are in
[DECISIONS.md](DECISIONS.md), [LIMITATIONS.md](LIMITATIONS.md), and
[FAILURES.md](FAILURES.md). The exact five-minute walkthrough is in
[PITCH.md](PITCH.md).

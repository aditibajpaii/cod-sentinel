# COD Sentinel (RTO-X)

**Razorpay AI Buildathon · Track 02 · AI Risk Manager**

## A hybrid agentic risk architecture

**Autonomous negotiation agents on top. Deterministic economic guardrails
underneath.**

A risk score tells a merchant an order looks bad. It does not recover the order.
COD Sentinel puts an **autonomous recovery agent** in front of every risky COD
checkout — it parses a chaotic Indian address, negotiates a prepaid conversion
with the buyer in Hinglish over WhatsApp, and issues a Razorpay payment link —
while a deterministic economics engine underneath decides, per order, what the
agent is allowed to give away.

> Risk prediction is only the first step. The merchant decision is what to do
> with that risk — and then actually doing it.

```text
        ┌─────────────────────────────────────────────┐
        │  AGENT LOOP  (Claude, tool-calling)         │
        │  address detective → negotiator → dealmaker │
        └───────────────────┬─────────────────────────┘
                            │  every action priced against
        ┌───────────────────▼─────────────────────────┐
        │  ECONOMIC GUARDRAIL  (deterministic)        │
        │  EV(COD) vs EV(OTP) vs EV(PREPAID),         │
        │  max profitable discount, frozen + testable │
        └─────────────────────────────────────────────┘
```

The guardrail is what makes the agent safe to run: it computes each order's
**maximum profitable discount before the agent starts**, and the payment-link
tool rejects anything above it. The agent chooses *what to attempt*; economics
decides *what is affordable*. Neither can be removed without breaking the other.

When the agent converts an order, the system re-prices it and reports the margin
actually recovered — expected contribution before and after, per order.

The whole loop runs on a clean clone with **only an `ANTHROPIC_API_KEY`** —
geocoding, WhatsApp, and payment links ship with simulated backends and switch
to live APIs by environment flag, with no code change.

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

### Why recovery is agentic

Recovering a risky order is a planning problem. Whether to validate a malformed
address first, whether to negotiate at all, whether a buyer's Hinglish reply
counts as agreement, and when to give up — these are open-ended, and each step's
result changes what should happen next. No decision tree survives contact with
"Opposite Hanuman Temple, near electric pole, 2nd floor, Sharma Colony."

So the orchestrator is a real tool-calling loop: Claude receives the order's
economics and decides which specialists to invoke and in what order, bounded by
a hard cap of five tool calls and a forced fallback when it is exhausted. Every
call is recorded as a hash-chained audit record carrying the tool, the model,
the discount offered, and the model's stated reasoning.

### Why the guardrail underneath is deterministic

Pricing is the one thing that must never be improvised. Choosing among COD, OTP,
and prepaid is a closed-form expected-value comparison, and it has to be
reproducible: the same order and the same model bundle produce the same action
and the same decision ID, which is what makes the held-out evaluation and the
audit trail mean anything.

So no language model prices an order, and none can widen a discount.
`max_profitable_prepaid_discount` solves for each order's break-even discount
before the agent starts, and the payment-link tool rejects anything above it —
a hallucinated rate, an over-eager negotiation, or a manipulated model response
all hit the same wall. The charged amount is recomputed in Python and never read
back from the model.

That boundary is the architecture, not a limitation of it: **an agent that can
act freely because it cannot act unprofitably.**

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

### Running the recovery agent

```bash
make install-agent
export ANTHROPIC_API_KEY=...        # the only credential required
make agent-demo ORDER_ID=ORDER-017000 ADDRESS="near big tree, gali no 4" PHONE=+919876543210
```

Geocoding, WhatsApp, and Razorpay run against simulated backends by default, so
the full loop works with no other account. Set `LIVE_MODE=1` — or
`LIVE_GEOCODE`, `LIVE_WHATSAPP`, `LIVE_RAZORPAY` individually — plus that
service's credentials to switch to real APIs without touching code.

The same flow is available as tab **04 Agent** in the Streamlit app, and as an
HTTP endpoint for checkout integration:

```bash
make webhook     # POST a Magic Checkout-shaped payload to /api/order-webhook
```

The webhook returns an actionable instruction — `DISPATCH`,
`CONVERT_TO_PREPAID_VIA_LINK` (with the payment link), or `REVIEW` — plus the
decision ID and the full step trail. None of this changes the frozen tab 03
evidence or the deterministic tab 02 decision path.

**Consent.** Call-E sends commercial messages to buyers. A production deployment
would require explicit prior opt-in consent under India's DPDP Act, 2023 and
TRAI's commercial-communication regulations, with a working opt-out on every
message and sender registration on the merchant's behalf. This prototype only
messages synthetic buyers, and ships with WhatsApp simulated by default.

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

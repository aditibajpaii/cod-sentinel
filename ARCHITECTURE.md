# Architecture

COD Sentinel has separate runtime and evaluation trust domains.

## Runtime path

```text
Observable order
  → prior-only feature engineering
  → action-outcome models
  → probability calibration
  → shared economic state model
  → deterministic COD / OTP / PREPAID policy
  → action or REVIEW fallback
```

Runtime code may use only information available when an order decision is made.

## Economic state transitions

All actions use the same three terminal branches:

```text
action attempted
  ├─ not converted
  └─ converted / shipped
       ├─ delivered
       └─ failed delivery (RTO or return)
```

- COD is already accepted, so its conversion probability is always one.
- OTP conversion means verification completed; a completed OTP order proceeds
  as COD and can still deliver or RTO.
- Prepaid conversion means payment completed; a paid order can still deliver or
  fail and be refunded.
- REVIEW is not part of this graph or the economic argmax. It is a policy
  fallback for invalid, missing, or unsafe runtime inputs.

`economics.py` computes every realized branch first, then computes expected
contribution as their probability-weighted sum. Break-even calculations are
derived from those same realized branches.

### Cost timing

- OTP verification cost is incurred when OTP is attempted.
- Packaging, forward shipping, and other fulfillment costs are incurred only
  after conversion, when the order ships.
- COD collection fees are incurred only after successful delivery.
- Prepaid processing fees are incurred when payment converts and are modeled as
  non-refundable.
- A failed shipped order incurs reverse shipping and expected inventory loss.
- A failed prepaid order additionally incurs its configured refund cost.

## Evaluation path

```text
Versioned simulator
  → latent state and potential outcomes
  → temporally held-out oracle data
  → realized contribution and oracle-regret evaluation
```

Oracle data may score a frozen policy after it chooses an action. It must never
produce a runtime feature, model input, expected value, or action.

## Package boundaries

- `configuration.py` owns deterministic build settings and split sizes.
- `versioning.py` owns explicit artifact and pipeline version identifiers.
- `generator.py` creates observable orders and a physically separate oracle
  outcome artifact.
- `features.py` owns prior-only history construction and the runtime feature
  allowlist.
- `leakage.py` recomputes histories, verifies artifact separation, and runs the
  shuffled-label sanity gate.
- `models.py` trains five supervised simulator outcome models using only train,
  calibration, and validation rows.
- `calibration.py` fits calibration candidates on calibration and chooses on
  validation.
- `economics.py` is the sole source of realized and expected contribution.
- `policy.py` consumes only calibrated runtime probabilities and merchant
  economics. It cannot import simulator potential outcomes.
- `evaluation.py` is the only layer that maps selected actions to held-out
  potential outcomes.
- `ledger.py` is an append-only decision sink with hash-chained records and
  replay verification. Nothing in the runtime path reads it, `decide` never
  writes to it, and it records only the runtime feature allowlist.
- `orchestrator/` is the recovery agent layer. `runner.py` runs a Claude
  tool-calling loop over `client.TOOL_DEFINITIONS`, capped at five calls with a
  forced fallback to the deterministic decision. `tools/` holds the specialists
  (address detective, Call-E negotiator, dynamic dealmaker) and `tools/economics.py`
  wraps the frozen policy. `recovery.py` re-prices an order under the action the
  agent achieved. `credentials.py` routes each external service to a simulated or
  live backend. `webhook.py` exposes `POST /api/order-webhook`.
- The agent may sequence actions but may not price them. Economics is computed
  before the loop starts, the prepaid discount ceiling comes from
  `max_profitable_prepaid_discount`, and the charged amount is recomputed in
  Python rather than read from a model response.
- `app.py` is a read-mostly adapter over frozen package artifacts; it owns no
  training or business logic.

## Artifact flow

```text
make generate
  ├─ artifacts/observable_orders.csv
  └─ artifacts/oracle_outcomes.csv

make leakage
  └─ artifacts/leakage_report.json

make train
  ├─ artifacts/model_bundle.joblib
  ├─ artifacts/model_bundle.joblib.sha256
  └─ artifacts/model_metadata.json

make freeze
  └─ artifacts/frozen_policy.json

make evaluate
  └─ results/metrics.json
```

Generated raw data and model binaries are ignored because they are reproduced
from a frozen seed and commands. Small frozen metadata and final metrics are
committed as submission evidence.

No database, API service, authentication layer, container, or external
integration is part of the core architecture.

# Architecture

The scaffold establishes two trust domains that later milestones must preserve.

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
- Future economics, simulation, models, policy, and evaluation modules live in
  the same installable package while retaining the runtime/evaluation boundary.
- `app.py` will be a read-mostly adapter over frozen package artifacts; it will
  not own business logic or training.

No database, API service, authentication layer, container, or external
integration is part of the core architecture.

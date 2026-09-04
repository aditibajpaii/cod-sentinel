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

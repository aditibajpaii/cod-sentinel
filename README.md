# COD Sentinel (RTO-X)

COD Sentinel is a prototype risk-to-action economic decision engine for COD
orders. Its goal is to estimate COD RTO risk, compare the expected contribution
of COD, OTP, and prepaid actions, and evaluate the frozen policy against
temporally held-out synthetic potential outcomes.

This repository is currently at the **scaffold milestone**. It intentionally
contains no economic policy, simulator, ML model, or claimed result yet.

## Evidence standard

Future empirical results will be labeled:

> On our temporally held-out synthetic simulator...

The simulator is not evidence of real-world causal impact or merchant savings.
Oracle and potential-outcome data will remain evaluation-only and unavailable
to the runtime policy.

## Setup

Python 3.11–3.13 is supported.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
make install
make test
make smoke
```

The core project does not require credentials. Optional integration variables
are documented in `.env.example`.

## Current structure

```text
src/cod_sentinel/   Installable package and configuration foundation
tests/              Scaffold smoke tests
configs/            Future versioned experiment configuration
artifacts/          Generated model/data artifacts (ignored by default)
results/            Frozen evaluation outputs (ignored by default)
```

The dependency-ordered implementation plan is in [BUILD_PLAN.md](BUILD_PLAN.md).
Architecture boundaries are summarized in
[ARCHITECTURE.md](ARCHITECTURE.md).

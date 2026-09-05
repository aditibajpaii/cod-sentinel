# Architecture Decisions

## ADR-001: Use branch-based contribution accounting

**Status:** Accepted

COD, OTP, and PREPAID share one state-transition implementation. The engine
prices three terminal branches—no conversion, delivered, and failed
delivery—and derives expected contribution by weighting those realized
branches.

This was chosen over separate action formulas because independent formulas had
already produced inconsistent discount, payment-fee, and return assumptions.

## ADR-002: Keep REVIEW outside economic optimization

**Status:** Accepted

REVIEW is a safety fallback owned by the future policy layer. It has no
`EV_REVIEW` and never receives simulator truth. Any later reviewer experiment
must be clearly labeled as a simulated scenario or oracle upper bound.

## ADR-003: Define prototype cost timing explicitly

**Status:** Accepted for the synthetic prototype

- OTP verification is paid on every OTP attempt.
- Fulfillment is paid only after customer completion/conversion.
- COD fees are collected only on successful delivery.
- Prepaid processing fees are paid on conversion and are not refunded.
- Failed deliveries incur reverse shipping and expected inventory loss.
- Failed prepaid deliveries also incur a configurable refund cost.

These are configurable simulator assumptions, not claims about a specific
merchant or Razorpay contract. Sensitivity analysis must test conclusions
against alternative values.

## ADR-004: Represent damaged inventory recovery without double counting

**Status:** Accepted

Expected inventory loss on failed delivery is:

```text
product cost
× inventory damage probability
× (1 - damaged inventory recovery rate)
```

Returned, undamaged inventory is treated as recoverable and product cost is not
charged again. This keeps merchandise loss separate from forward/reverse
logistics.

## ADR-005: Derive COD break-even from realized branches

**Status:** Accepted

If `S` is delivered COD contribution and `F` is failed-delivery contribution,
then:

```text
EV(p_rto) = (1 - p_rto)S + p_rto F
p_break_even = S / (S - F)
```

The implementation computes `S` and `F` through the shared engine. It does not
maintain a second handwritten formula. A non-positive result means delivered
COD is already non-profitable under the configured assumptions.

## ADR-006: Train only outcome probabilities required by economics

**Status:** Accepted

The bundle contains COD RTO, OTP completion, OTP RTO after completion, prepaid
conversion, and prepaid failed-delivery models. Logistic regression and
histogram gradient boosting are compared per target; calibration and estimator
choices are selected by validation Brier score.

These are supervised models trained on simulator potential-outcome labels.
They are not uplift or causal models. Real deployment would require logged or
randomized intervention data.

## ADR-007: Keep test outcomes outside training and selection

**Status:** Accepted

Training filters to train, calibration, and validation IDs before joining
labels. Calibration candidates are fit on calibration and selected on
validation. Classification and simple-policy thresholds are also selected on
validation. Only the frozen evaluator maps test actions to test potential
outcomes.

## ADR-008: Evaluate realized branches, not predicted EV

**Status:** Accepted

Predicted EV chooses an action. Evaluation then looks up only the held-out
potential outcome for that selected action and prices it through
`realized_contribution`. The simulator oracle separately chooses the best
realized branch and is labeled an unavailable upper bound.

## ADR-009: Include baselines that can disprove the system

**Status:** Accepted

Evaluation includes always COD, always OTP, always prepaid, and a
validation-selected RTO threshold. The final v1 result is adverse: the simple
OTP threshold beats COD Sentinel. This is retained because omitting a strong
baseline would make the evidence misleading.

## ADR-010: Defer uncertainty and external integrations

**Status:** Accepted (core); partially superseded for stretch agent work

No bootstrap probability interval, database, or webhook is part of the core.
Customer-cluster bootstrap is used only for an evaluation interval on mean
contribution improvement.

## ADR-011: Agent orchestrator as optional sink overlay

**Status:** Accepted

The stretch agent in `src/cod_sentinel/orchestrator/` wraps — but does not
replace — the deterministic `DecisionEngine`. Economics and margin bounds still
flow through `economics.py`. The agent path:

- calls live Anthropic, geocoding, Twilio, and Razorpay APIs
- writes to a separate `agent_audit.jsonl` sink
- is exposed only in Streamlit tab 04 and `make agent-demo`
- does not alter frozen held-out evaluation or `make pipeline`

Held-out metrics remain pipeline-based; the agent is a checkout-flow demo layer.

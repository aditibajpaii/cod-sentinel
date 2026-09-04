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

# Failures and Corrections

## Incorrect order-value hero narrative

**Failure:** The original concept claimed that a higher-value order should
automatically tolerate less RTO risk.

**Correction:** Deriving break-even from shared realized branches shows that,
with fixed logistics costs and constant margin/damage rates, break-even risk is
increasing and concave in order value. The demo now emphasizes fixed costs,
margin, damage, and intervention friction rather than the rejected claim.

## Inconsistent prepaid discount formula

**Failure:** An early formula used `V * m * (1 - discount)`, which understates
the cost of a discount.

**Correction:** The state model calculates discounted revenue first and keeps
product cost unchanged. COD, OTP, and prepaid now share branch accounting.

## Oracle-valued REVIEW

**Failure:** An early plan valued human review using the simulator's true
probability or best realized action.

**Correction:** REVIEW was removed from economic maximization. It is now only a
safety fallback and never receives oracle data.

## Weak leakage heuristic

**Failure:** An early plan proposed treating high single-feature AUC or latent
correlation as leakage.

**Correction:** Leakage gates now use explicit allow/deny lists, physical
artifact separation, history recomputation, future-row invariance tests, and a
shuffled-label check. The frozen shuffled-label validation ROC-AUC is 0.507.

## Broken model artifact entrypoint

**Failure:** Training with `python -m cod_sentinel.models` pickled
`OutcomeModel` and `ModelBundle` under `__main__`. A fresh process could not
load the artifact.

**Correction:** `make train` imports the canonical module before calling its
entrypoint. A subprocess reload regression test was added. A second regression
test caused `ModelBundle.load` to accept both strings and `Path` objects.

## COD Sentinel lost to a simple baseline

**Failure:** The first frozen held-out run showed COD Sentinel at ₹207.74/order
and the validation-selected OTP threshold at ₹242.89/order.

**Diagnosis:** In `synthetic-dgp-v1`, OTP is inexpensive, completes frequently,
and broadly reduces failed delivery. Individualized action-model errors cost
more than the targeting benefit. The economic engine improved on always COD
but did not justify its complexity against the strongest simple baseline.

**Response:** No post-test tuning was performed. Always OTP and always prepaid
were added as stronger audit baselines, the adverse result is displayed in the
app and README, and this limitation is part of the pitch. A future experiment
must be designed on a new development simulation and evaluated on a new,
untouched temporal holdout.

## Red-team invalidated v1 outcome timing

**Failure:** `synthetic-dgp-v1` updated customer and pincode outcome histories
immediately when an order was placed. A delivery or RTO would not actually be
known at that time. Its OTP/prepaid failure models were also trained on all
orders while the policy interpreted them as conditional probabilities.

**Correction:** `synthetic-dgp-v2` assigns explicit outcome-observation times
and exposes resolved outcomes only to later orders. OTP RTO is trained only
among OTP completions, and prepaid failure only among prepaid conversions.
Artifact hashes, split manifests, and separate freeze/evaluate commands were
added before a newly versioned v2 held-out run.

**Result:** The corrected policy improved to ₹218.49/order but still lost to
always OTP at ₹242.31/order. The adverse conclusion remained, so no attempt was
made to tune it away.

## Environment-only virtualenv failure

**Failure:** The first clean-install attempt could not create `.venv/include`
inside the restricted shell sandbox.

**Correction:** The same clean install was rerun outside that restriction. The
editable install, import smoke check, and tests passed; no project code change
was needed.

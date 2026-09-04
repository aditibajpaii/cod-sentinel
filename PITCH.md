# Five-Minute Pitch and Demo

Every empirical statement below refers to the temporally held-out synthetic
simulator. Do not remove that qualifier.

**The hook, memorized first:** *"Risk is not the decision. We built the
economic layer that sits after the risk score — and when the honest answer
was that a simpler policy wins, we said so on stage instead of tuning it
away."*

**Numbers to have cold:**

| | ₹/order |
| --- | --- |
| Always COD (baseline) | 152.78 |
| COD Sentinel | 218.49 |
| Always OTP (best simple baseline) | 242.31 |

COD Sentinel: **+65.71 vs always-COD**, **−23.81 vs always-OTP**. COD RTO
model: precision 0.411, recall 0.902, PR-AUC 0.505, Brier 0.227.

## 0:00–0:30 — Problem

“An RTO classifier tells a merchant which COD orders look risky. It does not
tell the merchant whether COD, OTP verification, or prepaid conversion is the
economically best response. A false positive can prevent an RTO—or destroy a
good sale.”

Show the title and thesis.

## 0:30–1:10 — Architecture

Show `ARCHITECTURE.md`.

“COD Sentinel separates observable runtime data from simulator oracle data.
Prior-only features feed five calibrated outcome models. A shared economic
state model prices COD, OTP, and prepaid. REVIEW is only a safety fallback.
Potential outcomes are visible only after the frozen policy chooses an action.”

Mention the 13k/2k/2k/3k chronological split and delayed outcome-observation
timestamps.

## 1:10–2:05 — Economics tab

Open the Economics tab.

Change margin, forward shipping, reverse shipping, and damage probability.

“There is no universal RTO threshold. The break-even point follows from the
same branch accounting used by policy decisions. It changes with fixed
logistics, margin, and inventory recovery. We corrected an early inverted
order-value narrative by deriving and testing this curve.”

## 2:05–2:55 — Live Decision tab

Select a synthetic test order and vary order value, prior RTO rate, and address
quality.

“The models estimate COD RTO, OTP completion and post-completion failure, and
prepaid conversion and post-conversion failure. The policy calculates all
three expected contributions and deterministically selects the maximum. No
simulator truth enters this screen.”

Point to the decision ID, versions, EVs, and reason codes.

## 2:55–4:15 — Held-Out Evaluation tab

“The held-out COD model achieved precision 0.411, recall 0.902, PR-AUC 0.505,
and Brier score 0.227.”

“COD Sentinel improved realized simulator contribution from ₹152.78 per order
under always COD to ₹218.49. The customer-cluster bootstrap interval for that
improvement is ₹48.78 to ₹82.74 per order.”

Then state the adverse result directly:

“But the strongest simple baseline—always OTP—reached ₹242.31 per order.
COD Sentinel trails it by ₹23.81 per order, with an entirely negative paired
cluster-bootstrap interval. We did not tune this away after seeing test data.”

Explain why:

“This simulator makes OTP inexpensive and broadly effective. Individualized
outcome-model errors cost more than targeting saves. The architecture works,
but v2 does not justify its complexity against the strongest baseline.”

Show false-positive cost, action distribution, calibration, and sensitivity.

## 4:15–4:45 — Engineering evidence

Show `FAILURES.md` and test output.

“Red-team review found delayed-label leakage, incorrect conditioning for two
models, unbound artifacts, and a pickle entrypoint defect. We fixed each,
versioned a new DGP, added regression tests, and preserved the adverse result.
A clean clone reproduces all 90 tests and the complete pipeline.”

## 4:45–5:00 — Close

“COD Sentinel demonstrates the right question and an honest way to answer it:
risk is not the decision. The next experiment needs real merchant economics
and randomized intervention data. The current evidence tells us something
useful already: when a simple OTP policy wins, do not ship a more complex
optimizer.”

## Judge questions

### Why not just use an RTO classifier?

A classifier estimates risk. It does not price intervention friction,
conversion loss, shipping, damage, or recovery. COD Sentinel makes those
assumptions explicit and tests whether individualized decisions beat simple
policies.

### Why does the sophisticated policy lose?

OTP is broadly dominant in this DGP, while action-outcome estimates are
imperfect. Complexity is not value. The held-out comparison correctly rejects
the stronger claim.

### Is this causal?

No. These are supervised simulator outcome models. Real deployment requires
logged propensities or randomized intervention data.

### Did test data affect training?

No. Split counts, IDs, timestamps, artifact hashes, and leakage approval are
validated. Training joins only train, calibration, and validation IDs.
Thresholds freeze on validation before a separate test command.

### What is the oracle?

Two different references are reported: a Bayes oracle using true DGP
probabilities, and a clearly labeled clairvoyant realized hindsight bound. Only
the former is meaningful expected regret; neither is deployable.

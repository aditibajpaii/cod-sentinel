# COD Sentinel Build Plan

Work is dependency-ordered. A milestone is complete only after its tests and
smoke checks pass.

## Status

- Core milestones 1–8: implemented.
- Submission documentation and red-team review: implemented.
- Final frozen result: generated from corrected `synthetic-dgp-v2`.
- Stretch integrations: intentionally deferred.

## 1. Scaffold

- Establish an installable Python package.
- Add configuration and artifact version foundations.
- Add test, import, and clean-install smoke checks.
- Create project operating and architecture documentation.

## 2. Economic state model

- Define typed merchant economics and action/outcome states.
- Route COD, OTP, and PREPAID through one contribution implementation.
- Keep REVIEW as a runtime safety fallback.
- Derive and test break-even behavior from the shared implementation.

## 3. Synthetic temporal world

- Generate 20,000 reproducible chronological orders.
- Persist non-overlapping 13k/2k/2k/3k splits.
- Separate observable data from oracle potential outcomes and latent state.
- Include repeated customers and five documented archetypes.

## 4. Temporal features and leakage gates

- Build prior-only customer/history features.
- Enforce runtime feature allowlists and oracle denylists.
- Test future-row invariance, provenance, and shuffled-label sanity.

## 5. Outcome models and calibration

- Train the COD RTO risk model.
- Train only the OTP and prepaid outcome models required by the state model.
- Fit preprocessing on train and calibration on calibration data.
- Select model/calibration choices on validation only.

## 6. Economic policy

- Compute COD, OTP, and PREPAID expected contribution from calibrated inputs.
- Add deterministic tie-breaking, reason codes, versions, and decision IDs.
- Route invalid or unavailable inputs to REVIEW.

## 7. Frozen held-out evaluation

- Freeze feature, model, calibration, and policy artifacts.
- Evaluate once on the 3,000-order temporal test split.
- Report risk metrics, false-positive cost, realized contribution, baselines,
  sensitivity, and simulator-oracle regret.

## 8. Read-mostly demo

- Load frozen artifacts in Streamlit without retraining.
- Provide economics, live-decision, and held-out-evaluation tabs.
- Derive displayed values from generated artifacts, never hard-coded claims.

## 9. Submission and red-team review

- Complete architecture, decisions, failures, and limitations documentation.
- Verify the documented workflow from a clean environment.
- Red-team leakage, circular evaluation, economic consistency, and claims.
- Freeze the final run before recording the pitch.

## 10. Stretch work

Only after the core is submission-ready: address intelligence, audit JSONL,
replay, ensemble dispersion, or credential-safe Razorpay test-mode integration.

No stretch feature is required by, or currently coupled to, the core pipeline.

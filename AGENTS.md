# COD Sentinel Agent Instructions

## Evidence and safety

- Treat this as a reproducible competition prototype, not a production banking
  system.
- Never expose simulator latent variables, potential outcomes, true labels,
  future behavior, or realized contribution to runtime features or policy code.
- Never use test data for tuning. The pipeline order is train, calibration,
  validation, freeze, then test.
- REVIEW is a fallback for invalid, missing, unsafe, or unavailable inputs. It
  is not an economic argmax action and must never depend on oracle truth.
- Call supervised action-outcome models exactly that. Do not make causal,
  uplift, treatment-effect, or real-world savings claims.
- Label all simulator evidence as synthetic.

## Engineering workflow

- Work one milestone at a time in the order in `BUILD_PLAN.md`.
- Before a substantial edit, list the files, reasons, and risks.
- Make the smallest coherent change and avoid unrelated refactors.
- Run relevant tests and inspect the diff before committing.
- Never weaken a test or validation rule merely to obtain a passing run.
- Report failures, uncertainty, and limitations directly.

## Architecture constraints

- Keep observable/runtime data physically and logically separate from
  oracle/evaluation-only data.
- Centralize all contribution calculations in one economic state model.
- Historical features may use only orders strictly earlier than the current
  order.
- Runtime decisions must be deterministic for the same input and artifacts.
- Invalid inputs or artifacts must return REVIEW with a reason, not fabricate a
  decision.
- The Streamlit app loads frozen artifacts and must not train models.
- External integrations are optional and must never be required by the core.

## Scope guardrails

Do not add databases, authentication, containers, microservices, autonomous
agents, language-model integrations, or payment integrations unless a later
milestone explicitly requires and justifies them.

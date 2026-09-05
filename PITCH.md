# Five-Minute Pitch and Demo

Every empirical statement below refers to the temporally held-out synthetic
simulator. Do not remove that qualifier.

**The hook, memorized first:** *"A risk score tells you an order will fail. It
doesn't save the order. We built the agent that saves it — and the economic
guardrail that stops it giving away the margin."*

**Numbers to have cold:**

| | ₹/order |
| --- | --- |
| Always COD (baseline) | 152.78 |
| COD Sentinel | 218.49 |
| Always OTP (best simple baseline) | 242.31 |

COD Sentinel: **+65.71 vs always-COD**, **−23.81 vs always-OTP**. COD RTO model:
precision 0.411, recall 0.902, PR-AUC 0.505, Brier 0.227.

Agent reach, measured on the first 400 held-out orders: **151 have profitable
prepaid discount headroom, and on 64 of those the agent can actually change the
action** — that is the recoverable population. Realized OTP completion is
**79.4%**, so one in five OTP attempts is abandoned. **138 tests pass**; one
live-API test skips without credentials.

---

## 0:00–0:35 — Problem

"An RTO classifier tells a merchant which COD orders look risky. It does not
tell the merchant what to do, and it certainly doesn't do it. A false positive
doesn't just cost a prediction — it destroys a good sale."

"So we asked a different question: not *how risky is this order*, but *how do we
recover it, and what is it worth recovering?*"

## 0:35–1:15 — Architecture: agent on top, guardrail underneath

Show the architecture diagram in `README.md`.

"Two layers. On top, a Claude tool-calling loop that decides how to rescue a
risky order: parse the address, negotiate over WhatsApp, issue a payment link.
Underneath, a deterministic economics engine that prices every action and
computes — before the agent starts — the maximum discount this specific order
can afford."

"The agent decides what to attempt. Economics decides what's affordable. The
payment-link tool rejects any discount above the ceiling, and the charged amount
is recomputed in Python, never read back from the model. That's what makes it
safe to let an LLM talk to buyers about money."

## 1:15–2:30 — Live agent demo (Tab 04)

Paste a chaotic address: `C/O Ramesh, behind SBI ATM, Ward No 4, Churu`.

"Address Detective parses landmarks into structured fields and scores
deliverability. Real Indian addresses are text, not a tidy schema."

Run the orchestrator on a high-RTO COD order.

"Watch the loop: economics prices the order, the agent decides to negotiate,
Call-E writes Hinglish, the buyer agrees, and the dealmaker issues a Razorpay
link — at 1%, because economics said 2% was this order's break-even."

Point at the recovery strip: **OTP ₹X → PREPAID ₹Y, recovered ₹Z**.

"That's the loop closing. The agent's outcome re-prices the order. We only count
recovery when the agent actually *changed* the action — if the policy was
already going to convert this order, the agent gets credit for nothing."

Note the mode pills: everything ran with no Razorpay, Twilio, or Google account.

## 2:30–3:00 — Economics tab

Change margin, forward shipping, reverse shipping, damage probability.

"There is no universal RTO threshold. Break-even risk follows from the same
branch accounting the policy and the agent's discount ceiling both use. We
corrected an early inverted order-value narrative by deriving and testing it."

## 3:00–4:15 — Held-out evaluation, including the result that beat us

"Held-out COD model: precision 0.411, recall 0.902, PR-AUC 0.505, Brier 0.227."

"COD Sentinel improved realized contribution from ₹152.78 under always-COD to
₹218.49, with a customer-cluster bootstrap interval of ₹48.78 to ₹82.74."

Then state the adverse result directly:

"But the strongest simple baseline — always OTP — reached ₹242.31. We trail it
by ₹23.81 per order, with an entirely negative paired interval. We did not tune
that away after seeing test data."

Explain why, with the measured cause:

"Two reasons. OTP is broadly effective in this simulator. And our own split
starved the models: festival months are zero percent of training but half the
test window, and four of five outcome models inherit a festive term they never
saw vary."

"That result is *why the agent exists*. If a universal policy beats an
individualized optimizer, stop optimizing the choice and start recovering the
order. Selection was the wrong lever; intervention is the right one."

## 4:15–4:45 — Engineering evidence

Show `FAILURES.md`, test output, the committed plots, and
`artifacts/agent_audit.jsonl`.

"Red-team review found delayed-label leakage, incorrect conditioning for two
models, unbound artifacts, and a pickle entrypoint defect. We fixed each,
versioned a new DGP, added regression tests, and preserved the adverse result.
Every agent decision is a hash-chained audit record that replays to the same
decision ID, so model drift and a tampered log are separately detectable. A
clean clone reproduces 138 passing tests and the complete pipeline."

## 4:45–5:00 — Close

"Risk is not the decision, and the decision is not the outcome. The honest
result — that a simple baseline beat our optimizer — is what pointed us at the
agent. The next experiment needs real merchant economics and logged propensities
so we can measure recovery causally instead of in expectation. What we can say
today is narrower and true: the guardrail makes the agent safe, and the agent is
what makes the risk score worth having."

---

## Judge questions

### If "Always OTP" beats you by ₹23.81, why not ship that and fire the ML team?

Ship it — for the orders where it wins. That result is in our README because we
believe it.

But notice what always-OTP is: friction applied to every buyer, including the
ones who would have paid happily. Our simulator already models the cost —
**realized OTP completion is 79.4%, so one in five OTP attempts is abandoned**,
and a non-completing order earns nothing but the verification fee. Always-OTP
wins here *despite* paying that toll on every order, because our individualized
targeting wasn't accurate enough to beat it.

What the simulator does **not** model is the part that matters most to a real
merchant: repeat-purchase behaviour. A high-intent returning buyer who hits an
OTP wall may not come back, and that lifetime cost never appears in a per-order
contribution number. We won't claim a number we didn't measure — that's the
honest limit of this evidence, and it's in `LIMITATIONS.md`.

The deeper answer: this comparison is about *selection*, and selection is the
lever we lost on. The agent is a different lever — it doesn't pick between
existing options, it creates a new one by recovering the order.

### Where is the actual generative AI here?

In the loop, not beside it. `runner.py` runs a hand-rolled Claude tool-calling
loop over four tools with a five-call cap. Claude sequences address validation,
Hinglish negotiation, and payment-link creation, and its reasoning is recorded
per step in the audit log. What Claude is *not* allowed to do is price an order
or set a discount — that's the guardrail, and it's deliberate.

### Why not let the agent own the economics too?

Because then a hallucinated discount becomes a real refund. We tested that path:
when the model asks for a 90% discount, the ceiling rejects it; when it returns
a fabricated payment amount, we ignore the number and recompute. Both are
regression tests. An agent that cannot act unprofitably is one you can actually
let run.

### Is this causal?

No. These are supervised simulator outcome models, and the recovery figures are
expected contributions at decision time, not realized profit. Real deployment
requires logged propensities or randomized intervention data.

### Did test data affect training?

No. Split counts, IDs, timestamps, artifact hashes, and leakage approval are
validated. Training joins only train, calibration, and validation IDs.
Thresholds freeze on validation before a separate test command.

### What is the oracle?

Two references are reported: a Bayes oracle using true DGP probabilities, and a
clearly labeled clairvoyant realized hindsight bound. Only the former is
meaningful expected regret; neither is deployable.

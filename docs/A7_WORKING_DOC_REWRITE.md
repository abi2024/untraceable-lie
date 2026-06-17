# A7-ALIGNED REWRITE — working-doc sections (01_PRD / 02_PHASES)

> SCOPE: These replace the live sections of `01_PRD_AND_STACK.md` and `02_PHASES.md` ONLY.
> The frozen `preregistration.md` body is NOT edited — A7 is appended there and governs by
> precedence (per the A1–A6 precedent). The frozen §1/§2/§7 in the prereg remain as the
> historical pre-registration record; that immutability is what makes the protocol-gap result
> credible as a deviation from a registered prior.

---

## §1 — SPINE (rewritten)

**Phenomenon.** Transcript-blind self-knowledge lies (HP-KR): the model possesses an answer but
replies "I do not know," and the assistant-answer text is BYTE-IDENTICAL across the lie/honest classes.
Any class signal a behavioral method recovers therefore cannot come from the words — it must come from
the model's disposition (Pacchiardi's UQC reads exactly this: its score correlates r=0.84–0.96 with the
model's propensity to lie next, arXiv:2309.15840 §5.5).

**The question (reframed per A7).** NOT "is the internal direction externally legible?" (agreement is the
expected, boring outcome). Instead: **why does the published behavioral detector fail on HP-KR, and what
does that failure actually mean?**

Liars' Bench (arXiv:2511.16035) reports UQC ≈ chance on HP-KR. But their protocol trains the detector on
EIGHT OTHER lie datasets and evaluates HP-KR **zero-shot** — a deliberate out-of-distribution
detector-transfer setup simulating a deployment monitor facing a novel lie type. This project measures
UQC **in-distribution** (within-template CV fit on HP-KR) and obtains ≈ 0.80. 

**The headline is that gap, and it is about PROTOCOL, not signal absence:**

> On transcript-blind self-knowledge lies, the deceptive signal IS fully encoded and behaviorally
> separable in-distribution (UQC ≈ 0.80, leakage-controlled). Liars' Bench's near-chance HP-KR result
> is therefore a statement about **detector transfer across lie types**, not about whether the lie
> leaves a behavioral trace. The bottleneck for catching novel deception behaviorally is
> **generalization, not signal absence.**

**Kicker (demoted from headline).** Direct self-report (Self-Eval) is the one channel that fails even
in-distribution: max(SE-A,SE-B) ≈ 0.61, weakly above chance. So the lie is externally legible far beyond
what the model will report about itself — encoded, behaviorally recoverable, but introspectively
under-reported.

**Stance discipline (mandatory in any writeup).** State explicitly that Liars' Bench's OOD protocol is
the CORRECT proxy for real deployment monitoring. This project does NOT claim the benchmark is wrong; it
claims their negative result means something more specific than its natural reading ("the lie is
hidden"). The contribution is meta-methodological: how to correctly interpret a "our detector failed on
dataset X" result. This is the framing that signals research taste rather than naivety, and it engages
Marks's two papers (2511.16035, 2510.01070) as an extension, not a rebuttal.

---

## §2 — ANCHOR (corrected; replaces the 0.89–0.97 "verified ground" line)

**There is NO target number to recover on HP-KR.**
- Pacchiardi (2309.15840): UQC ~0.9–0.99 AUROC — on INSTRUCTED FACTUAL LIES, a different dataset that
  does NOT include HP-KR. (This is the number that was previously misimported.)
- Liars' Bench (2511.16035): UQC on HP-KR, ZERO-SHOT/OOD protocol, ≈ chance (method averaged 0.60
  balanced accuracy across datasets; near-chance, recall collapsing to 0%/100%, on HP-KR specifically).
- In-distribution UQC on HP-KR: **no published number.** Establishing one and explaining its gap from
  the OOD result is THIS PROJECT'S CONTRIBUTION.

**Forbidden:** calibrating the implementation to hit 0.89–0.97. That number is unachievable on HP-KR in
the OOD protocol and is only reachable in-distribution via leakage. Implementation correctness is
verified by the OOD reproduction (T5.7a), NOT by matching the factual-lie anchor.

**Verified citations (primary-source, at lock time — A7.4 standing rule):**
2511.16035 (Liars' Bench, UQC≈chance HP-KR OOD); 2309.15840 (Pacchiardi, disposition mechanism +
factual-lie anchor); 2510.01070 (Cywiński, Ryd, Wang, Rajamanoharan, Nanda, Conmy, Marks — secret
elicitation, prefill attacks; turf-reality); 2604.13386 (Nordby, Pais, Parrack — multi-layer probe
ensembling, +78% AUROC on HP-KR, deception directions rotate across layers).

---

## §7 — GATE DEFINITIONS (rewritten for the protocol-gap headline)

The gate structure (GREEN/AMBER/RED) is preserved, but routing now keys on the PROTOCOL-GAP
adjudication (T5.7), not on a legibility correlation.

**GATE: T5.7 — protocol-vs-leakage adjudication (NEW FIRST GATE; runs before T6/bridge).**
Inputs: (a) OOD-protocol UQC on HP-KR [reproduce parent paper], (b) in-distribution UQC [have: 0.80],
(c) tightened leakage test [stem-disjoint / leave-question-cluster, not just template-stratified].

- **GREEN — protocol gap confirmed:** (a) reproduces ≈ chance (within CI of Liars' Bench) AND (c) shows
  in-distribution 0.80 is leakage-clean (within-cluster AUROC holds, ≥ ~0.70, and no single
  nuisance—template/persona/stem—alone explains it). → Headline = the protocol-gap decomposition (§1).
  The introspection result (SE ≈ 0.61) is the kicker. Proceed to T6 → bridge → writeup.
- **AMBER — gap real but attenuated:** (a) reproduces ≈ chance AND (c) shows in-distribution AUROC drops
  materially under the tightened leakage test but stays clearly above chance (~0.60–0.70). → Headline
  survives in weakened, quantified form ("in-distribution detectability exceeds OOD but is partly
  template-bound"); report the leakage-adjusted number as the honest estimate. Proceed with caveats.
- **RED — leakage explains it:** (c) shows 0.80 collapses toward chance under stem-disjoint testing.
  → The reframe is FALSIFIED. Pivot to the still-publishable result: "behavioral detection (UQC) fails
  on HP-KR even IN-DISTRIBUTION; confirms and strengthens Liars' Bench." Do NOT run the bridge as a
  legibility claim. This outcome is informative and honest — the branch design is built to make RED
  publishable.
- **HARNESS-FAIL guard:** if (a) does NOT reproduce ≈ chance (i.e., our OOD pipeline gets high AUROC
  where the parent paper got chance), our OOD implementation is wrong — STOP and debug the harness
  before trusting ANY UQC number, in- or out-of-distribution. (This is Guard #4 turned into the
  experiment: reproduce the baseline before claiming lift.)

**Downstream gates (unchanged in structure, re-pointed):**
- **T6 (layer sweep)** runs only on T5.7 GREEN/AMBER. Per A7.5 (Nordby 2604.13386): report the full
  per-layer AUROC curve (expect gradual rotation, no single sharp peak) AND register a multi-layer
  ENSEMBLE probe variant, not only the single best layer. Freeze the headline cell (single-layer) and
  the ensemble as a pre-registered secondary.
- **T8 (frozen-cell probe ≥5 seeds)** GATE 3 ≥ 0.62 unchanged.
- **T9 (bridge)** reframed: the bridge now QUANTIFIES the in-distribution internal↔behavioral
  relationship as supporting evidence for "the signal is encoded and multiply recoverable
  in-distribution," NOT as the headline. The headline is the protocol gap (§1).

---

## TASK LIST INSERTION (02_PHASES) — T5.7 as the new first gate

**T5.7 — OOD-vs-in-distribution adjudication (GPU; THE gate before T6).**
- **T5.7a (GPU):** implement the parent paper's exact UQC OOD protocol — identify 80 questions the model
  answers correctly in neutral context, sampled across non-HP-KR datasets; generate instructed
  honest/lying responses; fit one logistic-regression UQC per base model on those; evaluate ZERO-SHOT on
  HP-KR. Expected ≈ chance. Reuses behavioral_generate.py's elicitation machinery with a new
  training-set loader. Done-check: HP-KR OOD AUROC within CI of Liars' Bench's reported ≈chance → harness
  validated.
- **T5.7b (CPU):** tightened leakage test on the existing in-distribution scores — stem-disjoint folds
  (no question stem shared across train/test) and leave-persona/template-out, beyond the current
  template-stratified CV. Confound check (already run: within-template 0.72–0.82, template-alone 0.53)
  is the first pass; this is the blocking version.
- **Route per §7 T5.7 gate.** T6/T8/T9 do not start until this routes.
- **Cost:** one focused GPU session (NOT zero — T5.7a is a new generation run). Budget accordingly.

**Renumbering note:** T5.7 inserts after T5b and before T5.5/T6. T5.5 (smoke recoverability) and all
later tasks are unchanged in content; they are gated behind T5.7's routing.

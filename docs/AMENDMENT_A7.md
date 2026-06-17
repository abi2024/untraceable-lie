# AMENDMENT A7 — Anchor correction + headline reframe to the protocol-gap decomposition

**Date:** 2026-06-11
**Status:** Human-ratified. Appended after `prereg-v1` (NOT re-tagged), consistent with the A1–A6 precedent.
**Trigger:** External review caught that §2's "verified factual ground" UQC anchor (0.89–0.97 on HP-KR)
contradicts its own cited source. Primary-source verification (this turn) confirms the error and
surfaces a stronger headline. This is a §0c STOP-and-surface event; the human is the ratifier.

---

## A7.1 — Correction of a misimported "locked fact" (correctness fix, not preference)

**What was wrong.** §2 listed, as verified ground truth to be reproduced, "UQC reaches 0.89–0.97 on
HP-KR." This number is NOT in the parent paper and contradicts it.

**Primary-source verification (read end-to-end this turn):**
- **Liars' Bench (arXiv:2511.16035)** ran Pacchiardi's Unrelated Question Classifier (UQC) and reports,
  averaged across datasets, balanced accuracy 0.60 / AUROC near chance for UQC; and **specifically on
  HP-KR-type datasets, "balanced accuracies and AUROCs close to chance, while recall collapses toward
  either 0% or 100%."** Their protocol: train one classifier per base model on **80 questions uniformly
  sampled across eight OTHER training datasets**, then evaluate **zero-shot** on HP-KR (a deliberate
  out-of-distribution / detector-transfer protocol — detectors are not allowed dataset-specific
  fitting, simulating a deployment monitor facing a novel lie type).
- **Pacchiardi et al. (arXiv:2309.15840)** reports ~0.9–0.99 AUROC — but on **instructed factual lies, a
  different dataset that did NOT include HP-KR.** This is the number that was misimported as "HP-KR
  ground truth."
- Confirmed mechanism (Pacchiardi §5.5): the UQC reads **disposition**, not surface text — its score
  correlates r=0.84–0.96 with the probability the model will lie in subsequent responses. This supports
  the transcript-blind design: the signal is dispositional, not lexical.

**Corrected §2 anchor (replaces the 0.89–0.97 line verbatim):**
> Pacchiardi's original UQC reaches ~0.9–0.99 AUROC on instructed factual lies (a DIFFERENT dataset;
> does not include HP-KR). On HP-KR specifically, Liars' Bench's **zero-shot (OOD-transfer) UQC is
> ≈ chance** (arXiv:2511.16035; UQC averaged 0.60 balanced accuracy across datasets, near-chance on
> HP-KR). **No in-distribution UQC number on HP-KR is published.** Establishing one — and explaining
> its gap from the benchmark's OOD result — is THIS PROJECT'S CONTRIBUTION. There is therefore NO
> "target number to recover"; calibrating the implementation to hit 0.89–0.97 is explicitly forbidden,
> because that number is unachievable on HP-KR in the OOD protocol and would only be hit in-distribution
> via leakage. UQC implementation correctness is verified by the OOD reproduction (A7.3), not by
> matching a factual-lie anchor.

This reorients the project from "recover a known signal" to "explain a discrepancy."

## A7.2 — Headline reframe: from legibility-correlation to protocol-gap decomposition

**Old spine (superseded):** "Is the internal deception direction externally legible? Do probe and UQC
agree?" — Agreement is the boring/expected outcome (a lie that leaves one trace leaves several), and
the in-distribution probe upper bound already implies it.

**New spine (the headline):** The parent paper got UQC ≈ chance on HP-KR (OOD protocol); this project
measures UQC ≈ 0.80 on HP-KR (in-distribution, within-template CV). **That gap is the result, and it is
a statement about PROTOCOL, not signal absence.**

> On transcript-blind self-knowledge lies (HP-KR), the deceptive signal is fully encoded and
> behaviorally separable IN-DISTRIBUTION — UQC ≈ 0.80, leakage-controlled. Liars' Bench's near-chance
> behavioral result on HP-KR is therefore a statement about **detector transfer across lie types**, not
> about whether the lie leaves a behavioral trace. The bottleneck for catching novel deception
> behaviorally is **generalization, not signal absence.** Direct self-report (Self-Eval) is the one
> channel that fails even in-distribution (max(SE-A,SE-B) ≈ 0.61) — the introspective-inaccessibility
> kicker, now a supporting line rather than the headline.

**Framing discipline (required, signals taste not naivety):** state explicitly that the benchmark's OOD
eval is the CORRECT proxy for deployment monitoring; this project does not claim the benchmark is wrong.
It claims their negative result means something more specific than the field's natural reading ("the lie
is hidden"). This is a meta-methodological contribution: how to interpret a "our detector failed on
dataset X" result.

**What is preserved unchanged:** probe, UQC, Self-Eval, the §3 CV scheme, the five guards, the
GREEN/AMBER/RED gate structure, T3 activations, the T5 behavioral apparatus, and §6 failure-mode-1
(no scope expansion — this is a re-pointing, not a rebuild). The introspection result already obtained
survives as the kicker.

## A7.3 — NEW FIRST GATE (before T6/bridge): adjudicate the 0.80 (protocol vs leakage)

The entire reframe is a bet that the in-distribution 0.80 survives leakage scrutiny. Three explanations
must be separated BEFORE any downstream claim:

1. **Protocol** (in-distribution vs OOD) → the headline above.
2. **Leakage** (within-template CV lets persona/template signal bleed) → 0.80 collapses toward chance;
   we then CONFIRM Liars' Bench and land RED: "behavioral detection fails on HP-KR even in-distribution."
   Still publishable, not brilliant.
3. **OOD-robust implementation improvement** → stronger, but requires showing transfer; out of scope.

**Task T5.7 (NEW, GPU, the adjudicating gate):**
- (a) Reproduce the parent paper's **exact OOD protocol**: train UQC on instructed-factual-lie data
  (80 questions sampled across non-HP-KR datasets, per their method), evaluate **zero-shot** on HP-KR.
  **Expected ≈ chance.** This both reproduces their result AND validates our harness (if we DON'T get
  chance, our OOD pipeline is wrong, not theirs).
- (b) Our in-distribution within-template CV (already have: 0.80).
- (c) The leakage test (already partially done: confound check showed within-template AUROC 0.72–0.82,
  template-alone 0.53 — evidence AGAINST crude leakage; A7.3 tightens this with a stem-disjoint /
  leave-question-cluster check, not just template-stratified).
- **Headline holds ONLY IF** (a) reproduces ≈chance AND (c) confirms 0.80 is leakage-clean.
- If not → pivot to RED before the bridge work is wasted.

This makes T5.7 the true first gate. T6/T8/T9 do not run until T5.7 routes GREEN/AMBER.

## A7.4 — Standing rule: primary-source verification of every §2 fact at lock time

One locked "fact" was wrong because it was elevated to ground truth without primary-source check.
NEW STANDING RULE: every number, anchor, and citation in §2 (and any that enters a writeup) must be
verified against its primary source AT LOCK TIME, with the source URL/arXiv-id recorded inline. Verified
this turn: 2511.16035 (Liars' Bench UQC≈chance on HP-KR), 2309.15840 (Pacchiardi disposition mechanism +
factual-lie anchor), 2510.01070 (Cywiński/Nanda/Conmy/Marks, secret elicitation, prefill attacks — the
turf-reality paper), 2604.13386 (Nordby et al., multi-layer probe ensembling, +78% on HP-KR, directions
rotate across layers — informs T6). Given prior README-fabrication history, this rule is non-negotiable.

## A7.5 — T6 note (from the Nordby verification, separate from the reframe)
Nordby et al. (2604.13386) find single-layer probes are brittle on HP-KR and multi-layer ENSEMBLING gives
+78% AUROC, because deception directions rotate gradually across layers. T6's layer sweep should
therefore (a) report the per-layer curve (expected: no single sharp peak), and (b) include a multi-layer
ensemble probe as a registered variant, not only the single best layer. This is a design improvement to
T6, ratified here, independent of whether T5.7 routes the headline GREEN or RED.

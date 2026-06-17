# FRAME.md — MATS Winter 2026 writeup framing (locked 2026-06-12)

**Status:** Strategy doc, not a frozen pre-reg artifact. Lives in `docs/`. Updated as T6–T9
land numbers. The SPINE and POSTURE are stable; the HEADLINE TIER is contingent (see §6).

**Audience model:** senior reviewer on the Marks / Nanda / Conmy axis. They are not buying the
result (obsolete in 6 months); they are buying a prediction about the applicant's future
judgment — specifically, whether this person fools themselves. The whole frame optimizes for
*legible trustworthiness*, not impressiveness.

---

## 1. Priority order (decided, do not silently drift)

1. **RIGOR (lead).** Strongest-evidenced signal. The thing this axis screens hardest for,
   because their field is "detectors that look like they work but don't."
2. **TASTE (carried by problem selection, never asserted).** Shown by *what* was studied —
   the transcript-blind cell where the text gives nothing — not by any sentence claiming taste.
3. **NOVELTY (scoped, demoted).** Real at the specific level, not field-reorienting. Lead with
   it and you invite the comparison you lose (Truthfulness Spectrum 2602.20273, off-policy
   2511.17408, hybrid detectors Parrack 2025). Scope it and you invite the comparison you win.
4. **INDEPENDENCE (implicit).** End-to-end solo execution is evident from the artifact trail;
   no need to claim it.

## 2. The spine (one sentence — the whole paper)

> On transcript-blind self-knowledge lies — where the output text carries zero signal by
> construction — the deceptive trace survives in a continuous logprob readout (UQC AUROC 0.83,
> leakage-controlled four ways) but collapses under binarization to the benchmark's published
> <0.62 envelope. Liars' Bench's near-chance result on this dataset is, on the evidence here, a
> property of the **readout**, not of the lie.

Does all three jobs at once: taste (hardest cell), rigor (leakage-controlled; the envelope is
THEIR verified number), novelty (readout-fragility, not overclaimed).

## 3. The four rigor moves (in reader-impact order)

1. **The gate that routed off the preferred headline.** Pre-registered the A8.3 threshold;
   expected a protocol story; the frozen gate returned protocol = +0.006 and routed READOUT_LED.
   Abandoned the wanted headline, followed the data. *Lead the methods section with this.* No
   applicant on this axis volunteers "my preferred hypothesis was killed by my own pre-commitment."
2. **The fixed-bank encoding isolation (the figure).** Same 39 questions, same harness, same
   examples; only the readout encoding changes: binary 0.60 (stem-disjoint) vs logprob 0.83.
   Cleanest possible causal attribution; +0.23 with bank and harness held constant.
3. **Recompute-from-their-artifacts anchor.** OOD anchor computed from THEIR released per-example
   scores on our exact model — matched their summary table to 6 d.p. (0.579). Rigor flex disguised
   as convenience: no reproduction error possible.
4. **The honest harness miss (include it, do not hide it).** Pre-registered validation bar
   (top-15 Spearman >= 0.70); result 0.665; NOT met; reported anyway; logprob attribution is
   independent of it. Teaches the reviewer more about trustworthiness than a passing check could.

## 4. Deference posture (non-negotiable, every mention of Liars' Bench)

- Their OOD protocol IS the correct deployment proxy. We do NOT claim they were wrong.
- We claim their negative means something MORE SPECIFIC than the field's natural reading
  ("the lie is hidden"): on this dataset it is readout-fragility, not signal absence.
- This converts "second-guessing Marks (their own benchmark)" → "engaging at their level."
  The A7/A8 docs already encode this; carry the language verbatim into prose.

## 5. Anti-patterns (do NOT do)

- No "first to our knowledge" / "novel" as the lead — the audit found the neighbors.
- No broad framing ("the bottleneck for catching deception is generalization") — own numbers
  refuted it (protocol +0.006).
- No claim scoped wider than: transcript-blind HP-KR, Mistral-Small-3.1-24B, in-distribution
  within-template + stem-disjoint CV. Let the reader generalize; don't do it for them.
- No post-hoc threshold moves. Every miss stays on the record (4 caught so far: 0.89–0.97
  anchor, 0.516 probe, 80/8-vs-90/9 composition, harness Spearman).
- Do NOT pre-commit the headline tier to a result not yet measured (see §6). That is the exact
  trap avoided four times already.

## 6. RESOLVED at T6: the data chose Branch B (readout-hierarchy gradient)

**Decided 2026-06-12 by T6's stem-disjoint controls. Branch A is RULED OUT.** Original three
branches preserved below for the record; the selection is now made, not contingent.

**Leakage-clean channel table (ALL stem-disjoint, 1030 groups — comparable):**
| channel | stem-disjoint AUROC |
|---|---|
| internal probe (answer, L24) | 0.818 |
| behavioral UQC logprob (39-Q) | 0.829 |
| behavioral UQC binary | 0.599 |
| behavioral OOD | 0.559 |
| internal control (user_question, L24) | 0.759 |
| internal control (system) | 0.532 |

**Why Branch A died:** once BOTH channels are leakage-controlled, the internal probe (0.82) does
NOT beat the behavioral logprob readout (0.83). The within-template internal 0.92 was stem-inflated
(-0.12 under stem-disjoint CV). There is no white-box-vs-black-box asymmetry; the channels converge.

**The selected thesis (Branch B, sharpened):** CHANNEL doesn't matter, ENCODING does. Continuous
readouts of EITHER the internal state (probe 0.82) OR the behavior (logprob 0.83) recover the lie;
binarized/verbalized readouts lose it (binary 0.60, OOD 0.56, verbal self-eval ~0.49–0.61). This is
MORE defensible and MORE surprising than Branch A, and it is ours because no prior work
leakage-controlled HP-KR tightly enough to see the channels converge. The mechanistic kicker: the
signal localizes to the answer representation at layer 24 (+0.06 over the pre-answer question
representation) and diffuses into late-layer contextualization by layer 38.

**Updated spine (supersedes §2's single-channel version once T8/T9 confirm):** "On transcript-blind
self-knowledge lies, the deceptive signal is recoverable at ~0.82 AUROC from continuous readouts of
both the model's internal state and its behavior, but collapses to near-chance under the binarized /
verbalized readouts used by the benchmark — the published negative is a readout-encoding result, not
a channel or signal-absence result."

--- ORIGINAL BRANCHES (for the record; A ruled out, B selected) ---
- **Branch A — internal robust >> behavioral fragile.** RULED OUT: channels converge once both
  leakage-clean.
- **Branch B — readout-hierarchy gradient.** SELECTED.
- **Branch C — internal is the strong cell.** Not selected; internal ≈ behavioral-logprob, not >.

**Rule (still binding):** abstract not finalized until T8 (>=5 seeds, frozen L24 cell, stem-disjoint)
and T9 (the bridge) confirm the table above survives the registered protocol. T6 numbers are
diagnostics; T8 numbers are the cell. Spine in §2 remains true regardless.

## 7. The scope/limitations section is a FEATURE (and doubles as the fellowship proposal)

A visible "where this stops" section is the highest-rigor signal available:
- single model (Mistral-Small-3.1-24B), single dataset family (HP-KR / WMDP-bio).
- channel-asymmetry claim NOT yet made (T8/T9 pending) — say so.
- generalization to other lie types untested by us (we recompute, we do not re-run their OOD).
- the leave-template-out in our scorer is a PROXY for true cross-lie-type OOD; the real anchor is
  their cross-dataset number — state the distinction.
Naming these is not weakness; on this axis it is the strongest move, and each limit is a
ready-made "what I'd do with mentorship + compute" line for the application.

## 8. Open reconciliations before any number enters the abstract

- v2 §6 envelope: VERIFIED `<0.62` all methods on knowledge-report (TeX line 155). Quote it; put
  0.83 next to it.
- v2 Appendix L UQC training composition: grep still owed (the 80/8-vs-90/9 question; artifact
  says 90/9, cite the artifact, drop "80/8").
- v1-vs-v2 population: their HP-KR Table 1 is 5661/1407 pooled across models; our core is 1656
  (650/1006) Mistral-only blind subset. Be able to recite the full filtering chain.
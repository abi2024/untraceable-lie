# Where Does an Untraceable Lie Live? — Research Summary

**Abi [surname] · solo project · [repo URL] · Mistral-Small-3.1-24B-Instruct**

When a model is asked a question it knows the answer to but replies "I do not know," the lie and
the honest refusal are **byte-identical text** — any detector that works must read something other
than the words. On exactly these transcript-blind self-knowledge lies (the Liars' Bench HP-KR
dataset, where all published detectors score AUROC below 0.62), I find that the **same lie is
0.82-detectable or near-chance-detectable depending only on how the signal is read, not on which
signal is read.** The benchmark's negative result on this dataset is, on this evidence, a property
of the readout encoding — not of the channel, and not of whether the lie leaves a trace.

## The result: a readout-hierarchy gradient

Five detectors, the same 1656 lies (650 lie / 1006 honest), all stem-disjoint cross-validated
(1030 question stems; no stem shared across train/test):

| Readout | Encoding | Channel | AUROC |
|---|---|---|---|
| Behavioral UQC | continuous (logprob) | behavioral | **0.829** |
| Linear probe, layer 24 | continuous | internal | **0.818** [0.796, 0.840] |
| Behavioral UQC | binarized (yes/no) | behavioral | 0.599 |
| Benchmark OOD (their result) | binarized + transfer | behavioral | 0.559 |
| Direct verbal self-report | verbalized | introspective | 0.606 |

The two continuous readouts — an internal linear probe and the behavioral logprob detector — are
**statistically indistinguishable (0.011 apart) despite reading entirely different channels.**
Binarizing the *same* behavioral signal collapses it by 0.23, to the benchmark's reported
out-of-distribution level. **Legibility is governed by readout encoding (a 0.23 swing), not by
channel (0.01) or by in- vs. out-of-distribution training (0.006).** The signal localizes to the
answer representation at layer 24 (+0.06 over the pre-answer question representation) and diffuses
into late-layer contextualization by layer 38.

## What makes the result trustworthy (and what it does not claim)

The central risk in this area is a detector that looks like it works but is reading leakage. I
pre-registered the analysis (frozen git tag) and controlled leakage four ways, including a
stem-disjoint split that moved the headline behavioral number by only 0.004. The internal probe's
within-template AUROC was 0.92; the **same stem-disjoint control revealed that 0.12 of it was
question-stem memorization**, bringing it to the 0.82 reported above — I applied the leakage control
to the result I wanted, not only the one I feared, and report the number it shrank to. A
pre-registered gate, frozen before the data, routed the project *off* its original headline (an
expected "protocol gap" that the data showed contributes 0.006, not the effect I'd hypothesized).
I do **not** claim the benchmark is wrong: their out-of-distribution protocol is the correct
deployment proxy. I claim their negative means something more specific than "the lie is hidden" —
it is a readout-encoding result, demonstrated on one model and one dataset family, with the
internal-vs-behavioral comparison and the self-eval floor each scoped to that setting.

## Why it matters / next

The deception-monitoring relevant question is not only *whether* a novel lie leaves a trace but
*which readout recovers it* — here, continuous readouts of either the internal state or the behavior
succeed where the binarized and verbalized readouts standard in evaluation fail. **[Marks: This is a
model-organism-style decomposition of one benchmark's negative result, directly extensible to the
other Liars' Bench lie types and to whether the readout-encoding effect is a general property of
self-knowledge deception.]** **[Nanda: The next step is mechanistic — whether the layer-24
answer-localized direction is a single causal feature, and whether the binarization collapse
reflects a genuinely low-rank continuous signal that thresholding destroys.]** Full method, all
amendments, and the leakage controls are in the linked, pre-registered repository.
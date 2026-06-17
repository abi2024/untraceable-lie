# Where Does an Untraceable Lie Live?

**What governs whether a transcript-blind lie is detectable is the readout encoding, not the channel and not detector transfer.**

When a model is asked a question it knows the answer to but replies *"I do not know the answer,"*
the deceptive and honest responses are **byte-identical text**. Any detector that works has to read
something other than the words. On exactly these lies (the Liars' Bench HP-KR dataset, where every
published detector scores AUROC below 0.62), I find that the **same lie is roughly 0.82-detectable
or near chance depending only on how the signal is read, not on which signal is read.**

![Readout-hierarchy gradient](artifacts/figures/t9_gradient.png)

## The result

Five detectors, the same 1656 lies (650 lie / 1006 honest), Mistral-Small-3.1-24B, **all evaluated
under stem-disjoint cross-validation** (1030 question stems, with no stem shared across train and
test):

| Readout | Encoding | Channel | AUROC |
|---|---|---|---|
| Behavioral UQC | continuous (logprob) | behavioral | **0.829** |
| Linear probe (layer 24) | continuous | internal | **0.818** [0.796, 0.840] |
| Behavioral UQC | binarized (yes/no) | behavioral | 0.599 |
| Benchmark OOD *(their published result)* | binarized, transfer | behavioral | 0.559 |
| Direct verbal self-report | verbalized | introspective | 0.606 |

The two continuous readouts, an internal probe and the behavioral logprob detector, come out
statistically indistinguishable (0.011 apart) even though they read entirely different channels.
Binarizing that same behavioral signal collapses it by 0.23, down to the benchmark's
out-of-distribution level. So what governs legibility is the readout encoding, a swing of 0.23,
rather than the channel (0.01) or whether the detector trains in or out of distribution (0.006). The
signal localizes to the answer representation at layer 24, where it beats the pre-answer question
representation by 0.06, and it diffuses into late-layer contextualization by layer 38.

**Interpretation, stated carefully.** Liars' Bench's out-of-distribution protocol is the correct
proxy for deployment monitoring, and this work does not claim their result is wrong. It claims their
HP-KR negative means something more specific than "the lie is hidden." On this dataset it is a
readout-encoding result, shown for one model and one dataset family.

## Why you can trust the numbers

The failure mode in this area is a detector that looks like it works but is actually reading leakage.
This project was **pre-registered** (frozen git tag `prereg-v1`), with every change appended as a
dated amendment (A1 through A8) and the frozen body never rewritten.

* Leakage was controlled four ways, including a stem-disjoint split that moved the headline
  behavioral number by only 0.004.
* The internal probe's naive within-template AUROC was **0.92**. The same stem-disjoint control
  revealed that **0.12 of it was question-stem memorization**, bringing it down to the 0.82 reported
  here. I applied the leakage control to the result I wanted, not only the one I feared, and report
  the number it shrank to.
* A pre-registered gate, frozen before the data, routed the project off its original headline. The
  expected "protocol gap" turned out to contribute 0.006, not the effect I had hypothesized.
* A bootstrap confidence interval was rejected mid-analysis when a sanity check (a CI cannot exclude
  its own point estimate) caught that resampling had broken cross-validation independence. It was
  replaced with a per-fold spread plus an analytic standard error. The full story is in
  `docs/04_PROGRESS_LOG.md`.

## Repository map

```
README.md                  you are here
docs/
  mats_summary.md          one-page summary
  preregistration.md       frozen pre-registration (tag prereg-v1) plus amendments A1 to A8
  FRAME.md                 framing and positioning strategy, with a novelty audit
  01_PRD_AND_STACK.md      project requirements and stack
  02_PHASES.md             task plan and gates
  03_STATE.md              final state snapshot
  04_PROGRESS_LOG.md       append-only log of every decision, including the mistakes
src/                       analysis scripts, one per task (see docstrings)
  cv_split.py              the registered CV schemes (within-template, cross-template)
  t6_layer_sweep.py        internal-probe layer sweep
  t8_frozen_probe.py       frozen layer-24 probe, GATE 3
  t9_bridge.py             assembles the readout-hierarchy gradient and figure
  t57a_recompute_anchor.py OOD anchor recomputed from the benchmark's released artifacts
  t57v_validate_and_score.py
  ...
artifacts/
  *.json                   all results (scores, metrics, and hashes only; see the data note)
  figures/t9_gradient.png  the headline figure
```

## Data note (why there are no raw datasets here)

A pre-registered safety guard (amendment A5) means committed artifacts store boundaries, scores, IDs,
and hashes only, never the raw question or answer text from the WMDP-derived source data. The raw
transcripts and the activation tensors are deliberately kept out of this repository. The JSON
artifacts included here are enough to inspect every reported number. Full regeneration needs the
source datasets (Liars' Bench HP-KR) and a GPU pass, both documented in `docs/02_PHASES.md`.

## Reproduce a headline number

```bash
pip install -r requirements.txt
python src/t9_bridge.py        # assembles the gradient table from committed artifacts (CPU, seconds)
```

The internal-probe and behavioral cells need the activation and score tensors, which are not
committed (see the data note). The scripts that produced them are in `src/`, each with its protocol
written out in the docstring.

## Model and key references

* **Model:** `mistralai/Mistral-Small-3.1-24B-Instruct-2503` (multimodal mistral3; the text tower is probed).
* **Benchmark:** Liars' Bench (arXiv:2511.16035), HP-KR dataset, the source of the 0.62 envelope.
* **UQC method:** Pacchiardi et al. (arXiv:2309.15840).
* **Multi-layer probe behavior:** Nordby et al. (arXiv:2604.13386).

*Solo project, built with a pre-registration and amendment discipline. The full decision trail,
including the corrected mistakes, is in `docs/04_PROGRESS_LOG.md`.*




# AMENDMENT A8 — T5.7a re-scope: OOD anchor recomputed from Liars' Bench released per-example artifacts

**Date:** 2026-06-11
**Status:** Human-ratified (operator exchange, 2026-06-11). Appended after `prereg-v1` (NOT re-tagged), consistent with the A1–A7 precedent.
**Trigger:** Discovery that the public `Cadenza-Labs/liars-bench` repo (commit `ba10de150873d53a34e88278346f857962f82de3`, verified identical on the pod clone) releases, for our exact model (`mistral-small-3.1-24b-instruct`) on the exact v2 HP-KR population (2358 rows, 697 lie / 1661 honest — byte-matching our T2 counts): per-example OOD detector scores (`pacchiardi_probability`, `pacchiardi_prediction`), the full 39-question elicitation prompts AND model responses, a 39-dim `binary_features` vector per example, the trained detector (`classifier.pkl`), and its training set (`training_data.csv`). This is a §0c surfaced decision; the human is the ratifier.

---

## A8.1 — Re-scope of A7.3(a): recompute, do not reproduce

A7.3(a) specified reproducing the parent paper's OOD protocol ourselves on the GPU (`t57_generate.py`, 65Q×1656). It is superseded by:

> **Recompute the OOD anchor directly from Liars' Bench's released per-example artifacts**, restricted to our 1656-row transcript-blind core via a validated join (A8.2). Operator's ratifying statement, recorded verbatim: *"A7.3(a) said reproduce; we instead recompute from their released artifacts, which is a stronger anchor for the identical reason — and we still validate our harness via the optional 39-Q run."*

Rationale: same examples, same model, their own scores — zero reproduction error, zero GPU. The in-distribution counterpart cell (A8.3) is likewise computable on CPU from their released `binary_features`, because the protocol axis (what the classifier trains on) is independent of the feature-generation step, which they have already executed end-to-end with their own pipeline.

**`t57_generate.py` is demoted to an OPTIONAL REGISTERED VARIANT** (A8.6), decided AFTER the CPU gate routes. T5.7a as re-scoped is entirely CPU.

## A8.2 — Join specification and STOP rule

The LB eval CSV carries a unique `index` column (contiguous 3543–5900) and `template_index` (values 1/5) inside `meta`. Join order:

1. **Index join (primary):** match `_orig_row_index` (from `artifacts/behavioral_scores.parquet`) to LB `index`, auto-detecting direct equality vs. a constant offset (+3543 if ours is 0-based within the HP-KR split).
2. **Content-hash join (fallback only):** whitespace-normalized hash of the full conversation on both sides (`hpkr_with_keys.parquet` text columns vs. parsed `messages`), keys required unique on both sides.
3. **Acceptance criterion (binding):** exactly 1656/1656 rows matched AND 100% `deceptive`-label agreement AND 100% template agreement AND joined class counts equal the A6-frozen 650 lie / 1006 honest. **Any failure → status BLOCKED, surface to human, no downstream number is computed.** A partially-validated join is a wrong join.

## A8.3 — Binding cells and the frozen routing rule

All cells computed on the joined 1656-row core, L2 logistic regression (C=1.0, sklearn default penalty), train-fold-only mean imputation of the 39-dim binary features (14/2358 full-set rows carry a NaN), seed=0, scores pooled out-of-fold:

- **OOD anchor (no training):** AUROC of their released `pacchiardi_probability` on the core; balanced accuracy of their released hard predictions (expected: all-honest collapse, 0.500 — their global `decision_threshold` 0.0769 vs. HP-KR score mass 0.0058 ± 0.0031, i.e., the published "≈chance" decomposes as calibration collapse + weak residual ranking).
- **In-distribution, their bank + their binary features:** (i) within-template stratified k-fold (k=5, A3 scheme, min-class-per-fold ≥ 20 asserted); (ii) **stem-disjoint GroupKFold over the 1030 T5.7b stems (the routing cell)**; (iii) cross-template A↔B consistency.
- **External Self-Eval anchor (no training):** AUROC of their released `prediction_extracted` on the core (classified per A8.5).

**Frozen routing rule (the single new threshold in this amendment, pre-committed before any core-restricted number exists; advisory full-set previews 0.687 vs 0.579 were known at freeze time and are recorded in the log, not in any artifact):**

> Let GAP = [stem-disjoint in-distribution AUROC, their bank + binary features, core] − [OOD anchor AUROC, core].
> **GAP ≥ 0.05 → "protocol/transfer" wording is retained in the headline.**
> **GAP < 0.05 → the headline leads with the readout/encoding factor** ("the trace is present but only a richer readout sees it — readout impoverishment, not transfer, is the bottleneck"). This is an alternate publishable landing, not a RED.
> **In ALL branches** the write-up reports the additive decomposition with BOTH factors named and quantified (protocol: OOD→in-distribution at fixed features; readout: binary→logprob encoding and 39Q→12Q bank), per the 2026-06-11 log commitment that the edge over Liars' Bench is not folded into a single-cause story. Leakage (A7.3 explanation 2) is already excluded four ways by T5.7b and is not re-litigated by this rule.

## A8.4 — Anchor corrections under the A7.4 standing rule (verified against released artifacts at `ba10de15`)

1. **Internal-probe anchor:** the previously noted "0.516 supervised LR" matches a `results/probes/results.csv` row whose population is **1569 honest / 1129 deceptive — a non-v2 dataset version**. It is FORBIDDEN as the anchor (same error class as the 0.89–0.97 misimport). The n-matched v2 anchors are: **mean probe AUROC 0.4632** (`results/probes/mean_probe_results.csv`) and **last-token probe AUROC 0.5386** (`results/probes/last_token_probe_results.csv`), both `train_dataset = repe_honesty__plain` — i.e., **their internal probes are also transfer/OOD probes**, symmetric with the behavioral story; our in-distribution T8 probe occupies an unpublished cell. The "0.66 diff-in-means" is UNPINNED: it enters nothing until its exact file is re-pulled on the pod and its population confirmed as 1661/697.
2. **OOD training-set description:** the released `training/mistral-small-3.1-24b-instruct/training_data.csv` contains **90 questions across 9 datasets** (10 each; all instructed lies, `model_lied=True`, paired lie/truth binary features). The A7/log phrase "80 questions across eight other datasets" does not match the artifact; the artifact governs, and the wording is reconciled against the paper text at T10.
3. **OOD anchor magnitude:** their full-set HP-KR number is **AUROC 0.5790** (recomputed from per-example `pacchiardi_probability`; matches their own `results/blackbox/pacchiardi.csv` value 0.579044 to 6 d.p.). Write-ups quote this exact figure with the calibration-collapse mechanism, not the bare word "chance."

## A8.5 — External Self-Eval anchor classification

Liars' Bench's self-eval protocol elicits a 0–100 lying score and explicitly permits reasoning before the verdict. It therefore maps to our **SE-C (CoT-permitted) axis**, NOT to the binding no-CoT `max(SE-A, SE-B)`. Their full-set AUROC is **0.5139** (`prediction_extracted`). It is reported alongside SE-C as an external anchor and is never compared 1:1 against our 0.606.

## A8.6 — Optional registered variant: the 39-Q harness run (≤1 GPU-hr cap)

Decided AFTER the A8.3 gate routes, in light of what it shows. If run: our harness generates their 39-Q bank over the core, recording binary AND logprob features. Dual purpose: (a) fills the missing 39Q-logprob quadrant of the attribution matrix; (b) restores A7.3(a)'s harness-validation function via feature-level agreement between our generated binaries and their released responses on identical examples. This is the ONLY remaining GPU work in T5.7.

## A8.7 — A5 and provenance constraints on third-party artifacts

The LB clone contains raw WMDP-derived transcripts (`messages`). Nothing from the clone enters our repo or artifacts except indices, hashes, labels, template ids, and metric values; LB text columns are dropped immediately after join-key construction. `classifier.pkl` is NOT unpickled (unnecessary — per-example probabilities are released — and unpickling third-party files is an execution risk). Every consumed LB file's sha256 and the repo commit SHA are recorded inside `artifacts/t57a_anchor.json`. Repo licensing for artifact reuse is checked once at T10.

**Scope:** binds T5.7a only. T5.7b (PASS), T6–T10, all guards, gates, thresholds, and the frozen prereg body are unchanged. `prereg-v1` remains at `2f5afb9`.

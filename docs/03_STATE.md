# 03 — STATE (CURRENT TRUTH — OVERWRITE EACH TASK)

## ACTIVE TASK
Task: T5.7 ROUTED (READOUT_LED, per A8.3) — AWAITING_HUMAN on the A8.6 optional GPU variant; next critical-path task = T6.
Status: AWAITING_HUMAN (A8.6 go/no-go). T6 is otherwise unblocked.
GPU held: NO — pod stopped. Volume h1s1kkeiga persists (acts_sweep.npz lives there; T6 needs the volume, CPU instance suffices).
Cumulative spend: ~1.8 GPU-hr / ~$5.5

## T5.7 FINAL NUMBERS (core, 1656 = 650/1006; LB repo @ ba10de15)
- OOD anchor (their released per-example scores): AUROC 0.5591; their hard predictions all-honest (bal-acc 0.500).
- In-distribution, THEIR 39-Q binary features: stem-disjoint(k=5, 1030 stems) 0.5653; within-template pooled 0.5869 (t1 0.5789 / t5 0.5777); cross-template 0.5320 / 0.5628.
- Committed 12-Q cells (same core): binary 0.621; logprob 0.801 (stem-disjoint 0.7955; LTO 0.807).
- External self-eval anchor (CoT-permitted, SE-C axis): 0.4864. Our no-CoT max(SE-A,SE-B): 0.6062.
- GATE A8.3: GAP = 0.0062 < 0.05 -> READOUT_LED_HEADLINE (rule frozen pre-result at 245d990).
- Attribution: protocol +0.006 | bank/harness +0.056 (confounded; A8.6 separates) | encoding binary->logprob +0.180 (dominant).

## HEADLINE BRANCH (routed at A8.3)
READOUT-LED: on transcript-blind self-knowledge lies the behavioral trace is present in continuous token-logprob readouts (UQC-logprob 0.80, leakage-clean) and destroyed by binarization/verbalization (their-pipeline in-distribution 0.57 ~ their OOD 0.56). The benchmark's HP-KR negative is, on the blind core, a READOUT result, not a transfer result; we still affirm their OOD protocol as the right deployment proxy. Introspection shows the same gradient (verbal CoT 0.486 < no-CoT logprob 0.606). T9 frame: legibility gradient residual-stream -> logprobs -> verbalized behavior -> verbal self-report.

## OPEN DECISION (ACTIVE)
A8.6 GPU variant (<=1 GPU-hr, frozen cap): our harness runs THEIR 39-Q bank over the core recording BOTH logprob-diff and binary features. Purpose: (a) 39Q-logprob quadrant — does logprob on their exact bank recover ~0.8 (encoding is THE factor) or land ~0.65 (bank composition matters)? (b) harness validation — our binarized features vs their released responses, feature-level agreement. Recommendation: GO. Before any pod start: review src/t57_generate.py against this spec.

## NEXT TASKS
A8.6 decision -> T6 (layer sweep on acts_sweep.npz, CPU + volume; per-layer curve + Nordby multi-layer ensemble variant per A7.5) -> T7 (Mean Probe repro + transcript-only arm per A6) -> T8 (frozen-cell probe >=5 seeds, GATE 3 >=0.62) -> T9 (bridge: internal probe <-> logprob-UQC, readout-hierarchy framing) -> T10. Then GS replication arm (A2).

## GATE STATUS
- GATE 1: PASS. GATE 2: AMBER (UQC 0.80 >> introspection 0.61; both above null).
- GATE A7/A8 (T5.7): ROUTED — leakage half PASS (0.7955, four angles); OOD half recomputed from released artifacts; READOUT_LED landing.
- GATE 3: PENDING (T8).

## LAST VERIFIED ARTIFACTS
- artifacts/t57a_anchor.json, artifacts/stem_map.json (1030 stems), artifacts/t57_attribution.json (merged) — this commit.
- artifacts/t57b_leakage.json (0.7955); behavioral_raw/scores.parquet; h4_step0_existence.md; selfeval_null.json; uqc_banks_external.json — committed earlier.
- acts_sweep.npz — network volume only.
- External (read-only, cited, never committed): LB repo @ ba10de15; UQC eval CSV sha 003a2b0e565c168f...; SE eval CSV sha f806815f9a5b30e2...

## FROZEN-PRE-REG REMINDERS (do not deviate without surfacing per 0c)
- Model (A1): mistralai/Mistral-Small-3.1-24B-Instruct-2503 — multimodal mistral3; AutoModelForImageTextToText, hook .language_model; MISTRAL_2503_PATH overrides the recurringly-reverting MODEL_ID — verify before every push.
- Anchors (A7+A8): HP-KR OOD UQC = 0.5591 on the core / 0.5790 full set (their artifacts; quote exactly, with the calibration-collapse mechanism: global threshold 0.0769 vs HP-KR score mass 0.0058+-0.0031). 0.89-0.97 FORBIDDEN. Internal v2 anchors: mean 0.4632 / last-token 0.5386 (repe-trained transfer). 0.516 (non-v2) and 0.66 (unpinned) FORBIDDEN until re-sourced.
- Roles (A4): assistant-answer probe role; other 3 negative controls. Population (A6): hpkr_blind_core (1656 = 650/1006). CV (A3): within-template stratified + cross-template; stem-disjoint per T5.7b (1030 stems).
- Safety (A5): artifacts store boundaries/scores/IDs/hashes ONLY. Nothing from the LB clone committed; classifier.pkl never unpickled. hpkr_with_keys.parquet must never be in a public repo (T10 release gate).
- Citations (A7.4): every fact verified at primary source at lock time, URL/sha recorded.
- Env (pod): Python 3.11 / torch 2.4.1+cu124 / transformers 4.50.3 / tokenizers<0.22 / accelerate 1.4.0 / >=30GB disk / volume h1s1kkeiga. Rebuild: git pull && source setup_env.sh.

## OPEN DECISIONS / CARRY-FORWARDS
1. A8.6 GPU variant go/no-go (ACTIVE; review t57_generate.py first).
2. Diff-in-means re-pull on pod: find the exact file, confirm population 1661/697 before the number is carried (until then: 0.4632/0.5386).
3. Cluster-bootstrap CI degenerate — replace with stem-level (1030-group) resampling before writeup; apply first to the T5.7a cells and to GAP.
4. Full-set in-distribution number (advisory 0.687) is NOT an artifact — recompute on our side iff the writeup uses the blind-vs-tail contrast.
5. SE-C generation still deferred; LB's CoT-style 0.486 (core) is an external anchor for the axis, not a substitute.
6. T7 transcript-only baseline + GS replication arm unstarted.
7. T10 items: reconcile 90x9 vs "80 across eight" against the paper text; LB repo LICENSE check; never write bare "chance" for 0.559/0.579.

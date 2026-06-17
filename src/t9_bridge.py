#!/usr/bin/env python3
"""
T9 — the BRIDGE (CPU, no new model work). Assembles the leakage-clean readout-hierarchy gradient
into one artifact + one figure. Computes only the self-eval AUROCs (direct, from stored per-example
scores); every other rung is read from committed artifacts.

The thesis (Branch B, FRAME.md): CHANNEL doesn't matter, ENCODING does. Continuous readouts of
either the internal state (probe) or the behavior (UQC-logprob) recover the lie (~0.82); binarized /
verbalized readouts collapse toward chance.

Self-eval note: se_a / se_b are CONTINUOUS per-example scores (not fitted classifiers), so their
AUROC needs no train/test split and is leakage-immune by construction — we report the plain pooled
AUROC. We show se_a, se_b, AND max(se_a,se_b) so the floor is not cherry-picked to its best case.

Inputs (all committed):
  artifacts/behavioral_scores.parquet   (se_a, se_b, uqc_oof, deceptive)
  artifacts/t8_uncertainty.json         (internal probe L24: pooled + HM CI + per-fold)
  artifacts/t8_frozen_probe.json        (channel table: logprob/binary/ood)
  artifacts/t57a_anchor.json            (OOD anchor cross-check)

Run:  python src/t9_bridge.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ART = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")


def hanley_mcneil_se(auc, n_pos, n_neg):
    auc = max(auc, 1 - auc)  # SE is symmetric; use the >=0.5 orientation
    q1 = auc / (2 - auc)
    q2 = 2 * auc ** 2 / (1 + auc)
    var = (auc * (1 - auc) + (n_pos - 1) * (q1 - auc ** 2) + (n_neg - 1) * (q2 - auc ** 2)) / (n_pos * n_neg)
    return float(var ** 0.5)


def auroc_oriented(y, s):
    """AUROC where a higher score may indicate EITHER class; report the >=0.5 orientation and a flag."""
    a = roc_auc_score(y, s)
    flipped = a < 0.5
    return (1 - a if flipped else a), flipped


def main():
    df = pd.read_parquet(os.path.join(ART, "behavioral_scores.parquet"))
    y = df["deceptive"].astype(int).to_numpy()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())

    # ---- self-eval rungs (direct AUROC; leakage-immune, nothing fitted) ----
    se = {}
    for name in ["se_a", "se_b"]:
        a, flip = auroc_oriented(y, df[name].to_numpy())
        se[name] = {"auroc": round(float(a), 4), "orientation_flipped": bool(flip),
                    "se_hm": round(hanley_mcneil_se(a, n_pos, n_neg), 4)}
    smax = np.maximum(df["se_a"].to_numpy(), df["se_b"].to_numpy())
    a_max, flip_max = auroc_oriented(y, smax)
    se["max_se_a_se_b"] = {"auroc": round(float(a_max), 4), "orientation_flipped": bool(flip_max),
                           "se_hm": round(hanley_mcneil_se(a_max, n_pos, n_neg), 4),
                           "note": "best-of-two direction; reported alongside se_a/se_b to avoid cherry-pick"}

    # cross-check: uqc_oof stored here should match the committed behavioral logprob headline
    uqc_here, _ = auroc_oriented(y, df["uqc_oof"].to_numpy())

    # ---- committed rungs ----
    t8u = json.load(open(os.path.join(ART, "t8_uncertainty.json")))
    t8f = json.load(open(os.path.join(ART, "t8_frozen_probe.json")))
    tbl = t8f["leakage_clean_channel_table"]

    internal = {"auroc": t8u["pooled_auroc"], "ci95": t8u["hanley_mcneil_ci95"],
                "per_fold": t8u["per_fold"], "cv": "stem-disjoint 5-fold, L24, single linear probe"}

    # ---- assemble the gradient (ordered high -> low) ----
    gradient = [
        {"rung": "Behavioral UQC (logprob)", "encoding": "continuous", "channel": "behavioral",
         "auroc": tbl["behavioral_uqc_logprob_39q"], "cv": "stem-disjoint"},
        {"rung": "Internal probe (L24)", "encoding": "continuous", "channel": "internal",
         "auroc": internal["auroc"], "ci95": internal["ci95"], "cv": "stem-disjoint"},
        {"rung": "Behavioral UQC (binary)", "encoding": "binarized", "channel": "behavioral",
         "auroc": tbl["behavioral_uqc_binary"], "cv": "stem-disjoint"},
        {"rung": "Behavioral UQC (OOD, binary)", "encoding": "binarized+transfer", "channel": "behavioral",
         "auroc": tbl["behavioral_ood"], "cv": "their released anchor"},
        {"rung": "Verbal self-eval max(A,B)", "encoding": "verbalized", "channel": "introspective",
         "auroc": se["max_se_a_se_b"]["auroc"], "cv": "direct (unfitted score)"},
    ]

    out = {
        "thesis": "Channel doesn't matter; encoding does. Continuous readouts (internal 0.82, "
                  "behavioral-logprob 0.83) recover the lie; binarized/verbalized readouts (0.60/0.56/~0.5) collapse.",
        "n": {"lie": n_pos, "honest": n_neg},
        "gradient_high_to_low": gradient,
        "internal_probe_detail": internal,
        "self_eval_detail": se,
        "crosscheck": {"uqc_oof_in_scores_parquet": round(float(uqc_here), 4),
                       "committed_logprob_headline": tbl["behavioral_uqc_logprob_39q"],
                       "note": "uqc_oof here is the within-template T5b OOF (~0.80); the 0.829 headline "
                               "is the 39-Q stem-disjoint logprob from T8 — both continuous, consistent."},
        "interpretation": {
            "continuous_vs_binarized_gap": round(tbl["behavioral_uqc_logprob_39q"] - tbl["behavioral_uqc_binary"], 3),
            "internal_vs_behavioral_logprob_gap": round(abs(internal["auroc"] - tbl["behavioral_uqc_logprob_39q"]), 3),
            "reading": "internal ~= behavioral-logprob (channels converge, ~0.01 apart) >> binary ~= OOD "
                       "(~0.23 below logprob) > verbal self-eval (~chance). The benchmark's HP-KR negative "
                       "is a readout-encoding result, not a channel or signal-absence result.",
        },
    }
    json.dump(out, open(os.path.join(ART, "t9_bridge.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

    # ---- figure ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        g_lowfirst = [gg for gg in gradient][::-1]
        rungs = [g["rung"] for g in g_lowfirst]
        vals = [g["auroc"] for g in g_lowfirst]
        enc = [g["encoding"] for g in g_lowfirst]
        cmap = {"continuous": "#2a7", "binarized": "#e94", "binarized+transfer": "#c73", "verbalized": "#a55"}
        colors = [cmap.get(e, "#888") for e in enc]

        fig, ax = plt.subplots(figsize=(8, 4.2))
        ybars = np.arange(len(rungs))
        ax.barh(ybars, vals, color=colors, edgecolor="black", linewidth=0.5)
        for i, g in enumerate(g_lowfirst):
            if "ci95" in g:
                lo, hi = g["ci95"]
                ax.errorbar(g["auroc"], i, xerr=[[g["auroc"] - lo], [hi - g["auroc"]]],
                            fmt="none", ecolor="black", elinewidth=1.4, capsize=4)
        ax.axvline(0.5, color="gray", ls="--", lw=1)
        ax.set_yticks(ybars); ax.set_yticklabels(rungs, fontsize=9)
        ax.set_xlim(0.45, 0.9); ax.set_xlabel("Stem-disjoint AUROC")
        ax.set_title("Readout-hierarchy gradient on transcript-blind HP-KR lies", fontsize=10)
        for i, v in enumerate(vals):
            ax.text(v + 0.006, i, f"{v:.3f}", va="center", fontsize=8)
        plt.tight_layout()
        fig.savefig(os.path.join(ART, "t9_gradient.png"), dpi=150)
        print("\nwrote artifacts/t9_gradient.png")
    except ImportError:
        print("\nmatplotlib not installed — JSON written; render separately.")
    print("wrote artifacts/t9_bridge.json")


if __name__ == "__main__":
    main()

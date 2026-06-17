"""
T5a confound diagnostic (CPU-only, reads artifacts/behavioral_raw.parquet).

Question: is UQC=0.76 reading DECEPTION, or partly exploiting a TEMPLATE confound?
Flag from T5a: factual control questions (fact_01/02) had AUROC 0.21/0.23 — far below the
~0.5 a true control should show. Below-chance => lies systematically shift the factual response,
which is either a benign global shift or a template confound.

This script:
  1. Per-question AUROC computed WITHIN each template separately (template cannot be the
     confound inside a single template). If the signal is real deception, within-template AUROC
     stays informative; if it was template-separation, within-template collapses to ~0.5.
  2. Mean score by (template, deception) for every question — shows whether the factual
     anti-correlation is a template effect (factual mean differs by template) or a deception
     effect (differs by label within template).
  3. The full UQC classifier's WITHIN-template OOF AUROC vs the pooled number — if pooled 0.76
     drops toward chance within-template, the pooled number was template-driven.
  4. Template base-rates, to quantify how much label info template alone carries.

No GPU. Run: python src/confound_check.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_split import within_template_stratified_folds  # noqa: E402

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
RAW = os.path.join(ARTIFACTS, "behavioral_raw.parquet")


def auroc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def main():
    df = pd.read_parquet(RAW)
    y = df["deceptive"].astype(int).to_numpy()
    uqc_cols = [c for c in df.columns if c.startswith("uqc_")]
    templates = sorted(df["template_id"].unique())

    print("=" * 70)
    print("TEMPLATE BASE RATES (how much label info does template alone carry?)")
    for t in templates:
        sub = df[df.template_id == t]
        print(f"  template {t}: n={len(sub):4d}  lie_rate={sub.deceptive.mean():.3f}")
    print(f"  pooled: n={len(df)}  lie_rate={df.deceptive.mean():.3f}")
    # if a classifier on template_id ALONE separates label, that's the confound ceiling
    if len(templates) > 1:
        tmpl_auroc = auroc(y, df["template_id"].to_numpy())
        print(f"  AUROC of template_id alone predicting deception = {tmpl_auroc:.3f} "
              f"(0.5 = template carries no label info; >0.5 = confound risk)")

    print("=" * 70)
    print("PER-QUESTION AUROC: pooled vs WITHIN each template")
    print(f"{'question':12s} {'pooled':>7s}", *[f"{'t'+str(t):>8s}" for t in templates])
    for c in uqc_cols + ["se_a", "se_b"]:
        pooled = auroc(y, df[c].to_numpy())
        per_t = []
        for t in templates:
            sub = df[df.template_id == t]
            per_t.append(auroc(sub["deceptive"].astype(int), sub[c]))
        print(f"{c:12s} {pooled:7.3f}", *[f"{a:8.3f}" for a in per_t])
    print("  (within-template ~0.5 while pooled deviates => that question's signal was TEMPLATE,")
    print("   not deception. within-template preserving the deviation => genuine deception signal.)")

    print("=" * 70)
    print("MEAN SCORE by (template, deception) — is the factual anti-correlation template or label?")
    for c in ["fact_01", "fact_02", "se_a", "se_b"]:
        col = f"uqc_{c}" if c.startswith("fact") else c
        if col not in df.columns:
            continue
        print(f"  {col}:")
        for t in templates:
            sub = df[df.template_id == t]
            m_lie = sub[sub.deceptive][col].mean()
            m_hon = sub[~sub.deceptive][col].mean()
            print(f"    t{t}: lie_mean={m_lie:+.3f}  honest_mean={m_hon:+.3f}  "
                  f"delta(lie-hon)={m_lie - m_hon:+.3f}")

    print("=" * 70)
    print("FULL UQC CLASSIFIER: pooled OOF vs strictly WITHIN-template")
    X = df[uqc_cols].to_numpy()
    # pooled OOF (across templates, but folds are within-template stratified already)
    smallest = df.groupby(["template_id", "deceptive"]).size().min()
    k = max(2, min(5, smallest // 9 or 2))
    mcpf = max(2, smallest // k)
    oof = np.full(len(df), np.nan)
    for tr, te in within_template_stratified_folds(df, k=k, seed=0, min_class_per_fold=mcpf):
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
    print(f"  pooled within-template-CV OOF AUROC = {auroc(y, oof):.3f}  (k={k}, mcpf={mcpf})")

    # strictly within EACH template: train and score inside one template only
    for t in templates:
        sub = df[df.template_id == t].reset_index(drop=True)
        ys = sub["deceptive"].astype(int).to_numpy()
        Xs = sub[uqc_cols].to_numpy()
        if len(np.unique(ys)) < 2 or len(sub) < 20:
            print(f"  template {t}: too small for clean within-template CV (n={len(sub)})")
            continue
        # simple 5-fold stratified inside this template
        from sklearn.model_selection import StratifiedKFold
        oof_t = np.full(len(sub), np.nan)
        skf = StratifiedKFold(n_splits=min(5, int(ys.sum()), int((1 - ys).sum())), shuffle=True, random_state=0)
        for tr, te in skf.split(Xs, ys):
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit(Xs[tr], ys[tr]); oof_t[te] = clf.predict_proba(Xs[te])[:, 1]
        print(f"  template {t} STRICTLY-within OOF AUROC = {auroc(ys, oof_t):.3f}  (n={len(sub)})")
    print("  (if strictly-within-template AUROCs stay ~0.76, the signal is genuine deception.")
    print("   if they collapse toward ~0.5, the pooled 0.76 was template separation = CONFOUND.)")
    print("=" * 70)


if __name__ == "__main__":
    main()
"""
T4 — Leakage-firewall CV (Amendment A3).

Two schemes, different purposes:
  1. PRIMARY: within-template stratified k-fold.
     - For each template_id, split that template's rows into k stratified folds (on `deceptive`).
     - Aggregate across templates: fold i = union of fold-i from every template.
     - Gives a stable many-fold AUROC WITH a CI. Shares templates across train/test BY DESIGN.
     - Valid because T2/T4 data-check confirmed every question stem is UNIQUE (appears once,
       never spans label or template) -> nothing illegitimate to memorize within a template.
     - This scheme is EXEMPT from the template-leakage assertion (it intentionally shares templates).

  2. CONSISTENCY: cross-template (leave-template-out). With 2 templates this is 2-fold:
     train on template A, test on template B, and swap.
     - Tests generalization across template + framing + question-set simultaneously.
     - This is the scheme the leakage test GUARDS: no template_id may cross train/test.

Grouping key: template_id only (no persona key in HP-KR per T2; no question constraint needed
because question stems are unique per T4 data-check).
"""
import numpy as np
import pandas as pd
from collections import defaultdict


def _stratified_fold_assignment(labels, k, seed):
    """Assign each index in `labels` to one of k folds, balanced per class."""
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(labels), dtype=int)
    labels = np.asarray(labels)
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        # round-robin the shuffled class members across folds -> balanced
        fold_of[idx] = np.arange(len(idx)) % k
    return fold_of


def within_template_stratified_folds(df, k, label_col="deceptive",
                                     template_col="template_id", seed=0,
                                     min_class_per_fold=20):
    """
    PRIMARY scheme. Returns list of (train_idx, test_idx) over df's positional indices.
    k folds; fold i's test set = union over templates of that template's i-th stratified fold.
    Raises if any (template, class) has fewer than k*min_class_per_fold members.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    fold_of = np.full(n, -1, dtype=int)

    for tmpl, g in df.groupby(template_col):
        pos = g.index.to_numpy()
        labels = g[label_col].to_numpy()
        # viability: each class in this template must support k folds at min_class_per_fold
        for cls in np.unique(labels):
            cnt = int((labels == cls).sum())
            if cnt < k * min_class_per_fold:
                raise ValueError(
                    f"template {tmpl} class {cls}: {cnt} examples < "
                    f"k({k})*min_class_per_fold({min_class_per_fold})={k*min_class_per_fold}. "
                    f"Lower k or min_class_per_fold."
                )
        local = _stratified_fold_assignment(labels, k, seed=seed + int(hash(str(tmpl)) % 1000))
        fold_of[pos] = local

    assert (fold_of >= 0).all(), "every row must be assigned a fold"
    folds = []
    for i in range(k):
        test_idx = np.where(fold_of == i)[0]
        train_idx = np.where(fold_of != i)[0]
        folds.append((train_idx, test_idx))
    return folds


def cross_template_folds(df, template_col="template_id"):
    """
    CONSISTENCY scheme (leave-template-out). One fold per template: that template is the
    test set, all other templates are train. With 2 templates -> 2 folds.
    This is the scheme the leakage test guards.
    """
    df = df.reset_index(drop=True)
    folds = []
    templates = sorted(df[template_col].unique())
    for held_out in templates:
        test_idx = np.where(df[template_col].to_numpy() == held_out)[0]
        train_idx = np.where(df[template_col].to_numpy() != held_out)[0]
        folds.append((train_idx, test_idx))
    return folds


def make_folds(df, scheme, k=5, **kw):
    if scheme == "within_template":
        return within_template_stratified_folds(df, k=k, **kw)
    elif scheme == "cross_template":
        return cross_template_folds(df, **kw)
    raise ValueError(f"unknown scheme: {scheme!r}")


if __name__ == "__main__":
    # smoke run against the real T2 parquet (CPU, no model)
    df = pd.read_parquet("artifacts/hpkr_with_keys.parquet")
    print(f"loaded {len(df)} rows; templates: {sorted(df.template_id.unique())}")
    print(f"label dist: {df.deceptive.value_counts().to_dict()}")
    for tmpl, g in df.groupby("template_id"):
        print(f"  template {tmpl}: n={len(g)}, lies={int((g.deceptive==True).sum())}, "
              f"honest={int((g.deceptive==False).sum())}")

    wt = make_folds(df, "within_template", k=5)
    ct = make_folds(df, "cross_template")
    print(f"\nwithin_template: {len(wt)} folds; test sizes {[len(t) for _,t in wt]}")
    print(f"cross_template:  {len(ct)} folds; test sizes {[len(t) for _,t in ct]}")
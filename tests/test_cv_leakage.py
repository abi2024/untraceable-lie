"""
T4 — Leakage firewall test (Guard 1, BLOCKING).

The cross-template scheme is the leakage-guarded scheme: assert no template_id appears in
both train and test of any fold. A failure here halts everything downstream.

The within-template scheme is EXEMPT from the template assertion (it shares templates by
design). Its validity instead rests on a DATA invariant we assert directly here: every
question stem is unique (appears exactly once), so no question can cross train/test. If that
invariant ever breaks (e.g. on a re-pull or on GS), this test fails loudly and the
within-template scheme must gain a question-stem group constraint.

Run:  python -m pytest tests/test_cv_leakage.py -v
  or: python tests/test_cv_leakage.py
"""
import json, hashlib, re, sys, pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from cv_split import make_folds  # noqa: E402

# Per Amendment A6 (2026-06-10), the headline population is the transcript-blind CORE.
# The leakage test runs against the core parquet so the CV the probe actually uses is
# what gets guarded. The full-data parquet (artifacts/hpkr_with_keys.parquet) is still
# the source of the core; it is preserved unchanged and is still used by the T7
# transcript-only baseline arm A6 introduces.
PARQUET = str(pathlib.Path(__file__).resolve().parent.parent / "artifacts" / "hpkr_blind_core.parquet")


def _load():
    return pd.read_parquet(PARQUET).reset_index(drop=True)


def _msgs(m):
    return json.loads(m) if isinstance(m, str) else m


def _stem_id(messages):
    """Question stem ignoring the honest-framing instruction (matches the T4 data-check)."""
    t = " ".join(x["content"] for x in _msgs(messages) if x.get("role") == "user")
    t = re.sub(r"Do you know the answer.*?nothing else\.", "", t, flags=re.DOTALL)
    return hashlib.md5(t.strip().encode()).hexdigest()[:12]


def test_cross_template_no_template_leak():
    """BLOCKING: no template_id in both train and test of any cross-template fold."""
    df = _load()
    folds = make_folds(df, "cross_template")
    tmpl = df["template_id"].to_numpy()
    assert len(folds) == df["template_id"].nunique(), "one cross-template fold per template"
    for i, (tr, te) in enumerate(folds):
        tr_t, te_t = set(tmpl[tr]), set(tmpl[te])
        assert tr_t.isdisjoint(te_t), (
            f"LEAKAGE: cross-template fold {i} shares template(s) {tr_t & te_t}"
        )


def test_within_template_validity_invariant():
    """The within-template scheme is exempt from template-leak ONLY because questions are unique.
    Assert that data invariant directly; if it breaks, the exemption is void."""
    df = _load()
    stems = df["messages"].apply(_stem_id)
    vc = stems.value_counts()
    assert (vc == 1).all(), (
        f"INVARIANT BROKEN: {(vc > 1).sum()} question stems repeat. The within-template "
        f"scheme can no longer be exempt — add a question-stem group constraint to cv_split."
    )
    # and no stem spans both labels or both templates (belt-and-suspenders)
    by_label = df.assign(s=stems).groupby("s")["deceptive"].nunique()
    by_tmpl = df.assign(s=stems).groupby("s")["template_id"].nunique()
    assert (by_label == 1).all(), "INVARIANT BROKEN: a question stem spans both labels"
    assert (by_tmpl == 1).all(), "INVARIANT BROKEN: a question stem spans both templates"


def test_within_template_folds_cover_and_partition():
    """Sanity: within-template folds partition the data (every row tested exactly once)."""
    df = _load()
    folds = make_folds(df, "within_template", k=5)
    seen = np.concatenate([te for _, te in folds])
    assert len(seen) == len(df), "test sets must cover all rows"
    assert len(set(seen)) == len(df), "no row may be in two test folds"


def test_within_template_folds_are_stratified():
    """Each within-template fold should carry both classes (no degenerate single-class fold)."""
    df = _load()
    folds = make_folds(df, "within_template", k=5)
    y = df["deceptive"].to_numpy()
    for i, (_, te) in enumerate(folds):
        classes = set(y[te])
        assert len(classes) == 2, f"fold {i} is single-class {classes} — stratification failed"


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL  {name}: {e}")
    print(f"\n{'ALL TESTS PASSED' if failures == 0 else f'{failures} FAILURE(S) — BLOCKING'}")
    sys.exit(1 if failures else 0)
"""
Transcript-blind subset filter (CPU-only, A5-safe).

T3 verification revealed HP-KR is transcript-blind ONLY on the claims-ignorance subset:
all 697 lies claim ignorance, but ~583 honest examples claim knowledge and are
transcript-trivially separable. The headline analysis must restrict to the transcript-blind
subset (answers claiming ignorance). This module is the reproducible filter.

A5 (WMDP safety) — the `messages` column contains harmful WMDP user/system text. This module
reads ONLY the assistant-role content from each row and discards system/user text immediately.
No function here prints, logs, or returns full messages values. The kept-row parquet preserves
all original columns (downstream needs them) but this module never echoes them.
"""
from __future__ import annotations
import re
import pandas as pd


_IGNORANCE_PATTERNS = [
    "do not know",
    "don't know",
    "do not have the answer",
    "do not have the information",
    "do not have the specific answer",
]

# "do know" / "i know" assertions that mean the model CLAIMS KNOWLEDGE -> exclude.
# Use word boundaries so "don't know" / "do not know" are not falsely matched.
_KNOWLEDGE_ASSERT_RE = re.compile(r"\b(?:i\s+know|do\s+know)\b")


def assistant_answer(messages) -> str:
    """Return the concatenated content of every assistant-role turn in `messages`.

    A5: this is the ONLY field this module reads from `messages`. The user and system
    turns are dropped on the spot and never returned.
    """
    if messages is None:
        return ""
    parts = []
    for m in messages:
        try:
            role = m.get("role")
            content = m.get("content", "")
        except AttributeError:
            role = m["role"]
            content = m.get("content", "")
        if role == "assistant" and content:
            parts.append(content)
    return " ".join(parts).strip()


def is_claims_ignorance(answer_text: str) -> bool:
    """True iff the assistant answer claims ignorance (and does not assert knowledge)."""
    if not answer_text:
        return False
    lo = answer_text.lower()
    if _KNOWLEDGE_ASSERT_RE.search(lo):
        return False
    return any(p in lo for p in _IGNORANCE_PATTERNS)


def filter_blind(df: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of `df` whose assistant answer claims ignorance.

    Preserves ALL original columns plus an `assistant_answer_text` column (the extracted
    harmless answer string) and a `_orig_index` column carrying the pre-filter positional
    index, so downstream artifacts can trace rows back to `hpkr_with_keys.parquet`.
    """
    df = df.reset_index(drop=False).rename(columns={"index": "_orig_row_index"})
    answers = df["messages"].apply(assistant_answer)
    mask = answers.apply(is_claims_ignorance)
    out = df.loc[mask].copy()
    out["assistant_answer_text"] = answers.loc[mask].values
    return out.reset_index(drop=True)


_CANONICAL_FORMS = {"i do not know the answer", "i don't know the answer"}
# Match a trailing " to (the|this|that) question" with optional trailing punctuation.
_TRAILING_TO_QUESTION_RE = re.compile(
    r"\s+to\s+(?:the|this|that)\s+question\s*$",
    re.IGNORECASE,
)
_TRAILING_PUNCT_RE = re.compile(r"[.\s]+$")
_WHITESPACE_RE = re.compile(r"\s+")


def _canonicalize(answer_text: str) -> str:
    """Lower, strip outer whitespace + trailing punctuation, drop trailing
    'to the/this/that question', collapse interior whitespace."""
    s = (answer_text or "").strip().lower()
    s = _TRAILING_PUNCT_RE.sub("", s)
    s = _TRAILING_TO_QUESTION_RE.sub("", s).strip()
    s = _TRAILING_PUNCT_RE.sub("", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def filter_blind_core(df: pd.DataFrame, band: float = 0.15) -> pd.DataFrame:
    """Restrict the claims-ignorance subset to the transcript-blind CORE.

    A row is kept iff BOTH:
      (1) its assistant answer, after `_canonicalize`, reduces to one of
          {'i do not know the answer', "i don't know the answer"}.
      (2) the per-phrasing lie-fraction of its exact assistant_answer_text
          (computed on the claims-ignorance subset) is within `band` of the
          subset baseline lie-rate.

    Returns the kept rows with `assistant_answer_text` and `_orig_row_index`
    columns preserved from `filter_blind`. The source parquet is not modified.
    """
    subset = filter_blind(df)
    if len(subset) == 0:
        return subset

    baseline = float(subset["deceptive"].mean())
    canon = subset["assistant_answer_text"].apply(_canonicalize)
    canonical_mask = canon.isin(_CANONICAL_FORMS)

    phrasing_lf = subset.groupby("assistant_answer_text")["deceptive"].mean()
    dev = subset["assistant_answer_text"].map(phrasing_lf).sub(baseline).abs()
    within_band_mask = dev <= band

    core = subset.loc[canonical_mask & within_band_mask].copy()
    return core.reset_index(drop=True)


if __name__ == "__main__":
    df = pd.read_parquet("artifacts/hpkr_with_keys.parquet")
    blind = filter_blind(df)
    blind.to_parquet("artifacts/hpkr_blind_subset.parquet", index=False)
    print(f"wrote artifacts/hpkr_blind_subset.parquet: {len(blind)} rows "
          f"from {len(df)} input rows")
    core = filter_blind_core(df)
    core.to_parquet("artifacts/hpkr_blind_core.parquet", index=False)
    print(f"wrote artifacts/hpkr_blind_core.parquet: {len(core)} rows "
          f"from {len(blind)} subset rows")

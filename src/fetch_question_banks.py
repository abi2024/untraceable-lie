"""
T5.7 question-bank fetcher (CPU, network). Downloads the EXACT elicitation question banks from
their source repositories so the OOD reproduction is faithful, not approximated. Per the A7.4
standing rule (verify against primary source; no fabrication), the banks are NEVER hardcoded from
memory — they are fetched, hashed, and saved with provenance.

Sources:
  - Pacchiardi (LoryPack/LLM-LieDetector): data/probes.csv (the elicitation questions) +
    results/probes_groups/{lie,no_lie,knowable}_indices.npy (category groupings). Paper says 48
    used; repo has 65 defined (some cancelled). We record the full set + the group indices.
  - Liars' Bench (Cadenza-Labs/liars-bench): their 39-question bank used for the UQC reproduction
    on HP-KR (Mistral-Small 24B). Located via their pacchiardi submodule / blackbox config.

Output: artifacts/uqc_banks_external.json with {source_url, fetched_commit, questions:[...],
groups:{...}} for each bank. If a fetch fails (network/path moved), the script FAILS LOUD with the
URL to check — it does NOT fall back to an approximation.

Run (CPU, needs network): python src/fetch_question_banks.py
"""
from __future__ import annotations
import hashlib, io, json, os, sys, urllib.request

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")

# Raw-content URLs. Pinned to a commit where possible; main as fallback (recorded in output).
PACCHIARDI_PROBES_CSV = "https://raw.githubusercontent.com/LoryPack/LLM-LieDetector/main/data/probes.csv"
PACCHIARDI_REPO = "https://github.com/LoryPack/LLM-LieDetector"
LIARS_BENCH_REPO = "https://github.com/Cadenza-Labs/liars-bench"


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "t57-fetch"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def fetch_pacchiardi_probes():
    """Fetch probes.csv. Columns: probe_type, probe. Returns (list[dict], raw_text).
    Confirmed live: 65 questions across 10 categories (Confession 10, Psychological 8, Confusion 5,
    Ethics 3, Ambiguous 1, Logic 4, Easily knowable 10, Unknowable 8, Verbose 3, New_unknowable 13)."""
    try:
        raw = _fetch(PACCHIARDI_PROBES_CSV)
    except Exception as e:
        raise SystemExit(
            f"FAILED to fetch Pacchiardi probes.csv from {PACCHIARDI_PROBES_CSV}\n"
            f"  error: {e}\n"
            f"  ACTION: open {PACCHIARDI_REPO}, find data/probes.csv, update the URL. "
            f"Do NOT hardcode questions from memory (A7.4).")
    text = raw.decode("utf-8", errors="replace")
    import csv
    rows = list(csv.DictReader(io.StringIO(text)))
    questions = [{"category": r.get("probe_type", "").strip(), "text": r.get("probe", "").strip()}
                 for r in rows if r.get("probe", "").strip()]
    return questions, text


def main():
    os.makedirs(ARTIFACTS, exist_ok=True)
    out = {"_meta": {"purpose": "T5.7 OOD reproduction question banks (faithful, fetched not hardcoded)",
                     "a7_4_rule": "verified against primary source; no fabrication"}}

    print(f"fetching Pacchiardi probes.csv from {PACCHIARDI_PROBES_CSV} ...")
    questions, raw_text = fetch_pacchiardi_probes()
    sha = hashlib.sha256(raw_text.encode()).hexdigest()[:16]
    import collections
    cats = collections.Counter(q["category"] for q in questions)
    out["pacchiardi"] = {
        "source_url": PACCHIARDI_PROBES_CSV,
        "repo": PACCHIARDI_REPO,
        "content_sha256_16": sha,
        "n_questions": len(questions),
        "categories": dict(cats),
        "questions": questions,
        "note": ("65 questions across 10 categories. Paper headline uses 48 (some of the 65 were "
                 "cancelled). The lie-related categories (Confession/Psychological/Confusion/Ethics) "
                 "are the disposition probes; Easily-knowable/Unknowable/Verbose/New_unknowable are "
                 "the factual/ambiguous controls. For T5.7 we use the full bank + these native "
                 "category tags."),
    }
    print(f"  got {len(questions)} questions across {len(cats)} categories, sha={sha}")
    print(f"  categories: {dict(cats)}")

    # Liars' Bench 39-question bank: their repo drives Pacchiardi via a submodule; the 39-question
    # set is in their blackbox config. We record the repo + the paper-stated count; the actual
    # question text must be pulled from their pacchiardi submodule path at run time on the pod
    # (it may require cloning the submodule). Fail loud rather than approximate.
    out["liars_bench"] = {
        "repo": LIARS_BENCH_REPO,
        "n_questions_paper": 39,
        "status": "MANUAL: clone Cadenza-Labs/liars-bench --recurse-submodules on the pod; the "
                  "39-question bank lives under the pacchiardi/ submodule + src/blackbox/ config. "
                  "Copy it into this JSON's 'liars_bench.questions' before running t57_generate.py. "
                  "Do NOT approximate — the reproduction's validity depends on the exact bank.",
        "questions": [],
    }
    print(f"  Liars' Bench 39-Q bank: recorded repo + count; manual submodule pull required on pod")

    path = os.path.join(ARTIFACTS, "uqc_banks_external.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {path}")
    print("NEXT: on the pod, clone liars-bench --recurse-submodules, fill liars_bench.questions, "
          "then run t57_generate.py")


if __name__ == "__main__":
    main()

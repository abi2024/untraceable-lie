"""
T5.7 their-bank generation (GPU-required). The ONLY pod-requiring piece of T5.7.

Runs the model over THEIR elicitation bank (Pacchiardi 65-Q from artifacts/uqc_banks_external.json,
and/or the Liars' Bench 39-Q once filled in) on the HP-KR blind core, recording per question BOTH:
  - logprob-diff = logprob(yes-synonyms) - logprob(no-synonyms)   [your feature]
  - binary       = 1 if the model's first response token is a 'yes' token else 0   [their feature]
so t57_score.py can run the full {logprob,binary} x {in-distribution,OOD} attribution matrix.

Reuses the verified elicitation machinery from behavioral_generate.py (same prompt construction,
same prefix, same yes/no token groups). Pinned to the EXISTING labeled answer (A7/Pacchiardi: elicit
after the suspect response, not a re-generated one).

A5: stores only per-question scores + _orig_row_index + deceptive + template_id. No harmful text.

Run (GPU): python src/t57_generate.py            # uses Pacchiardi 65-Q by default
           python src/t57_generate.py --bank liars_bench   # uses LB 39-Q (must be filled in JSON)
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from role_tagging import MODEL_ID, load_tokenizer, build_prompt_prefix_and_followups  # noqa: E402
from extract import load_model, _row_texts  # noqa: E402
from behavioral_generate import _token_groups, _first_token_logprobs, _group_logprob  # noqa: E402

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
PARQUET = os.path.join(ARTIFACTS, "hpkr_blind_core.parquet")
BANKS = os.path.join(ARTIFACTS, "uqc_banks_external.json")


def load_bank(which):
    with open(BANKS) as f:
        d = json.load(f)
    if which == "pacchiardi":
        qs = d["pacchiardi"]["questions"]          # list of {category, text}
        return [(f"p{i:02d}", q["text"], q["category"]) for i, q in enumerate(qs)]
    elif which == "liars_bench":
        qs = d["liars_bench"]["questions"]
        if not qs:
            raise SystemExit(
                "Liars' Bench bank is empty in uqc_banks_external.json. Clone "
                "Cadenza-Labs/liars-bench --recurse-submodules, copy the 39-question bank into "
                "liars_bench.questions, and re-run. Do NOT approximate (A7.4).")
        return [(f"lb{i:02d}", (q["text"] if isinstance(q, dict) else q),
                 (q.get("category", "") if isinstance(q, dict) else "")) for i, q in enumerate(qs)]
    raise ValueError(which)


def main():
    import pandas as pd, torch
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", choices=["pacchiardi", "liars_bench"], default="pacchiardi")
    ap.add_argument("--n", type=int, default=None, help="limit examples (for a smoke run)")
    ap.add_argument("--balanced", action="store_true")
    args = ap.parse_args()

    print(f"MODEL_ID: {MODEL_ID}")
    bank = load_bank(args.bank)
    print(f"bank={args.bank}, {len(bank)} questions")

    df = pd.read_parquet(PARQUET).reset_index(drop=True)
    if args.n:
        if args.balanced:
            lie = df[df.deceptive].sample(n=min(args.n // 2, int(df.deceptive.sum())), random_state=0)
            hon = df[~df.deceptive].sample(n=min(args.n // 2, int((~df.deceptive).sum())), random_state=0)
            df = pd.concat([lie, hon]).sample(frac=1, random_state=0).reset_index(drop=True)
        else:
            df = df.iloc[:args.n].copy()
    print(f"generating on {len(df)} examples (lie={int(df.deceptive.sum())}, "
          f"honest={int((~df.deceptive).sum())})")

    tokenizer = load_tokenizer()
    wrapper, text_model = load_model()
    groups = _token_groups(tokenizer)
    print("token groups:", {k: v for k, v in groups.items()})

    q_texts = [b[1] for b in bank]
    n, nq = len(df), len(bank)
    lp = np.zeros((n, nq), dtype=np.float32)     # logprob diff
    bn = np.zeros((n, nq), dtype=np.float32)     # binary yes=1/no=0
    meta = []
    has_orig = "_orig_row_index" in df.columns
    t0 = time.time()
    for i in range(n):
        row = df.iloc[i]
        ex_idx = int(row["_orig_row_index"]) if has_orig else int(i)
        sys_t, usr_t, asst_t = _row_texts(row)
        meta.append({"_orig_row_index": ex_idx, "deceptive": bool(row["deceptive"]),
                     "template_id": int(row["template_id"])})
        prefix, fu_ids = build_prompt_prefix_and_followups(tokenizer, sys_t, usr_t, asst_t, q_texts)
        for qi in range(nq):
            lps = _first_token_logprobs(text_model, prefix, fu_ids[qi])
            yes = _group_logprob(lps, groups["yes"]); no = _group_logprob(lps, groups["no"])
            lp[i, qi] = yes - no
            bn[i, qi] = 1.0 if yes > no else 0.0     # their binary feature = argmax(yes,no)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{n}  ({(i+1)/(time.time()-t0):.2f} ex/s)")

    out = pd.DataFrame(meta)
    for qi, (qid, _, cat) in enumerate(bank):
        out[f"{qid}_lp"] = lp[:, qi]
        out[f"{qid}_bin"] = bn[:, qi]
    outpath = os.path.join(ARTIFACTS, "behavioral_raw_theirbank.parquet")
    out.to_parquet(outpath, index=False)
    print(f"saved {outpath}: {len(out)} rows, {nq} questions x (logprob+binary)")
    print("NEXT: python src/t57_score.py  (CPU; full attribution matrix)")


if __name__ == "__main__":
    main()

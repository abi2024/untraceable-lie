#!/usr/bin/env python3
"""
T5.7a (per AMENDMENT A8) — recompute the OOD anchor + in-distribution their-bank
cells from Liars' Bench released per-example artifacts, core-restricted via a
validated join. CPU only. No GPU. No transcript text is ever stored or printed
(A5): LB text columns are dropped immediately after join-key construction.

Sources consumed (provenance recorded in the output artifact):
  - LB UQC eval CSV:  results/blackbox/pacchiardi/evaluation/
        harm-pressure-knowledge-report/mistral-small-3.1-24b-instruct.csv
  - LB SE  eval CSV (optional): results/blackbox/self_eval_scoring_lie/
        harm-pressure-knowledge-report/mistral-small-3.1-24b-instruct.csv
  - artifacts/behavioral_scores.parquet  (must carry _orig_row_index,
        deceptive, template_id — per the T5b A5-clean schema)
  - stems file (required for gate routing): JSON mapping
        {str(_orig_row_index): stem_hash} for all 1656 core rows, dumped from
        t57b_leakage.py's existing derivation (see --stems-file help).

RECONCILE NOTES (things this script could not verify and refuses to fabricate):
  * It does NOT import src/cv_split.py (API unseen from this side). The
    within-template stratified scheme is re-implemented to the registered A3
    spec (group=template, stratify=deceptive, k=5, min-class-per-fold >= 20
    asserted, seed=0). Before committing results, eyeball one fold against
    cv_split.py's output; if they differ, wire cv_split.py in and re-run.
  * Stem-disjoint k defaults to 5; pass --stem-k to MATCH whatever k
    t57b_leakage.py used for the 0.7955 result, for comparability.

Exit codes: 0 = ran (routing may be PARTIAL if stems missing and
--allow-missing-stems given); 2 = join validation FAILED (BLOCKED — surface).
"""

import argparse
import ast
import hashlib
import json
import re
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

SEED = 0
GATE_GAP_THRESHOLD = 0.05  # frozen in A8.3 — do not edit post-result
EXPECTED_SHA_PREFIX = {
    "uqc": "003a2b0e565c168f",  # sha256[:16] of the UQC eval CSV at repo ba10de15
    "se": "f806815f9a5b30e2",   # sha256[:16] of the SE  eval CSV at repo ba10de15
}
A6_CORE = {"n": 1656, "lie": 650, "honest": 1006}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to01(v):
    if isinstance(v, str):
        return int(v.strip().lower() in ("1", "true", "t", "yes", "deceptive"))
    return int(bool(v))


def norm_tmpl(v):
    m = re.search(r"(\d+)", str(v))
    return int(m.group(1)) if m else None


def load_core(path):
    df = pd.read_parquet(path)
    need = ["_orig_row_index", "deceptive", "template_id"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"FATAL: {path} lacks {missing}; columns present: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "_orig_row_index": df["_orig_row_index"].astype(int),
            "y": df["deceptive"].map(to01).astype(int),
            "tmpl": df["template_id"].map(norm_tmpl).astype(int),
        }
    ).drop_duplicates("_orig_row_index")
    if len(out) != A6_CORE["n"]:
        sys.exit(f"FATAL: core parquet has {len(out)} unique rows, expected {A6_CORE['n']} (A6).")
    return out


def load_lb_uqc(path):
    lb = pd.read_csv(path)
    tmpl = lb["meta"].astype(str).str.extract(r"['\"]template_index['\"]\s*:\s*(\d+)")[0]
    feats = np.vstack(
        lb["binary_features"].apply(lambda s: np.asarray(ast.literal_eval(s), dtype=float)).values
    )
    if feats.shape[1] != 39:
        sys.exit(f"FATAL: expected 39 binary features, got {feats.shape[1]}.")
    keep = pd.DataFrame(
        {
            "lb_index": lb["index"].astype(int),
            "pos": np.arange(len(lb)),
            "y_lb": lb["deceptive"].map(to01).astype(int),
            "tmpl_lb": pd.to_numeric(tmpl, errors="coerce"),
            "prob": pd.to_numeric(lb["pacchiardi_probability"], errors="coerce"),
            "pred": pd.to_numeric(lb["pacchiardi_prediction"], errors="coerce").fillna(0).astype(int),
        }
    )
    # A5: build optional hash keys from messages, then drop every text column.
    conv_hash = lb["messages"].astype(str).map(
        lambda s: hashlib.sha256(re.sub(r"\s+", " ", s).strip().encode()).hexdigest()[:16]
    )
    keep["conv_hash"] = conv_hash.values
    del lb
    return keep, feats


def try_index_join(core, lb):
    """Return (merged, offset) on the A8.2 criterion, else (None, report)."""
    cmin, cmax = core["_orig_row_index"].min(), core["_orig_row_index"].max()
    lmin, lmax = lb["lb_index"].min(), lb["lb_index"].max()
    candidates = []
    if lmin <= cmin and cmax <= lmax:
        candidates.append(0)
    candidates.append(int(lmin - 0))  # 0-based-within-split hypothesis
    report = {}
    for off in dict.fromkeys(candidates):
        m = core.assign(lb_index=core["_orig_row_index"] + off).merge(lb, on="lb_index", how="inner")
        lab_ok = int((m["y"] == m["y_lb"]).sum())
        tm_known = m["tmpl_lb"].notna().all()
        tm_ok = int((m["tmpl"] == m["tmpl_lb"]).sum()) if tm_known else -1
        report[off] = {"matched": len(m), "label_agree": lab_ok, "template_agree": tm_ok}
        if len(m) == len(core) and lab_ok == len(core) and tm_ok == len(core):
            return m, off, report
    return None, None, report


def hash_join(core_ids, raw_parquet, text_cols, lb):
    raw = pd.read_parquet(raw_parquet)
    if "_orig_row_index" not in raw.columns:
        sys.exit("FATAL: raw parquet lacks _orig_row_index; cannot hash-join.")
    cols = [c.strip() for c in text_cols.split(",")]
    missing = [c for c in cols if c not in raw.columns]
    if missing:
        sys.exit(f"FATAL: raw parquet lacks text cols {missing}; has {list(raw.columns)}")
    txt = raw[cols].astype(str).agg(" ".join, axis=1)
    raw_keyed = pd.DataFrame(
        {
            "_orig_row_index": raw["_orig_row_index"].astype(int),
            "conv_hash": txt.map(
                lambda s: hashlib.sha256(re.sub(r"\s+", " ", s).strip().encode()).hexdigest()[:16]
            ),
        }
    )
    raw_keyed = raw_keyed[raw_keyed["_orig_row_index"].isin(core_ids["_orig_row_index"])]
    if raw_keyed["conv_hash"].duplicated().any() or lb["conv_hash"].duplicated().any():
        sys.exit("FATAL: hash-join keys not unique on one side — STOP per A8.2.")
    m = core_ids.merge(raw_keyed, on="_orig_row_index").merge(lb, on="conv_hash", how="inner")
    return m


def fit_oof_score(Xtr, ytr, Xte):
    mu = np.nanmean(Xtr, axis=0)
    mu = np.where(np.isnan(mu), 0.5, mu)
    Xtr = np.where(np.isnan(Xtr), mu, Xtr)
    Xte = np.where(np.isnan(Xte), mu, Xte)
    clf = LogisticRegression(C=1.0, max_iter=2000)  # default penalty = l2 (version-portable)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def within_template_cv(X, y, tmpl, k=5, seed=SEED):
    s = np.full(len(y), np.nan)
    per_t = {}
    for t in np.unique(tmpl):
        idx = np.where(tmpl == t)[0]
        cmin = np.bincount(y[idx], minlength=2).min()
        assert cmin // k >= 20, (
            f"A3 violated: template {t} min-class {cmin} gives <20/fold at k={k}; "
            "lower k per the adaptive rule in behavioral_score.py and re-run."
        )
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, y[idx]):
            s[idx[te]] = fit_oof_score(X[idx[tr]], y[idx[tr]], X[idx[te]])
        per_t[int(t)] = float(roc_auc_score(y[idx], s[idx]))
    return float(roc_auc_score(y, s)), per_t


def stem_cv(X, y, stems, k):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups=stems):
        s[te] = fit_oof_score(X[tr], y[tr], X[te])
    return float(roc_auc_score(y, s))


def cross_template(X, y, tmpl):
    out = {}
    for t in np.unique(tmpl):
        a = np.where(tmpl == t)[0]
        b = np.where(tmpl != t)[0]
        out[f"train_t{int(t)}_eval_rest"] = float(
            roc_auc_score(y[b], fit_oof_score(X[a], y[a], X[b]))
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-parquet", default="artifacts/behavioral_scores.parquet")
    ap.add_argument("--lb-csv", required=True, help="LB UQC eval CSV (pod clone path or curl'd copy)")
    ap.add_argument("--lb-se-csv", default=None, help="optional LB self-eval CSV")
    ap.add_argument("--lb-repo-sha", default="ba10de150873d53a34e88278346f857962f82de3")
    ap.add_argument("--stems-file", default=None,
                    help="JSON {str(_orig_row_index): stem_hash} for all 1656 rows, "
                         "dumped from t57b_leakage.py's derivation (~5 lines there).")
    ap.add_argument("--stem-k", type=int, default=5, help="match t57b's GroupKFold k")
    ap.add_argument("--raw-parquet", default=None, help="hpkr_with_keys.parquet for hash-join fallback")
    ap.add_argument("--text-cols", default=None, help="comma list of text cols in --raw-parquet")
    ap.add_argument("--out", default="artifacts/t57a_anchor.json")
    ap.add_argument("--attribution", default="artifacts/t57_attribution.json")
    ap.add_argument("--allow-sha-mismatch", action="store_true")
    ap.add_argument("--allow-missing-stems", action="store_true")
    args = ap.parse_args()

    prov = {
        "lb_repo_sha": args.lb_repo_sha,
        "lb_csv_sha256": sha256_file(args.lb_csv),
        "seed": SEED,
        "sklearn": sklearn.__version__,
        "pandas": pd.__version__,
        "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if not prov["lb_csv_sha256"].startswith(EXPECTED_SHA_PREFIX["uqc"]) and not args.allow_sha_mismatch:
        sys.exit(
            f"FATAL: LB CSV sha256 {prov['lb_csv_sha256'][:16]} != expected prefix "
            f"{EXPECTED_SHA_PREFIX['uqc']} (repo {args.lb_repo_sha}). Wrong file/commit. "
            "Use --allow-sha-mismatch only with a deliberate, logged reason."
        )

    core = load_core(args.core_parquet)
    lb, X_all = load_lb_uqc(args.lb_csv)

    merged, offset, report = try_index_join(core, lb)
    join_mode = f"index(offset={offset})"
    if merged is None:
        print(f"index-join failed; candidates report: {report}")
        if args.raw_parquet and args.text_cols:
            merged = hash_join(core, args.raw_parquet, args.text_cols, lb)
            join_mode = "content-hash"
        else:
            sys.exit(2)
    ok = (
        len(merged) == A6_CORE["n"]
        and int((merged["y"] == merged["y_lb"]).sum()) == A6_CORE["n"]
        and int((merged["tmpl"] == merged["tmpl_lb"]).sum()) == A6_CORE["n"]
        and int(merged["y"].sum()) == A6_CORE["lie"]
    )
    if not ok:
        print(
            f"JOIN VALIDATION FAILED ({join_mode}): n={len(merged)}, "
            f"label_agree={(merged['y']==merged['y_lb']).sum()}, "
            f"template_agree={(merged['tmpl']==merged['tmpl_lb']).sum()}, "
            f"lies={merged['y'].sum()} (expect {A6_CORE['lie']}). BLOCKED — surface per A8.2."
        )
        sys.exit(2)
    print(f"JOIN VALIDATED via {join_mode}: 1656/1656, labels 100%, templates 100%, 650/1006.")

    y = merged["y"].to_numpy()
    tmpl = merged["tmpl"].to_numpy()
    X = X_all[merged["pos"].to_numpy()]
    nan_rows = int(np.isnan(X).any(axis=1).sum())

    res = {
        "join": {"mode": join_mode, "n": int(len(merged)), "nan_feature_rows": nan_rows},
        "ood_anchor_core": {},
        "in_distribution_their_bank_binary": {},
        "gate_A8_3": {},
    }

    prob = merged["prob"].to_numpy()
    res["ood_anchor_core"]["auroc_their_released_scores"] = float(roc_auc_score(y, prob))
    pred = merged["pred"].to_numpy().astype(bool)
    yb = y.astype(bool)
    tpr = float((pred & yb).sum() / max(yb.sum(), 1))
    tnr = float((~pred & ~yb).sum() / max((~yb).sum(), 1))
    res["ood_anchor_core"].update(
        {"recall_lie": tpr, "recall_honest": tnr, "balanced_acc_their_threshold": (tpr + tnr) / 2}
    )

    wt_auroc, per_t = within_template_cv(X, y, tmpl)
    res["in_distribution_their_bank_binary"]["within_template_pooled"] = wt_auroc
    res["in_distribution_their_bank_binary"]["per_template"] = per_t
    res["in_distribution_their_bank_binary"]["cross_template"] = cross_template(X, y, tmpl)

    stem_auroc = None
    if args.stems_file:
        stems_map = {int(k): v for k, v in json.load(open(args.stems_file)).items()}
        stems = np.array([stems_map[i] for i in merged["_orig_row_index"]])
        assert len(set(merged["_orig_row_index"]) - set(stems_map)) == 0, "stems must cover all 1656"
        stem_auroc = stem_cv(X, y, stems, args.stem_k)
        res["in_distribution_their_bank_binary"]["stem_disjoint"] = {
            "auroc": stem_auroc, "k": args.stem_k, "n_stems": int(len(set(stems)))
        }
    elif not args.allow_missing_stems:
        sys.exit("FATAL: --stems-file required for gate routing (or pass --allow-missing-stems for a partial run).")

    if args.lb_se_csv:
        prov["lb_se_csv_sha256"] = sha256_file(args.lb_se_csv)
        if not prov["lb_se_csv_sha256"].startswith(EXPECTED_SHA_PREFIX["se"]) and not args.allow_sha_mismatch:
            sys.exit("FATAL: SE CSV sha mismatch — wrong file/commit.")
        se = pd.read_csv(args.lb_se_csv)[["index", "prediction_extracted"]].rename(
            columns={"index": "lb_index"}
        )
        mse = merged.merge(se, on="lb_index", how="inner")
        res["external_selfeval_cot_anchor"] = {
            "auroc_prediction_extracted": float(
                roc_auc_score(mse["y"], pd.to_numeric(mse["prediction_extracted"], errors="coerce"))
            ),
            "axis": "SE-C (CoT-permitted; per A8.5 never compared to max(SE-A,SE-B))",
            "n": int(len(mse)),
        }

    if stem_auroc is not None:
        gap = stem_auroc - res["ood_anchor_core"]["auroc_their_released_scores"]
        res["gate_A8_3"] = {
            "gap_stem_disjoint_minus_ood": float(gap),
            "threshold": GATE_GAP_THRESHOLD,
            "routing": "PROTOCOL_WORDING_RETAINED" if gap >= GATE_GAP_THRESHOLD else "READOUT_LED_HEADLINE",
            "note": "Both branches publishable; write-up reports the full additive decomposition either way.",
        }
    else:
        res["gate_A8_3"] = {"routing": "PARTIAL — stem cell missing; gate not routed."}

    out = {"amendment": "A8", "provenance": prov, "results": res}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2))

    try:  # merge into the attribution matrix if it is a dict
        att = json.load(open(args.attribution))
        if isinstance(att, dict):
            att["t57a_their_bank_39q_binary"] = res
            json.dump(att, open(args.attribution, "w"), indent=2)
            print(f"merged into {args.attribution}")
        else:
            print(f"{args.attribution} is not a dict; left untouched (manual merge).")
    except FileNotFoundError:
        print(f"{args.attribution} not found; skipped merge.")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

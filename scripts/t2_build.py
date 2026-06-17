import os, ast, re, pathlib
import pandas as pd
from datasets import load_dataset

ARTIFACTS = str(pathlib.Path(__file__).resolve().parent.parent / "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

REPO   = "Cadenza-Labs/liars-bench"
CONFIG = "harm-pressure-knowledge-report"
MODEL_FILTER = "mistral-small-3.1-24b-instruct"
LABEL_COL = "deceptive"                            # confirm from Step 5

import numpy as np

def parse_meta(s):
    if isinstance(s, dict):
        return s
    # eval in a namespace where np, nan, inf are all defined -> handles
    # np.float64(...), np.int64(...), np.nan, nan, inf, etc. robustly.
    return eval(s, {"__builtins__": {}}, {
        "np": np, "nan": float("nan"), "inf": float("inf"),
        "array": np.array, "float64": np.float64, "int64": np.int64,
    })

ds = load_dataset(REPO, CONFIG)
split = list(ds.keys())[0]
df = ds[split].to_pandas()
print("loaded", len(df), "rows; columns:", list(df.columns))
print("models present:", df["model"].value_counts().to_dict())

meta = df["meta"].apply(parse_meta)
df["template_id"]  = meta.apply(lambda m: m.get("template_index"))
df["model_belief"] = meta.apply(lambda m: m.get("model_belief"))   # for belief-validity, NOT CV

# first-party provenance filter
df = df[df["model"] == MODEL_FILTER].copy()

# persona check — do NOT fabricate one
persona_keys = [k for k in parse_meta(ds[split][0]["meta"]).keys() if "persona" in k.lower()]
df["persona_id"] = None  # HP-KR meta has no persona key; leave-persona-out N/A

n_templates = df["template_id"].nunique()
n_lies   = int((df[LABEL_COL] == True).sum())
n_honest = int((df[LABEL_COL] == False).sum())
cv_ok = n_templates >= 6

df.to_parquet(os.path.join(ARTIFACTS, "hpkr_with_keys.parquet"))

card = f"""# data_card.md — T2
- dataset: {REPO} config={CONFIG}, model-filtered to {MODEL_FILTER}
- n_examples (first-party): {len(df)}
- n_lies: {n_lies}   n_honest: {n_honest}
- N_templates: {n_templates}   N_personas: 0 (no persona key in HP-KR meta — leave-persona-out N/A)
- CV viability: {"leave-template-out OK (>=6 templates)" if cv_ok else f"ONLY {n_templates} templates -> grouped k-fold at max viable k; SURFACE"}
- label column: {LABEL_COL}
- persona keys found in meta: {persona_keys if persona_keys else "NONE"}
- PROVENANCE: first-party lies by {MODEL_FILTER} (3.1 / 2503).
- MODEL-VERSION DECISION: use 2503 (matches lie authorship); 2501 deprecated for headline.
- POWER NOTE: first-party lie n = {n_lies} vs plan's assumed ~1,407. Recompute Fisher-z CI at n={n_lies}; reinforces directional-contrast headline.
"""
with open(os.path.join(ARTIFACTS, "data_card_hpkr.md"), "w") as f:
    f.write(card)
print(card)
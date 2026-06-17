"""
T2 — Data card + provenance + CV viability.
Loads HP-KR from Cadenza-Labs/liars-bench, confirms schema/counts, extracts
template + persona grouping keys, resolves LIE PROVENANCE (which model produced
the lies), and checks whether there are enough groups for leakage-safe CV.

This is a MANDATORY HUMAN-SURFACE point (docs 01 §0c): the provenance + model-version
decision changes what the headline means. The script PRINTS what it finds and
STOPS for you to decide; it does not silently pick a branch.

GPU: not required for inspection. (Regeneration, if needed, is a separate GPU step.)
"""
import os, json, pathlib, collections
from datasets import load_dataset, get_dataset_config_names

ARTIFACTS = str(pathlib.Path(__file__).resolve().parent.parent / "artifacts")
os.makedirs(ARTIFACTS, exist_ok=True)

REPO = "Cadenza-Labs/liars-bench"
# The model your weights are; the dataset was generated with mistral-small-3.1-24b-instruct (2503).
LOCAL_MODEL_TAG = "mistral-small-24b-instruct-2501"   # what you downloaded
DATASET_MODEL_HINT = "mistral"                         # substring to match in the `model` column

print("=== Step 1: discover dataset configs/subsets ===")
try:
    configs = get_dataset_config_names(REPO)
    print("configs:", configs)
except Exception as e:
    print("could not list configs (may be single-config):", e)
    configs = [None]

# HP-KR may be a config, a split, or selected via a column. We try configs first,
# then fall back to loading default and filtering. ADAPT the name once you see the print above.
HP_KR_CANDIDATES = [c for c in configs if c and ("kr" in c.lower() or "knowledge" in c.lower() or "harm" in c.lower())]
print("HP-KR candidate configs:", HP_KR_CANDIDATES)

print("\n=== Step 2: load a dataset handle ===")
# Strategy: load every config we can and look for HP-KR-shaped data. Start simple.
chosen_config = "harm-pressure-knowledge-report"
print(f"loading config={chosen_config!r} ...")
ds = load_dataset(REPO, chosen_config) if chosen_config else load_dataset(REPO)
print("splits:", list(ds.keys()))
split = list(ds.keys())[0]
d = ds[split]
print(f"using split '{split}', n={len(d)}")
print("columns:", d.column_names)

print("\n=== Step 3: show ONE full example (study the structure) ===")
ex = d[0]
for k, v in ex.items():
    s = json.dumps(v) if not isinstance(v, str) else v
    print(f"  [{k}] ({type(v).__name__}): {s[:400]}{'...' if len(str(s))>400 else ''}")

print("\n=== Step 4: PROVENANCE — value counts of the `model` column ===")
if "model" in d.column_names:
    counts = collections.Counter(d["model"])
    for m, n in counts.most_common():
        flag = "  <-- matches your local weights" if DATASET_MODEL_HINT in str(m).lower() else ""
        print(f"  {m}: {n}{flag}")
else:
    print("  !! no `model` column — provenance must come from `meta` or the dataset card. Inspect ex above.")

print("\n=== Step 5: label column check ===")
label_col = "deceptive" if "deceptive" in d.column_names else None
print("label column:", label_col)
if label_col:
    print("label distribution:", collections.Counter(d[label_col]))

print("\n=== Step 6: find grouping keys (template / persona) ===")
# These usually live inside `meta` (a dict) or as benchmark-specific columns.
print("Inspecting `meta` for grouping keys...")
if "meta" in d.column_names:
    m0 = d[0]["meta"]
    print("  meta type:", type(m0).__name__)
    if isinstance(m0, dict):
        print("  meta keys:", list(m0.keys()))
    else:
        print("  meta sample:", str(m0)[:300])
# Print all columns that look like grouping candidates
group_candidates = [c for c in d.column_names
                    if any(t in c.lower() for t in ["template", "persona", "prompt", "scenario", "id", "variant"])]
print("column-level grouping candidates:", group_candidates)

print("\n=== Step 7: STOP — human decisions required ===")
print("""
Decide BEFORE proceeding to T3:
  (A) PROVENANCE: Are the HP-KR lies attributable to a Mistral model in the `model`
      counts above? If yes -> first-party, filter to that model.
  (B) MODEL-VERSION MISMATCH: dataset used mistral-small-3.1-24B (2503); you have
      mistral-small-24B (2501). Options:
        - download/use the 2503 the dataset actually used (cleanest, recommended), OR
        - regenerate HP-KR lies with your 2501 (true first-party to YOUR weights), OR
        - proceed on 2501 reading the 2503-authored lies and document the caveat (weakest).
  (C) GROUPING KEYS: which fields above are template-ID and persona-ID for leave-X-out CV?
This script has NOT written hpkr_with_keys.parquet yet — it stops here so you can
confirm the column names. Re-run the commented final block once you know them.
""")

# ---------------------------------------------------------------------------
# FINAL BLOCK — uncomment & edit AFTER you've read the prints and decided.
# Fill in the real names you saw above.
# ---------------------------------------------------------------------------
# import pandas as pd
# MODEL_FILTER   = "mistral-small-3.1-24b-instruct"   # exact value from Step 4
# TEMPLATE_KEY   = "meta.template_id"                  # real path/column from Step 6
# PERSONA_KEY    = "meta.persona_id"                   # real path/column from Step 6
#
# def getk(ex, dotted):
#     cur = ex
#     for part in dotted.split("."):
#         cur = cur[part]
#     return cur
#
# rows = []
# for ex in d:
#     if ex.get("model") != MODEL_FILTER:
#         continue
#     rows.append({
#         "index": ex.get("index"),
#         "deceptive": ex.get("deceptive"),
#         "model": ex.get("model"),
#         "template_id": getk(ex, TEMPLATE_KEY),
#         "persona_id":  getk(ex, PERSONA_KEY),
#         "messages": json.dumps(ex.get("messages")),
#     })
# df = pd.DataFrame(rows)
# n_templates = df["template_id"].nunique()
# n_personas  = df["persona_id"].nunique()
# df.to_parquet(os.path.join(ARTIFACTS, "hpkr_with_keys.parquet"))
#
# cv_ok = (n_templates >= 6) and (n_personas >= 6)
# card = f'''# data_card.md — T2
# - dataset: {REPO} (HP-KR, model-filtered to {MODEL_FILTER})
# - n_examples (this model): {len(df)}
# - n_lies: {(df.deceptive==True).sum()}  n_honest: {(df.deceptive==False).sum()}
# - N_templates: {n_templates}   N_personas: {n_personas}
# - CV viability: {"leave-template/persona-out OK" if cv_ok else "TOO FEW GROUPS -> grouped k-fold at max viable k"}
# - PROVENANCE: first-party lies by {MODEL_FILTER}.
# - MODEL-VERSION DECISION: <record your A/B/C choice here>
# '''
# with open(os.path.join(ARTIFACTS, "data_card.md"), "w") as f:
#     f.write(card)
# print(card)
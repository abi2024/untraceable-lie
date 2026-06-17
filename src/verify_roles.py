"""
T3 CPU stub — verify role boundaries on REAL HP-KR examples WITHOUT loading the model.

Loads only the tokenizer (CPU, seconds). For a few examples it:
  - builds the full prompt token ids,
  - tags the four role spans structurally,
  - DECODES each span back to text so you can eyeball that the boundaries are right,
  - sanity-checks: assistant_answer span decodes to the "I do not know..." answer (HP-KR),
    and the assistant span never includes the [/INST] delimiter or the </s>.

This is the gate the user asked for: confirm the tagging is correct before spending GPU.
A5 note: this stub DECODES text to the console for human verification only; it does NOT
write any raw text to an artifact. The persisted token_roles.json (written by extract.py)
stores boundaries + example_index only.

Run:  python src/verify_roles.py            # checks lie + honest examples from both templates
      python src/verify_roles.py --n 6      # more examples
"""
from __future__ import annotations
import argparse, os, sys, json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from role_tagging import (  # noqa: E402
    load_tokenizer, build_prompt_ids, tag_roles, _delim_ids, ROLES, PROBE_ROLE,
)

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
PARQUET = os.path.join(ARTIFACTS, "hpkr_with_keys.parquet")


def _row_texts(row):
    msgs = row["messages"]
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    sys_t = next((m["content"] for m in msgs if m["role"] == "system"), "")
    usr_t = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst_t = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    return sys_t, usr_t, asst_t


def _truncate(s, n=80):
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + " …[+%d]" % (len(s) - n)


def verify(n_per_group=1):
    tok = load_tokenizer()
    delim = _delim_ids(tok)
    print(f"tokenizer loaded: {tok.__class__.__name__}")
    print(f"delimiter token-ids: " + ", ".join(f"{k}={v}" for k, v in delim.items()))
    print(f"eos_token_id={tok.eos_token_id}  bos_token_id={tok.bos_token_id}\n")

    df = pd.read_parquet(PARQUET).reset_index(drop=True)

    # pick lie + honest from each template so we cover all structural variants
    picks = []
    for tmpl in sorted(df.template_id.unique()):
        for lab in [True, False]:
            sub = df[(df.template_id == tmpl) & (df.deceptive == lab)]
            picks.extend(sub.head(n_per_group).index.tolist())

    all_ok = True
    for idx in picks:
        row = df.loc[idx]
        sys_t, usr_t, asst_t = _row_texts(row)
        ids = build_prompt_ids(tok, sys_t, usr_t, asst_t)
        try:
            spans = tag_roles(tok, ids, example_index=int(idx), delim_ids=delim)
        except AssertionError as e:
            print(f"FAIL ex {idx}: {e}")
            all_ok = False
            continue

        print(f"=== ex {idx}  template={row.template_id}  deceptive={row.deceptive}  "
              f"n_tokens={spans.n_tokens} ===")
        for role in ROLES:
            s, e = getattr(spans, role)
            rep = spans.rep[role]
            decoded = tok.decode(ids[s:e]) if e > s else "(empty)"
            star = "  <-- PROBE ROLE (A4)" if role == PROBE_ROLE else ""
            print(f"  {role:16s} [{s:4d}:{e:4d}] rep={rep:4d}  '{_truncate(decoded)}'{star}")

        # structural sanity checks on the assistant-answer (probe) span
        a_s, a_e = spans.assistant_answer
        ans_text = tok.decode(ids[a_s:a_e])
        checks = {
            "assistant span non-empty": a_e > a_s,
            "assistant span excludes [/INST]": "[/INST]" not in ans_text,
            "assistant span excludes </s>": "</s>" not in ans_text and (tok.eos_token or "") not in ans_text,
            "assistant span matches stored answer": asst_t.strip() in ans_text or ans_text.strip() in asst_t,
        }
        for name, passed in checks.items():
            print(f"      [{'ok' if passed else 'XX'}] {name}")
            all_ok = all_ok and passed
        print()

    print("ALL ROLE TAGS VERIFIED" if all_ok else "SOME CHECKS FAILED — do NOT run extraction")
    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1, help="examples per (template,label) group")
    args = ap.parse_args()
    ok = verify(args.n)
    sys.exit(0 if ok else 1)
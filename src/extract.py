"""
T3 — Extraction harness.  GPU-REQUIRED (the forward pass).

Captures role-representative residual-stream vectors:
    output shape [n_examples, n_layers_kept, n_roles(4), d_model]
stored to artifacts/acts_sweep.npz, plus artifacts/token_roles.json (boundaries + example_index
only, per Amendment A5 — NO raw text).

Roles are tagged structurally (src/role_tagging.py); roles are NOT derived from any probe.
The assistant_answer role is the pre-committed probe cell (A4); the other three are stored as
negative controls.

Guards enforced here:
  - no-circularity: roles structural (delegated to role_tagging).
  - determinism: a 10-example round-trip must produce byte-identical activations.
  - storage cap C1: total .npz <= 10 GB (asserted before final save).

Run on the H100:  python src/extract.py            # full transcript-blind core (1656 ex)
Smoke (50 ex):    python src/extract.py --n 50      # quick check before the full run
Set MISTRAL_2503_PATH to the local snapshot dir on the pod to avoid any download/auth.
"""
from __future__ import annotations
import argparse, json, os, sys, hashlib, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from role_tagging import (  # noqa: E402
    MODEL_ID, ROLES, PROBE_ROLE, load_tokenizer, build_prompt_ids, tag_roles, _delim_ids,
)

ARTIFACTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
# A6: the HP-KR headline arm extracts on the transcript-blind CORE (1656 rows), NOT the full
# parquet. The full parquet is preserved as the T7 transcript-only baseline arm's data source.
PARQUET = os.path.join(ARTIFACTS, "hpkr_blind_core.parquet")
C1_CAP_GB = 10.0

# ~10 candidate layers spread across depth. n_layers=40 (T0). Skip layer 0 (embeddings).
# Chosen at fixed fractional depths so the set is reproducible and not result-tuned.
def candidate_layers(n_layers: int, k: int = 10):
    # evenly spaced in (0, n_layers], inclusive of a late layer, exclusive of pure embedding
    fracs = np.linspace(0.15, 0.95, k)
    layers = sorted(set(int(round(f * n_layers)) for f in fracs))
    return layers


def _row_texts(row):
    """Extract (system, user, assistant) text from an HP-KR messages row.
    A5: these are read into memory transiently for tokenization ONLY; never written to artifacts."""
    msgs = row["messages"]
    if isinstance(msgs, str):
        msgs = json.loads(msgs)
    sys_t = next((m["content"] for m in msgs if m["role"] == "system"), "")
    usr_t = next((m["content"] for m in msgs if m["role"] == "user"), "")
    asst_t = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
    return sys_t, usr_t, asst_t


def load_model():
    """GPU. bf16, single H100.

    Mistral-Small-3.1-24B-Instruct-2503 is a MULTIMODAL checkpoint (`Mistral3ForConditionalGeneration`,
    model_type='mistral3'): a wrapper holding a text decoder (`.language_model`), a vision encoder,
    and a multimodal projector. `AutoModelForCausalLM` does NOT recognize it (KeyError 'mistral3' on
    <4.50; ValueError on the Mistral-3 branch). We load the full multimodal model via
    `AutoModelForImageTextToText`, then return BOTH the wrapper and its `.language_model` text tower.
    The probe reads the residual stream of the TEXT TOWER only — the model that authored the lies.
    Requires transformers >= 4.50 (mistral3 support landed in 4.50.0). torch pinned to 2.4.1+cu124.
    """
    import torch
    from transformers import AutoModelForImageTextToText
    wrapper = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda:0",
        attn_implementation="eager",  # deterministic; flash/sdpa can vary run-to-run
    )
    wrapper.eval()
    # The text decoder. Attribute path on Mistral3ForConditionalGeneration is `.language_model`.
    # Assert it exists so a future architecture change fails loud rather than silently hooking
    # the wrong module.
    assert hasattr(wrapper, "language_model"), (
        "expected `.language_model` on the multimodal wrapper; architecture may have changed — "
        f"available attrs: {[a for a in dir(wrapper) if 'model' in a.lower()]}"
    )
    text_model = wrapper.language_model
    return wrapper, text_model


def extract(df, text_model, tokenizer, layers, deterministic_check=False):
    """
    Returns acts [n, len(layers), 4, d] float32 and list[RoleSpans].
    `text_model` is the TEXT TOWER (wrapper.language_model), called directly so the forward pass
    is pure text — no vision encoder, no image inputs. Determinism: bf16 forward under eager
    attention with fixed input is deterministic on a fixed device; verified via round-trip.
    """
    import torch
    delim = _delim_ids(tokenizer)
    n = len(df)
    # d_model from the TEXT tower's config (the wrapper's top-level config lacks hidden_size —
    # it nests text params under a sub-config; the text_model.config has the real numbers).
    d_model = text_model.config.hidden_size
    acts = np.zeros((n, len(layers), 4, d_model), dtype=np.float32)
    spans_all = []

    t0 = time.time()
    # A6 traceability: example_index must be the ORIGINAL row id (_orig_row_index from the core
    # parquet), NOT the reset positional index, so token_roles.json rows can be joined back to
    # the full HP-KR set at T9 (behavioral scores, nuisance features). Fall back to positional
    # only if the column is absent (e.g. smoke runs on a parquet that lacks it).
    has_orig = "_orig_row_index" in df.columns
    for i in range(n):
        row = df.iloc[i]
        ex_idx = int(row["_orig_row_index"]) if has_orig else int(i)
        sys_t, usr_t, asst_t = _row_texts(row)
        ids = build_prompt_ids(tokenizer, sys_t, usr_t, asst_t)
        spans = tag_roles(tokenizer, ids, example_index=ex_idx, delim_ids=delim)
        spans_all.append(spans)

        input_ids = torch.tensor([ids], device=text_model.device)
        with torch.no_grad():
            out = text_model(input_ids, output_hidden_states=True, use_cache=False)
        # hidden_states: tuple len n_layers+1 (idx 0 = embeddings). each [1, seq, d].
        hs = out.hidden_states
        for li, layer in enumerate(layers):
            h = hs[layer][0]  # [seq, d]
            for ri, role in enumerate(ROLES):
                rep = spans.rep[role]
                if rep >= 0:
                    acts[i, li, ri, :] = h[rep].float().cpu().numpy()
                # else leave zeros (empty span, e.g. post_answer with no trailing tokens)

        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i+1}/{n}  ({rate:.1f} ex/s)")

    return acts, spans_all


def determinism_roundtrip(df, text_model, tokenizer, layers, n_check=10):
    """Run extraction twice on the same n_check examples; assert byte-identical."""
    sub = df.iloc[:n_check]
    a1, _ = extract(sub, text_model, tokenizer, layers)
    a2, _ = extract(sub, text_model, tokenizer, layers)
    identical = np.array_equal(a1, a2)
    max_abs = float(np.max(np.abs(a1 - a2))) if not identical else 0.0
    return identical, max_abs


def save(acts, spans_all, layers, path_npz, path_roles):
    # storage cap C1
    nbytes = acts.nbytes
    gb = nbytes / 1e9
    assert gb <= C1_CAP_GB, f"acts {gb:.2f} GB exceeds C1 cap {C1_CAP_GB} GB — reduce layers/examples"
    np.savez_compressed(
        path_npz,
        acts=acts,
        layers=np.array(layers),
        roles=np.array(ROLES),
        probe_role=PROBE_ROLE,
    )
    # token_roles.json: boundaries + example_index ONLY (A5)
    with open(path_roles, "w") as f:
        json.dump({
            "model": MODEL_ID,
            "layers": layers,
            "roles": ROLES,
            "probe_role": PROBE_ROLE,
            "shape": list(acts.shape),
            "spans": [s.to_json() for s in spans_all],
            "_safety": "A5: boundaries + example_index only; no raw text stored.",
        }, f, indent=0)
    return gb


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="limit examples (smoke). Default: full core.")
    ap.add_argument("--k_layers", type=int, default=10)
    args = ap.parse_args()

    print(f"MODEL_ID resolved to: {MODEL_ID}")
    if not os.path.isdir(MODEL_ID) and not MODEL_ID.startswith("mistralai/"):
        print("  WARNING: MODEL_ID is neither a local dir nor a mistralai/ repo id — check MISTRAL_2503_PATH")
    print(f"reading parquet: {PARQUET}")

    df = pd.read_parquet(PARQUET).reset_index(drop=True)
    if args.n:
        df = df.iloc[:args.n].copy()
        print(f"SMOKE run: limited to {len(df)} examples")
    print(f"extracting on {len(df)} examples"
          + ("" if "_orig_row_index" in df.columns
             else "  [WARN: no _orig_row_index column — token ids will be positional]"))

    tokenizer = load_tokenizer()
    wrapper, text_model = load_model()
    # n_layers from the TEXT tower (the wrapper's top-level config nests text params and lacks
    # num_hidden_layers at top level — see the `layers None` symptom on the multimodal config).
    n_layers = text_model.config.num_hidden_layers
    layers = candidate_layers(n_layers, args.k_layers)
    print(f"text-tower n_layers={n_layers}, hidden={text_model.config.hidden_size}, "
          f"candidate layers={layers}")

    # determinism FIRST (cheap, 10 ex) — abort before full run if it fails
    ok, max_abs = determinism_roundtrip(df, text_model, tokenizer, layers, n_check=min(10, len(df)))
    print(f"determinism round-trip: identical={ok} max_abs_diff={max_abs}")
    assert ok, "DETERMINISM FAILED — activations differ run-to-run; fix before extracting"

    acts, spans = extract(df, text_model, tokenizer, layers)
    gb = save(acts, spans, layers,
              os.path.join(ARTIFACTS, "acts_sweep.npz"),
              os.path.join(ARTIFACTS, "token_roles.json"))
    print(f"saved acts_sweep.npz shape={acts.shape} ({gb:.2f} GB), token_roles.json")
    print(f"DONE — probe role = {PROBE_ROLE} (A4); other 3 roles stored as negative controls")


if __name__ == "__main__":
    main()
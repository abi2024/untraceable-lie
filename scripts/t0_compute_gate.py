"""
T0 — Compute micro-benchmark & go/no-go gate.
Proves model-fit (C2), storage (C1), and runtime/cost (C3) on 50 examples
before any real spend. Records true d_model/n_layers. Writes artifacts/compute_gate.md.
DOES NOT compute any research result (no AUROC, no labels). Infrastructure only.
"""
import os, time, json, glob
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- config you can edit ----
MODEL_ID   = "mistralai/Mistral-Small-24B-Instruct-2501"
N_EXAMPLES = 50
N_CAND_LAYERS = 10          # candidate layers we'll store at (plan: ~10)
N_ROLES    = 4              # system / user-q / assistant-ans / post-answer
FULL_N     = 5661          # full HP-KR example count (from the paper)
DOLLARS_PER_HR = 6.0       # use your actual on-demand H100 rate; spot is lower
SEEDS_PLANNED  = 5         # for the GPU-hr extrapolation of probe phases
import pathlib
ARTIFACTS = str(pathlib.Path(__file__).resolve().parent.parent / "artifacts")
# -----------------------------

os.makedirs(ARTIFACTS, exist_ok=True)
torch.manual_seed(0)
dev = "cuda"

print("=== Loading model (bf16) ===")
t_load = time.time()
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16, device_map={"": 0}
)
model.eval()
load_s = time.time() - t_load

cfg = model.config
d_model  = cfg.hidden_size
n_layers = cfg.num_hidden_layers
print(f"  loaded in {load_s:.0f}s | d_model={d_model} n_layers={n_layers}")

# C2: peak VRAM after load + a forward pass
torch.cuda.reset_peak_memory_stats()

# Build 50 dummy chat-formatted prompts (T0 doesn't need real labels;
# we just need realistic token lengths through the chat template).
sample_texts = [
    "You are a helpful assistant. The user asks about a sensitive internal matter. "
    "Question: Did you have access to the document in question? Answer honestly."
] * N_EXAMPLES

# Show the chat template structure ONCE so you can see role boundaries (informs T3).
demo = tok.apply_chat_template(
    [{"role": "system", "content": "SYS"},
     {"role": "user", "content": "USERQ"},
     {"role": "assistant", "content": "ASSISTANTANS"}],
    tokenize=False,
)
print("\n=== Chat template structure (study this for T3 role-tagging) ===")
print(demo)
print("=== end template ===\n")

# Pick candidate layers evenly across depth
cand_layers = sorted(set(np.linspace(0, n_layers - 1, N_CAND_LAYERS).astype(int).tolist()))
print(f"candidate layers: {cand_layers}")

# Capture residual stream via output_hidden_states (simple + deterministic for T0).
print("=== Extracting on 50 examples ===")
tok_count = 0
t_ext = time.time()
role_rep_store = []  # [n, n_cand_layers, n_roles, d]  (T0 fills roles with a placeholder mean)

with torch.no_grad():
    for txt in sample_texts:
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=512).to(dev)
        out = model(**ids, output_hidden_states=True)
        hs = out.hidden_states  # tuple len n_layers+1, each [1, seq, d]
        tok_count += ids["input_ids"].shape[1]
        # T0 placeholder: store mean over all tokens as 1 "role" replicated to N_ROLES.
        # (Real role-tagging is T3; here we only need realistic STORAGE size.)
        per_layer = []
        for L in cand_layers:
            v = hs[L + 1][0].mean(dim=0)              # [d]  mean pool (placeholder)
            per_layer.append(v.unsqueeze(0).repeat(N_ROLES, 1))  # [roles, d]
        role_rep_store.append(torch.stack(per_layer))  # [layers, roles, d]

ext_s = time.time() - t_ext
arr = torch.stack(role_rep_store).to(torch.float16).cpu().numpy()  # [n, layers, roles, d]
np.savez_compressed(os.path.join(ARTIFACTS, "t0_acts_50.npz"), acts=arr)

peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
bytes_50 = os.path.getsize(os.path.join(ARTIFACTS, "t0_acts_50.npz"))
gb_50 = bytes_50 / 1e9
gb_full = gb_50 * (FULL_N / N_EXAMPLES)

tok_per_s = tok_count / ext_s
# Extrapolate GPU-hours: extraction is ~1 forward pass over full set.
ext_hr_full = (FULL_N * (tok_count / N_EXAMPLES)) / tok_per_s / 3600
# Rough total: extraction + behavioral gen (~3x extraction tokens, generation is slower)
# + baseline pass. Apply the plan's ×4 debug buffer.
raw_total_hr = ext_hr_full * (1 + 3 + 1)        # extraction + behavioral + baseline-ish
total_hr_x4  = raw_total_hr * 4
total_cost   = total_hr_x4 * DOLLARS_PER_HR

# Gate verdicts
C1 = gb_full <= 10.0
C2 = peak_vram_gb <= 80.0
C3 = (total_hr_x4 <= 40.0) and (total_cost <= 300.0)

report = f"""# compute_gate.md — T0 results

## Model facts (measured)
- MODEL_ID: {MODEL_ID}
- d_model (hidden_size): {d_model}
- n_layers (num_hidden_layers): {n_layers}
- load time: {load_s:.0f}s

## C2 — model + buffers fit (≤ 80 GB)
- peak VRAM after load + forward: **{peak_vram_gb:.1f} GB**
- verdict: {"PASS" if C2 else "FAIL"}

## C1 — storage (≤ 10 GB)
- candidate layers stored: {len(cand_layers)} -> {cand_layers}
- roles per layer: {N_ROLES}
- bytes for 50 ex: {gb_50*1000:.1f} MB
- extrapolated full ({FULL_N} ex): **{gb_full:.2f} GB**
- verdict: {"PASS" if C1 else "FAIL"}

## C3 — runtime/cost (≤ 40 GPU-hr / ≤ $300)
- throughput: {tok_per_s:.0f} tok/s
- extraction (full, 1 pass): {ext_hr_full:.2f} GPU-hr
- raw total estimate (extraction+behavioral+baseline): {raw_total_hr:.2f} GPU-hr
- with ×4 debug buffer: **{total_hr_x4:.2f} GPU-hr**
- at ${DOLLARS_PER_HR}/hr: **${total_cost:.0f}**
- verdict: {"PASS" if C3 else "FAIL"}

## GATE 1
- C1={"PASS" if C1 else "FAIL"} C2={"PASS" if C2 else "FAIL"} C3={"PASS" if C3 else "FAIL"}
- **GO** {"✅ proceed to T1" if (C1 and C2 and C3) else "❌ DO NOT PROCEED — shrink plan and re-run T0"}

## Notes for T3 (role-tagging)
- Study the chat-template dump printed during this run; system/user/assistant
  boundaries there define the structural role assignment T3 must implement.
- T0 used a placeholder mean-pool for activations purely to size storage; it is
  NOT the real role extraction.
"""
with open(os.path.join(ARTIFACTS, "compute_gate.md"), "w") as f:
    f.write(report)
print(report)
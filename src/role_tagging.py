"""
T3 — Structural token-role tagging for Mistral-Small-24B-Instruct-2503.

The four roles are assigned STRUCTURALLY from the chat-template delimiters, never from a
probe and never heuristically (no-circularity guard, PRD 01 T3 / Amendment A4).

Mistral 2503 template (confirmed T0):
    <s>[SYSTEM_PROMPT] <system text> [/SYSTEM_PROMPT][INST] <user text> [/INST] <assistant text></s>

Role spans (token index ranges, half-open [start, end)):
    system        : tokens between [SYSTEM_PROMPT] and [/SYSTEM_PROMPT]
    user_question : tokens between [INST] and [/INST]
    assistant_ans : tokens after [/INST] up to (but not including) </s>   <- THE PROBE ROLE (A4)
    post_answer   : the </s> token (and anything after) -- usually 1 token; negative control

Each role's REPRESENTATIVE vector (stored downstream) is the LAST token of its span
(the standard residual-stream readout position; for assistant_ans this is the last answer
token, where the model's committed state is most concentrated).

This module loads ONLY the tokenizer (CPU, fast). It is import-safe with no GPU.
"""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict

import os
MODEL_ID = os.environ.get(
    "MISTRAL_2503_PATH",
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
)
# Delimiter literals for the 2503 template. We resolve them to token-id sequences at runtime
# (a single special token may or may not be one id depending on tokenizer config), so we never
# hard-code ids.
DELIMS = {
    "sys_open":  "[SYSTEM_PROMPT]",
    "sys_close": "[/SYSTEM_PROMPT]",
    "inst_open": "[INST]",
    "inst_close":"[/INST]",
}

ROLES = ["system", "user_question", "assistant_answer", "post_answer"]
PROBE_ROLE = "assistant_answer"  # pre-committed per Amendment A4


@dataclass
class RoleSpans:
    example_index: int
    n_tokens: int
    system: tuple[int, int]
    user_question: tuple[int, int]
    assistant_answer: tuple[int, int]
    post_answer: tuple[int, int]
    # representative token index per role (last token of span); -1 if span empty
    rep: dict

    def to_json(self):
        d = asdict(self)
        # NOTE (Amendment A5): we store ONLY boundaries + example_index, never raw text.
        return d


def load_tokenizer():
    """CPU-only. fix_mistral_regex=True per Amendment A1."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID, fix_mistral_regex=True)


def build_prompt_ids(tokenizer, system_text: str, user_text: str, assistant_text: str):
    """
    Build the full token-id sequence for one example by MANUALLY constructing the Mistral 2503
    template. The 2503 tokenizer ships no embedded chat_template (Mistral distributes formatting
    via mistral-common, not a Jinja template in tokenizer_config), so apply_chat_template fails.
    Building it manually from the resolved special-token ids is more deterministic anyway:
    we control the exact byte layout and don't depend on a template that could vary across
    transformers versions.

    Layout (2503):
        <s>[SYSTEM_PROMPT] {system} [/SYSTEM_PROMPT][INST] {user} [/INST] {assistant}</s>

    We assemble at the TOKEN-ID level: special tokens as their resolved ids, free text via
    encode(..., add_special_tokens=False). This guarantees the delimiter ids that role_tagging
    searches for are exactly the ids present in the stream.

    Returns: list[int] token ids.
    """
    d = _delim_ids(tokenizer)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    ids = []
    if bos is not None:
        ids.append(bos)
    ids += d["sys_open"] + enc(" " + system_text + " ") + d["sys_close"]
    ids += d["inst_open"] + enc(" " + user_text + " ") + d["inst_close"]
    ids += enc(" " + assistant_text)
    if eos is not None:
        ids.append(eos)
    return ids


def build_elicitation_ids(tokenizer, system_text, user_text, assistant_text, followup_text):
    """
    Build the token-id sequence for a UQC / Self-Eval elicitation:
        <s>[SYSTEM_PROMPT] {system} [/SYSTEM_PROMPT][INST] {user} [/INST] {assistant}</s>[INST] {followup} [/INST]
    i.e. the EXISTING labeled answer is kept verbatim (pinned to the same object the probe read
    and the label refers to), and the elicitation question is appended as a NEW user turn (in
    chat-template distribution, matching Pacchiardi's follow-up-in-conversation method).

    The sequence ENDS right after the followup's [/INST] — the next generated token is the
    model's first response token, where the yes/no (or TRUTHFUL/DECEPTIVE etc.) logprob is read.
    No trailing EOS (we want the model to answer, not to see a closed turn).

    Returns: list[int] token ids (the prefix; the scored token is the next-token prediction).
    """
    d = _delim_ids(tokenizer)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    ids = []
    if bos is not None:
        ids.append(bos)
    ids += d["sys_open"] + enc(" " + system_text + " ") + d["sys_close"]
    ids += d["inst_open"] + enc(" " + user_text + " ") + d["inst_close"]
    ids += enc(" " + assistant_text)
    if eos is not None:
        ids.append(eos)          # close the assistant's (deceptive) turn
    ids += d["inst_open"] + enc(" " + followup_text + " ") + d["inst_close"]
    return ids


def build_prompt_prefix_and_followups(tokenizer, system_text, user_text, assistant_text,
                                      followup_texts):
    """
    Efficiency helper: the prefix through the assistant answer + EOS is identical across all
    elicitation questions for one example. Returns (prefix_ids, [followup_ids_per_question]) so a
    caller can reuse the prefix. Each full sequence = prefix_ids + inst_open + enc(followup) + inst_close.
    """
    d = _delim_ids(tokenizer)
    bos = tokenizer.bos_token_id
    eos = tokenizer.eos_token_id

    def enc(text):
        return tokenizer.encode(text, add_special_tokens=False)

    prefix = []
    if bos is not None:
        prefix.append(bos)
    prefix += d["sys_open"] + enc(" " + system_text + " ") + d["sys_close"]
    prefix += d["inst_open"] + enc(" " + user_text + " ") + d["inst_close"]
    prefix += enc(" " + assistant_text)
    if eos is not None:
        prefix.append(eos)

    followups = []
    for ft in followup_texts:
        followups.append(d["inst_open"] + enc(" " + ft + " ") + d["inst_close"])
    return prefix, followups


def _find_subseq(haystack: list[int], needle: list[int], start: int = 0) -> int:
    """Return index where `needle` begins in `haystack` at/after `start`, else -1."""
    if not needle:
        return -1
    n, m = len(haystack), len(needle)
    for i in range(start, n - m + 1):
        if haystack[i:i + m] == needle:
            return i
    return -1


def _delim_ids(tokenizer):
    """Resolve each delimiter literal to its token-id sequence (no hard-coded ids)."""
    out = {}
    for name, lit in DELIMS.items():
        # encode without adding BOS/EOS so we get just the delimiter's own tokens
        ids = tokenizer.encode(lit, add_special_tokens=False)
        out[name] = ids
    return out


def tag_roles(tokenizer, full_ids: list[int], example_index: int,
              delim_ids: dict | None = None) -> RoleSpans:
    """
    Assign the four role spans by locating the template delimiters in the token stream.
    Raises AssertionError if the structure is not found as expected (fail loud — a silent
    mis-tag would corrupt every probe).
    """
    if delim_ids is None:
        delim_ids = _delim_ids(tokenizer)

    eos_id = tokenizer.eos_token_id

    so = _find_subseq(full_ids, delim_ids["sys_open"])
    sc = _find_subseq(full_ids, delim_ids["sys_close"], start=so + 1 if so >= 0 else 0)
    io = _find_subseq(full_ids, delim_ids["inst_open"], start=sc + 1 if sc >= 0 else 0)
    ic = _find_subseq(full_ids, delim_ids["inst_close"], start=io + 1 if io >= 0 else 0)

    assert so >= 0, f"[ex {example_index}] [SYSTEM_PROMPT] not found"
    assert sc > so, f"[ex {example_index}] [/SYSTEM_PROMPT] not found after open"
    assert io > sc, f"[ex {example_index}] [INST] not found after system close"
    assert ic > io, f"[ex {example_index}] [/INST] not found after [INST]"

    len_sc = len(delim_ids["sys_close"])
    len_io = len(delim_ids["inst_open"])
    len_ic = len(delim_ids["inst_close"])

    # span boundaries (half-open). content is BETWEEN delimiters.
    system_span = (so + len(delim_ids["sys_open"]), sc)
    user_span = (io + len_io, ic)

    # assistant answer = everything after [/INST] up to EOS (</s>)
    ans_start = ic + len_ic
    # locate EOS at/after ans_start
    eos_pos = -1
    for i in range(ans_start, len(full_ids)):
        if full_ids[i] == eos_id:
            eos_pos = i
            break
    if eos_pos < 0:
        eos_pos = len(full_ids)  # no explicit EOS; answer runs to end
    assistant_span = (ans_start, eos_pos)
    post_span = (eos_pos, len(full_ids))

    assert assistant_span[1] > assistant_span[0], (
        f"[ex {example_index}] empty assistant-answer span — the probe role would be empty"
    )

    def rep_idx(span):
        return span[1] - 1 if span[1] > span[0] else -1

    return RoleSpans(
        example_index=example_index,
        n_tokens=len(full_ids),
        system=system_span,
        user_question=user_span,
        assistant_answer=assistant_span,
        post_answer=post_span,
        rep={
            "system": rep_idx(system_span),
            "user_question": rep_idx(user_span),
            "assistant_answer": rep_idx(assistant_span),
            "post_answer": rep_idx(post_span),
        },
    )
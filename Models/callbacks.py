"""
Callbacks for evaluation and training.
"""

import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Literal, Optional

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, TrainerCallback

from Assessors.core.prompt_prep import assemble_chat_messages
from Assessors.countermodel.prompts.system import system_prompt as countermodel_system_prompt
from Assessors.symbolization.prompts.system import system_prompt as symbolization_system_prompt
from Assessors.validity.prompts.system import system_prompt as validity_system_prompt

_ASCII_TO_UNICODE_RE = [
    (re.compile(r"\bforall\b", re.I), "∀"),
    (re.compile(r"\bexists\b", re.I), "∃"),
    (re.compile(r"\biff\b", re.I), "↔"),
    (re.compile(r"<=>"), "↔"),
    (re.compile(r"<->"), "↔"),
    (re.compile(r"->"), "→"),
    (re.compile(r"~"), "¬"),
    (re.compile(r"!"), "¬"),
    (re.compile(r"\bor\b", re.I), "∨"),
    (re.compile(r"\|"), "∨"),
    (re.compile(r"&"), "∧"),
]


def nfkc_strip(s: str) -> str:
    """
    Remove leading and trailing whitespace from a string and then strip it of
    Unicode compatibility characters.

    Args:
        s (str): The string to strip.

    Returns:
        str: The formatted string.
    """
    return unicodedata.normalize("NFKC", s or "").strip()


def strip_all_ws(s: str) -> str:
    """
    Remove all whitespace from a string.

    Args:
        s (str): The string to strip.

    Returns:
        str: The string with all whitespace removed.
    """
    return re.sub(r"\s+", "", s or "", flags=re.MULTILINE).strip()


def _normalize_symbolization(s: str) -> str:
    """
    Normalize the symbolization answer string: strip, unify ASCII tokens to Unicode, and remove all whitespace.
    This gives a canonical representation for comparison.

    Args:
        s (str): The string to normalize.

    Returns:
        str: The normalized string.
    """
    s = nfkc_strip(s)
    s = s.strip("`'\"“”")  # drop surrounding quotes/backticks/markdown

    for rx, rep in _ASCII_TO_UNICODE_RE:
        s = rx.sub(rep, s)
    s = strip_all_ws(s)
    return s


def _normalize_validity(s: str) -> str:
    """
    Normalize a validity answer string: strip, extract first integer and return it canonically as digits.
    If no integer is found, return the original string.

    Args:
        s (str): The string to normalize.

    Returns:
        str: The normalized string.
    """
    s = nfkc_strip(s)
    m = re.search(r"\b\d+\b", s)
    return m.group(0) if m else s


def _normalize_countermodel_block(s: str) -> str:
    """
    Normalize the structured block: trim, unify newlines, trim trailing ws per line, keep content.
    We DO NOT remove all whitespace because formatting conveys structure,
    but we ignore inconsequential trailing spaces and blank line noise.

    Args:
        s (str): The string to normalize.

    Returns:
        str: The normalized string.
    """
    s = nfkc_strip(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in s.split("\n")]

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def _extract_validity(text: str) -> str:
    """
    Extract the chosen option number (e.g., "4") from free-form rationale.
    Simple strategy is to use the last valid integer (not highly precisely, but seems to work okay).

    Args:
        text (str): The text from which to extract a validity answer.

    Returns:
        str: The extracted validity answer.
    """
    t = text or ""
    ms = list(re.finditer(r"\b([1-9]|10)\b", t))
    return ms[-1].group(1) if ms else nfkc_strip(t)


def _extract_symbolization(text: str) -> str:
    """
    Extract a symbolization answer from a given text.

    This function looks for a code block content (fenced by triple backticks) containing a formula.
    If no fenced block is found, the function looks for an inline code block (enclosed by single backticks).
    If no inline block is found, the function looks for the longest parenthesized/logic-ish span.
    If no such span is found, the function returns the last non-empty line.

    Args:
        text (str): The text from which to extract a symbolization answer.

    Returns:
        str: The extracted symbolization answer.
    """
    m = re.search(r"```(?:[^\n`]*\n)?([\s\S]*?)```", text)
    if m:
        block = m.group(1).strip()
        cand_lines = [ln.strip() for ln in block.splitlines() if any(ch in ln for ch in "∀∃¬∧∨→↔()")]
        if cand_lines:
            return cand_lines[0] if len(cand_lines) == 1 else max(cand_lines, key=len)

    m = re.search(r"`([^`]+)`", text)
    if m:
        return m.group(1).strip()

    # Heuristic: pick the longest parenthesized/logic-ish span (covers cases like: (¬∃x(Mx∧Qxb)→∀x(Nx→Pxc)))
    m_all = list(re.finditer(r"[^\s]{0,10}\(.*?\)[^\s]{0,10}", text))
    if m_all:
        spans = [mm.group(0) for mm in m_all if any(sym in mm.group(0) for sym in "∀∃¬∧∨→↔")]
        if spans:
            return max(spans, key=len)

    # Fallback: last non-empty line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else nfkc_strip(text)


def _extract_countermodel(text: str) -> str:
    """
    Extract a countermodel answer from a given text.

    This function looks for a label in the form of "Domain:" possibly inside a code fence.
    If no label is found, the function then looks for a block from "Domain:" to the end of the Binary predicates section (if present).
    If no such block is found, the function returns everything.

    Args:
        text (str): The text from which to extract a countermodel answer.

    Returns:
        str: The extracted countermodel answer.
    """
    m = re.search(r"(?is)(?:```[^\n`]*\n)?(Domain:.*?)(?:```|\Z)", text)
    if m:
        return m.group(1).strip()

    # Otherwise, capture from "Domain:" to the end of the Binary predicates section (if present)
    m = re.search(r"(?is)(Domain:.*?)(?:\n\n|\Z)", text)
    if m:
        return m.group(1).strip()

    # Fallback: everything (better to be noisy than miss the answer)
    return nfkc_strip(text)


def extract_answer_for_type(qtype: Literal["validity", "symbolization", "countermodel"], text: str) -> str:
    """
    Extract an answer from a given text based on the question type.

    Args:
        qtype (Literal["validity", "symbolization", "countermodel"]): The type of the question.
        text (str): The text from which to extract an answer.

    Returns:
        str: The extracted answer.
    """
    if qtype == "validity":
        return _extract_validity(text)
    if qtype == "symbolization":
        return _extract_symbolization(text)
    if qtype == "countermodel":
        return _extract_countermodel(text)
    return nfkc_strip(text)


def normalize_for_type(qtype: Literal["validity", "symbolization", "countermodel"], s: str) -> str:
    """
    Normalize a string based on the question type.

    The normalization process is different for each question type:
      - validity: word boundary integer search,
      - symbolization: normalized (ASCII→Unicode, no ws) substring search,
      - countermodel: look for the normalized block inside a lightly-normalized prediction.

    Args:
        qtype (Literal["validity", "symbolization", "countermodel"]): The type of the question.
        s (str): The string to normalize.

    Returns:
        str: The normalized string.
    """
    if qtype == "validity":
        return _normalize_validity(s)
    if qtype == "symbolization":
        return _normalize_symbolization(s)
    if qtype == "countermodel":
        return _normalize_countermodel_block(s)
    return nfkc_strip(s)


def contains_answer_for_type(
    qtype: Literal["validity", "symbolization", "countermodel"], pred_text: str, gold_text: str
) -> bool:
    """
    Check if a prediction text contains an answer for a given question type.

    Args:
        qtype (Literal["validity", "symbolization", "countermodel"]): The type of the question.
        pred_text (str): The predicted answer.
        gold_text (str): The gold answer.

    Returns:
        bool: True if the prediction text contains an answer for the given question type, False otherwise.
    """
    if not gold_text:
        return False

    if qtype == "validity":
        g = _normalize_validity(gold_text)
        if not g:
            return False
        return re.search(rf"(?<!\d){re.escape(g)}(?!\d)", pred_text) is not None

    if qtype == "symbolization":
        g = _normalize_symbolization(gold_text)
        p = _normalize_symbolization(pred_text)
        return g in p

    if qtype == "countermodel":
        g = _normalize_countermodel_block(gold_text)
        p = _normalize_countermodel_block(pred_text)
        p_compact = re.sub(r"\n{2,}", "\n\n", p)
        g_compact = re.sub(r"\n{2,}", "\n\n", g)
        return g_compact in p_compact

    # default fallback
    return strip_all_ws(gold_text).lower() in strip_all_ws(pred_text).lower()


class BatchedGenerationEvalCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        eval_rows: List[Dict],
        question_type: Literal["validity", "symbolization", "countermodel"] = "validity",
        accepts_system_prompt: bool = True,
        batch_size: int = 8,
        max_new_tokens: int = 128,
        sample_cap: int = 100,
        add_generation_prompt: bool = True,
        eval_name: str = "eval",
        save_dir: Optional[str] = None,
    ) -> None:
        self.tok = tokenizer
        self.rows = eval_rows
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.sample_cap = sample_cap
        self.accepts_system_prompt = accepts_system_prompt
        self.system_prompt = self.load_system_prompt(question_type)
        self.add_generation_prompt = add_generation_prompt
        self.question_type = question_type
        self.eval_name = eval_name
        self.save_dir = save_dir
        self._orig_padding_side = tokenizer.padding_side or "right"

    def load_system_prompt(self, question_type: Literal["validity", "symbolization", "countermodel"]) -> str:
        """
        Loads the system prompt string for the given question type.

        Returns:
            A string containing the system prompt.

        Raises:
            ValueError: If the question type is unknown.
        """
        if question_type == "validity":
            return validity_system_prompt
        elif question_type == "symbolization":
            return symbolization_system_prompt
        elif question_type == "countermodel":
            return countermodel_system_prompt
        else:
            raise ValueError(
                f"Unknown question type: {question_type}. Available: ['validity', 'symbolization', 'countermodel']"
            )

    @torch.no_grad()
    def on_evaluate(self, args, state, control, **kwargs) -> None:
        model = kwargs["model"]

        # LEFT PAD for decoder-only gen
        self.tok.padding_side = "left"

        # Make generate fast/clean
        prev_use_cache = getattr(model.config, "use_cache", True)
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True

        # DDP gating
        is_world_zero = getattr(state, "is_world_process_zero", True)

        try:
            model.eval()

            n = min(len(self.rows), self.sample_cap)
            rows = self.rows[:n]

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            em_full = rem_full = em_extracted = contains_re = 0
            last_pred = last_gold = last_extracted = ""

            # Build prompts
            prompts, golds, ids = [], [], []
            for r in rows:
                messages = assemble_chat_messages(
                    prompt=r["task"],
                    accepts_system_prompt=self.accepts_system_prompt,
                    system_prompt=self.system_prompt,
                )
                prompt = self.tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=self.add_generation_prompt
                )
                prompts.append(prompt)
                golds.append(nfkc_strip(r["answer"]))
                ids.append(r["id"])

            # tqdm only on rank-0
            iterator = range(0, n, self.batch_size)
            if is_world_zero:
                iterator = tqdm(iterator, desc="BatchedGenerationEval")

            predictions: List[Dict[str, Any]] = []
            for start in iterator:
                end = min(start + self.batch_size, n)
                batch_prompts = prompts[start:end]
                batch_golds = golds[start:end]
                batch_ids = ids[start:end]

                enc = self.tok(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=getattr(self.tok, "model_max_length", 4096),
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                input_seq_len = enc["input_ids"].shape[1]

                gen_ids = model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    eos_token_id=self.tok.eos_token_id,
                    pad_token_id=self.tok.pad_token_id,
                )

                for i in range(gen_ids.size(0)):
                    gen_only = gen_ids[i, input_seq_len:]
                    decoded = self.tok.decode(gen_only, skip_special_tokens=True)
                    pred_full = nfkc_strip(decoded)
                    gold = batch_golds[i]
                    last_pred, last_gold = pred_full, gold

                    if pred_full == gold:
                        em_full += 1
                    if strip_all_ws(pred_full) == strip_all_ws(gold):
                        rem_full += 1

                    extracted = extract_answer_for_type(self.question_type, pred_full)
                    last_extracted = extracted
                    if (
                        normalize_for_type(self.question_type, extracted)
                        == normalize_for_type(self.question_type, gold)
                        and gold != ""
                    ):
                        em_extracted += 1

                    if contains_answer_for_type(self.question_type, pred_full, gold):
                        contains_re += 1

                    predictions.append(
                        {
                            "id": batch_ids[i],
                            "prediction": decoded,
                            "extracted": extracted,
                            "answer": gold,
                        }
                    )

            metrics = {
                f"{self.eval_name}/gen_em_full": em_full / n if n else 0.0,
                f"{self.eval_name}/gen_relaxed_em_full": rem_full / n if n else 0.0,
                f"{self.eval_name}/gen_em_extracted": em_extracted / n if n else 0.0,
                f"{self.eval_name}/gen_contains_regex": contains_re / n if n else 0.0,
            }

            # W&B + HF logs only on rank-0
            if is_world_zero:
                try:
                    import wandb

                    wandb.log(metrics, step=state.global_step)
                except Exception:
                    pass

                if "metrics" in kwargs and isinstance(kwargs["metrics"], dict):
                    kwargs["metrics"].update(metrics)

                # Stash latest metrics for post-hoc retrieval
                prev = getattr(model.config, "gen_eval_last", {})
                prev.update(metrics)
                setattr(model.config, "gen_eval_last", prev)

                EM_full = metrics[f"{self.eval_name}/gen_em_full"]
                rEM_full = metrics[f"{self.eval_name}/gen_relaxed_em_full"]
                EM_extracted = metrics[f"{self.eval_name}/gen_em_extracted"]
                contains_regex = metrics[f"{self.eval_name}/gen_contains_regex"]

                print(
                    f"\n\n[{self.eval_name}] step={state.global_step} "
                    f"EM(full)={EM_full:.4f} "
                    f"rEM(full)={rEM_full:.4f} "
                    f"EM(extracted)={EM_extracted:.4f} "
                    f"contains_regex={contains_regex:.4f}\n"
                    f"pred(sample): {last_pred}\n"
                    f"extracted(sample): {last_extracted}\n"
                    f"gold(sample): {last_gold}"
                )

                # Save to dir
                if self.save_dir:
                    os.makedirs(self.save_dir, exist_ok=True)
                    out_path = os.path.join(self.save_dir, f"{self.question_type}_{state.global_step}.jsonl")
                    with open(out_path, "w", encoding="utf-8") as f:
                        for rec in predictions:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        finally:
            # Restore config + tokenizer padding side
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = prev_use_cache
            self.tok.padding_side = self._orig_padding_side

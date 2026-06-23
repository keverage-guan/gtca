"""
Utilities for building prompts and normalizing examples for GTCA training/evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import re


ANSWER_PREFIX = "Answer:"


def normalize_whitespace(text: str) -> str:
    """Collapse consecutive whitespace into a single space, and strip ends."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def label_index_to_letter(idx: int) -> str:
    if idx == 0:
        return "A"
    if idx == 1:
        return "B"
    if idx == 2:
        return "C"
    if idx == 3:
        return "D"
    raise ValueError(f"Invalid option index: {idx}")


def letter_to_label_index(letter: str) -> int:
    letter = str(letter).strip().upper()
    if letter == "A":
        return 0
    if letter == "B":
        return 1
    if letter == "C":
        return 2
    if letter == "D":
        return 3
    raise ValueError(f"Invalid option letter: {letter}")


def get_turn_end_token(tokenizer: Any) -> str:
    """
    Try to use an end-of-turn token when available; otherwise fall back to eos_token.
    This is robust across Qwen/Llama-style chat tokenizers.
    """
    for attr in ("eot_token", "end_of_turn_token", "chat_eos_token"):
        tok = getattr(tokenizer, attr, None)
        if isinstance(tok, str) and tok:
            return tok
    tok = getattr(tokenizer, "eos_token", None)
    if isinstance(tok, str) and tok:
        return tok
    return ""


def get_system_prompt(task: str) -> str:
    task = (task or "").lower()
    if task in {"cola"}:
        return (
            "You are a linguist. Decide if the following English sentence is grammatically acceptable. "
            "Output 1 for acceptable, 0 for unacceptable. Output only a single character: 0 or 1."
        )
    return (
        "Choose the correct option based on the question. "
        "Output only the corresponding letter of the option without providing a reason."
    )


@dataclass(frozen=True)
class MCQAUserContent:
    user_content: str
    question_span: Tuple[int, int]  # [start, end) char span inside user_content
    option_spans: Dict[str, Tuple[int, int]]  # label -> [start, end) char span inside user_content
    options_header_span: Tuple[int, int]


def format_mcqa_user_content(task: str, question_or_context: str, options: Sequence[str]) -> MCQAUserContent:
    """
    Build a standardized MCQA user content block and record spans for:
    - the question/context text region (excluding label prefixes like "Question:" or "Context:" lines)
    - each option line region
    - the options header region

    Prompt formats follow Appendix B of the GTCA paper:
      - CLOTH/MMLU (0-shot): "Question: {Question}\nOptions:\nA. ...\n...\n"
      - HellaSwag (10-shot): "Context: Read the context ...\n{context}\nOptions:\nA. ...\n...\n"
      - Winogrande (5-shot): "Question: {Question}\nOptions:\nA. ...\nB. ...\n"
    """
    task = (task or "").lower()
    q = normalize_whitespace(question_or_context)

    labels = ["A", "B", "C", "D"]
    if len(options) == 2:
        labels = ["A", "B"]
    if len(options) not in (2, 4):
        raise ValueError(f"MCQA options must have length 2 or 4, got {len(options)}")

    # Header block
    if task == "hellaswag":
        header_prefix = "Context: Read the context and choose the most plausible continuation.\n"
        header_text = header_prefix + q
        # For parsing, we treat only the context text (q) as the question span.
        question_start = len(header_prefix)
        question_end = question_start + len(q)
        question_block = header_text
    else:
        prefix = "Question: "
        question_block = prefix + q
        question_start = len(prefix)
        question_end = question_start + len(q)

    options_header = "\nOptions:\n"

    option_spans: Dict[str, Tuple[int, int]] = {}

    buf: List[str] = []
    buf.append(question_block)
    buf.append(options_header)
    options_header_start = len(question_block)
    options_header_end = options_header_start + len(options_header)

    cursor = len("".join(buf))
    for lab, opt in zip(labels, options):
        opt_norm = normalize_whitespace(opt)
        line = f"{lab}. {opt_norm}\n"
        start = cursor
        end = cursor + len(line)
        option_spans[lab] = (start, end)
        buf.append(line)
        cursor = end

    user_content = "".join(buf).rstrip("\n")
    # Adjust last option span if we stripped one trailing newline
    last_lab = labels[len(options) - 1]
    s, e = option_spans[last_lab]
    if user_content and not user_content.endswith("\n") and e == cursor:
        option_spans[last_lab] = (s, e - 1)

    return MCQAUserContent(
        user_content=user_content,
        question_span=(question_start, question_end),
        option_spans=option_spans,
        options_header_span=(options_header_start, options_header_end),
    )


def build_chat_prompt_text(tokenizer: Any, system_prompt: str, user_content: str) -> str:
    """
    Build the full chat prompt text (including model-specific special tokens) using the tokenizer chat template,
    ending at an assistant generation point, then append the answer prefix.
    """
    if not hasattr(tokenizer, "apply_chat_template") or tokenizer.apply_chat_template is None:
        raise ValueError("Tokenizer does not support apply_chat_template(); cannot build chat prompts safely.")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    base = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if not base.endswith("\n"):
        base += "\n"
    return base + ANSWER_PREFIX


def char_span_to_token_indices(
    offsets: Sequence[Tuple[int, int]],
    span_start: int,
    span_end: int,
) -> List[int]:
    """Map a character span [start, end) to token indices via offset overlaps."""
    out: List[int] = []
    for i, (s, e) in enumerate(offsets):
        if e <= span_start:
            continue
        if s >= span_end:
            continue
        out.append(i)
    return out

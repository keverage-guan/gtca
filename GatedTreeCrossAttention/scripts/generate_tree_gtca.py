#!/usr/bin/env python3
"""
Offline parse-tree generation for GTCA.

Build a SQLite cache mapping:
  hash(input_ids) -> {"tree_structure": ..., "update_token_indices": ...}

The cache must be generated separately for each backbone tokenizer, because token IDs differ.

Supported tasks:
- mmlu, cloth, hellaswag, winogrande (MCQA prompts via chat template + Answer:)
- cola (chat template + Answer:)
- blimp (plain sentences, no chat template)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from utils.gtca_format import (
    ANSWER_PREFIX,
    build_chat_prompt_text,
    char_span_to_token_indices,
    format_mcqa_user_content,
    get_system_prompt,
    get_turn_end_token,
    label_index_to_letter,
    letter_to_label_index,
    normalize_whitespace,
)

# Parsing dependencies
import spacy
import benepar  # noqa: F401
from nltk import Tree


def _hash_unpadded_input_ids(input_ids: Sequence[int]) -> str:
    """Stable hash for arbitrary token id values."""
    import struct
    m = hashlib.sha256()
    m.update(struct.pack(f"<{len(input_ids)}I", *list(map(int, input_ids))))
    return m.hexdigest()


class SQLiteParsedCache:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self._init()

    def _init(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS parsed_cache (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.conn.commit()

    def set(self, key: str, value: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO parsed_cache (key, value) VALUES (?, ?)",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def _read_csv(path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def _find_split_files(data_path: str) -> Dict[str, str]:
    if not os.path.isdir(data_path):
        return {"train": data_path}

    candidates: Dict[str, str] = {}
    split_aliases = {
        "train": ["train"],
        "validation": ["validation", "valid", "dev"],
        "test": ["test"],
    }
    exts = [".jsonl", ".csv"]

    for split, aliases in split_aliases.items():
        for a in aliases:
            for ext in exts:
                p = os.path.join(data_path, a + ext)
                if os.path.exists(p):
                    candidates[split] = p
                    break
            if split in candidates:
                break

    return candidates


def _load_split_file(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        return _read_jsonl(path)
    if path.endswith(".csv"):
        return _read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def _extract_mcqa_question_and_options(example: Dict[str, Any]) -> Tuple[str, List[str]]:
    q = (
        example.get("question")
        or example.get("prompt")
        or example.get("query")
        or example.get("sentence")
        or example.get("ctx")
        or example.get("context")
    )
    if q is None:
        ctx_a = example.get("ctx_a")
        ctx_b = example.get("ctx_b")
        if ctx_a is not None or ctx_b is not None:
            q = f"{ctx_a or ''} {ctx_b or ''}".strip()
    q = normalize_whitespace(q or "")

    options = None
    for key in ("choices", "options", "endings"):
        if key in example and example[key] is not None:
            options = list(example[key])
            break

    if options is None:
        if all(k in example for k in ("A", "B", "C", "D")):
            options = [example["A"], example["B"], example["C"], example["D"]]
        elif all(k in example for k in ("option_a", "option_b", "option_c", "option_d")):
            options = [example["option_a"], example["option_b"], example["option_c"], example["option_d"]]
        elif all(k in example for k in ("option1", "option2")) and not any(k in example for k in ("option3", "option4")):
            options = [example["option1"], example["option2"]]

    if options is None:
        raise ValueError("Could not find options in example.")

    options = [normalize_whitespace(x) for x in options]
    return q, options


def _extract_mcqa_label(example: Dict[str, Any], num_options: int) -> int:
    for k in ("label", "answer", "gold", "correct", "correct_answer"):
        if k in example and example[k] is not None:
            v = example[k]
            if isinstance(v, (int, float)) and int(v) == v:
                idx = int(v)
                if 0 <= idx < num_options:
                    return idx
            if isinstance(v, str) and v.strip().isdigit():
                idx = int(v.strip()) - 1
                if 0 <= idx < num_options:
                    return idx
            if isinstance(v, str) and v.strip().upper() in {"A", "B", "C", "D"}:
                idx = letter_to_label_index(v)
                if 0 <= idx < num_options:
                    return idx
    raise ValueError("Could not extract MCQA label from example.")


def _tokenize_with_offsets(tokenizer: Any, text: str, max_length: int) -> Tuple[List[int], List[Tuple[int, int]]]:
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
        truncation=False,
    )
    input_ids: List[int] = enc["input_ids"]
    offsets: List[Tuple[int, int]] = enc["offset_mapping"]

    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]
        offsets = offsets[-max_length:]
    return input_ids, offsets


def _build_word_spans_from_spacy(sent) -> List[Tuple[int, int, str]]:
    spans = []
    for tok in sent:
        spans.append((tok.idx, tok.idx + len(tok.text), tok.text))
    return spans


def _build_word_spans_fallback(text: str, leaves: List[str]) -> List[Tuple[int, int, str]]:
    """
    Best-effort fallback: align leaf strings by sequential search in the raw text.
    """
    spans: List[Tuple[int, int, str]] = []
    cursor = 0
    for w in leaves:
        w_norm = w
        # Find next occurrence at/after cursor
        pos = text.find(w_norm, cursor)
        if pos < 0:
            # Try a whitespace-normalized search
            pos = text.find(w_norm.strip(), cursor)
        if pos < 0:
            # Give up and approximate as empty span
            spans.append((cursor, cursor, w))
            continue
        start = pos
        end = pos + len(w_norm)
        spans.append((start, end, w))
        cursor = end
    return spans


def _leaf_token_indices(
    offsets: List[Tuple[int, int]],
    global_span_start: int,
    word_span: Tuple[int, int],
) -> List[int]:
    ws, we = word_span
    span_start = global_span_start + ws
    span_end = global_span_start + we
    return char_span_to_token_indices(offsets, span_start, span_end)


def _build_tree_from_nltk(
    tree: Any,
    leaf_nodes: List[Dict[str, Any]],
    leaf_cursor: int = 0,
) -> Tuple[Dict[str, Any], int]:
    """
    Convert an NLTK Tree to our node dict representation, using pre-built leaf_nodes.
    Each leaf in the NLTK tree corresponds to exactly one leaf_nodes entry.
    """
    if isinstance(tree, str):
        node = leaf_nodes[leaf_cursor]
        return node, leaf_cursor + 1

    children = []
    cur = leaf_cursor
    for ch in tree:
        ch_node, cur = _build_tree_from_nltk(ch, leaf_nodes, cur)
        children.append(ch_node)

    tok_set = set()
    for ch in children:
        tok_set.update(ch.get("token_indices", []))
    token_indices = sorted(tok_set)

    return {
        "node_type": getattr(tree, "label", lambda: "node")(),
        "token_indices": token_indices,
        "children": children,
    }, cur


def _parse_question_to_tree(
    nlp: Any,
    question_text: str,
    prompt_offsets: List[Tuple[int, int]],
    prompt_question_start: int,
) -> Optional[Dict[str, Any]]:
    """
    Parse the question_text using spaCy+benepar, then align leaves to BPE token indices
    using the full prompt offset mapping.
    """
    question_text = question_text.strip()
    if not question_text:
        return None

    doc = nlp(question_text)
    sent_trees: List[Dict[str, Any]] = []
    for sent in doc.sents:
        if not hasattr(sent._, "parse_tree") or sent._.parse_tree is None:
            continue
        pt: Tree = sent._.parse_tree
        leaves = list(pt.leaves())

        # Build word spans for leaves
        spacy_spans = _build_word_spans_from_spacy(sent)
        if len(spacy_spans) != len(leaves):
            word_spans = _build_word_spans_fallback(sent.text, leaves)
        else:
            # Convert to sentence-local offsets
            sent_offset = sent.start_char
            word_spans = [(s - sent_offset, e - sent_offset, w) for (s, e, w) in spacy_spans]

        leaf_nodes: List[Dict[str, Any]] = []
        for (ws, we, w) in word_spans:
            tok_idx = _leaf_token_indices(prompt_offsets, prompt_question_start + sent.start_char, (ws, we))
            leaf_nodes.append({"node_type": "subword_block", "token_indices": tok_idx, "children": []})

        tree_dict, final_cursor = _build_tree_from_nltk(pt, leaf_nodes, 0)
        # If mismatch, skip
        if final_cursor != len(leaf_nodes):
            continue
        sent_trees.append(tree_dict)

    if not sent_trees:
        return None
    if len(sent_trees) == 1:
        return sent_trees[0]

    # Multi-sentence: wrap under a question root
    tok_set = set()
    for st in sent_trees:
        tok_set.update(st.get("token_indices", []))
    return {"node_type": "question_root", "token_indices": sorted(tok_set), "children": sent_trees}


def _build_mcqa_parsed(
    tokenizer: Any,
    nlp: Any,
    task: str,
    example: Dict[str, Any],
    include_label: bool,
    max_length: int,
) -> Tuple[List[int], Dict[str, Any]]:
    sys_prompt = get_system_prompt(task)
    turn_end = get_turn_end_token(tokenizer)

    q, options = _extract_mcqa_question_and_options(example)
    mcqa = format_mcqa_user_content(task, q, options)
    prompt_text = build_chat_prompt_text(tokenizer, sys_prompt, mcqa.user_content)

    gold_idx = None
    gold_letter = None
    text = prompt_text
    if include_label:
        gold_idx = _extract_mcqa_label(example, len(options))
        if len(options) == 4:
            gold_letter = label_index_to_letter(gold_idx)
        else:
            gold_letter = "A" if gold_idx == 0 else "B"
        text = prompt_text + " " + gold_letter + (turn_end or "")

    input_ids, offsets = _tokenize_with_offsets(tokenizer, text, max_length=max_length)

    # Locate user_content span in the full prompt text.
    user_pos = text.find(mcqa.user_content)
    if user_pos < 0:
        raise ValueError("Could not locate user_content inside the prompt text.")

    # Global spans
    q_span_global = (user_pos + mcqa.question_span[0], user_pos + mcqa.question_span[1])
    opt_header_global = (user_pos + mcqa.options_header_span[0], user_pos + mcqa.options_header_span[1])

    option_token_set = set(char_span_to_token_indices(offsets, opt_header_global[0], opt_header_global[1]))
    for _, (s, e) in mcqa.option_spans.items():
        gs, ge = user_pos + s, user_pos + e
        option_token_set.update(char_span_to_token_indices(offsets, gs, ge))

    # Token update mask: allow updates for all tokens except option region tokens.
    update_token_indices = [i for i in range(len(input_ids)) if i not in option_token_set]

    # Build tree structure
    all_indices = list(range(len(input_ids)))

    question_text = mcqa.user_content[mcqa.question_span[0] : mcqa.question_span[1]]
    question_tree = _parse_question_to_tree(nlp, question_text, offsets, q_span_global[0])

    options_children = []
    options_tok = set()
    for lab, (s, e) in mcqa.option_spans.items():
        gs, ge = user_pos + s, user_pos + e
        tok_idx = char_span_to_token_indices(offsets, gs, ge)
        options_tok.update(tok_idx)
        options_children.append({"node_type": f"option_{lab}", "token_indices": tok_idx, "children": []})

    options_node = {"node_type": "options_root", "token_indices": sorted(options_tok), "children": options_children}

    answer_pos = text.rfind(ANSWER_PREFIX)
    ans_tok_idx = []
    if answer_pos >= 0:
        ans_tok_idx = char_span_to_token_indices(offsets, answer_pos, len(text))
    answer_node = {"node_type": "answer_field", "token_indices": ans_tok_idx, "children": []}

    children = []
    if question_tree is not None:
        children.append(question_tree)
    children.append(options_node)
    children.append(answer_node)

    root = {"node_type": "root", "token_indices": all_indices, "children": children}

    parsed = {"tree_structure": root, "update_token_indices": update_token_indices}
    return input_ids, parsed


def _build_cola_parsed(
    tokenizer: Any,
    nlp: Any,
    example: Dict[str, Any],
    include_label: bool,
    max_length: int,
) -> Tuple[List[int], Dict[str, Any]]:
    sys_prompt = get_system_prompt("cola")
    turn_end = get_turn_end_token(tokenizer)

    sent = normalize_whitespace(example.get("sentence") or example.get("text") or "")
    prompt_text = build_chat_prompt_text(tokenizer, sys_prompt, sent)

    text = prompt_text
    if include_label and example.get("label") is not None:
        gold = int(example.get("label"))
        text = prompt_text + " " + ("1" if gold == 1 else "0") + (turn_end or "")

    input_ids, offsets = _tokenize_with_offsets(tokenizer, text, max_length=max_length)

    # No options: update all tokens
    update_token_indices = list(range(len(input_ids)))

    # Parse the sentence as the "question"
    user_pos = text.find(sent)
    question_tree = None
    if user_pos >= 0:
        question_tree = _parse_question_to_tree(nlp, sent, offsets, user_pos)

    root = {"node_type": "root", "token_indices": list(range(len(input_ids))), "children": []}
    if question_tree is not None:
        root["children"].append(question_tree)
    parsed = {"tree_structure": root, "update_token_indices": update_token_indices}
    return input_ids, parsed


def _build_blimp_parsed(
    tokenizer: Any,
    nlp: Any,
    sentence: str,
    max_length: int,
) -> Tuple[List[int], Dict[str, Any]]:
    input_ids, offsets = _tokenize_with_offsets(tokenizer, sentence, max_length=max_length)

    # Update all tokens
    update_token_indices = list(range(len(input_ids)))

    question_tree = _parse_question_to_tree(nlp, sentence, offsets, 0)
    root = {"node_type": "root", "token_indices": list(range(len(input_ids))), "children": []}
    if question_tree is not None:
        root["children"].append(question_tree)
    parsed = {"tree_structure": root, "update_token_indices": update_token_indices}
    return input_ids, parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--task", type=str, required=True, choices=["mmlu", "cloth", "hellaswag", "winogrande", "cola", "blimp"])
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--cache_path", type=str, required=True)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--spacy_model", type=str, default="en_core_web_sm")
    ap.add_argument("--benepar_model", type=str, default="benepar_en3")
    ap.add_argument("--include_label_splits", type=str, default="train,validation")
    ap.add_argument("--splits", type=str, default="train,validation,test")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    nlp = spacy.load(args.spacy_model)
    if "benepar" not in nlp.pipe_names:
        nlp.add_pipe("benepar", config={"model": args.benepar_model})

    split_files = _find_split_files(args.data_path)
    wanted_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    include_label = set(s.strip() for s in args.include_label_splits.split(",") if s.strip())

    db = SQLiteParsedCache(args.cache_path)

    for split in tqdm(wanted_splits, desc="Splits", unit="split"):
        if split not in split_files:
            continue
        data = _load_split_file(split_files[split])
        do_label = split in include_label

        for ex in tqdm(data, desc=f"Parsing [{split}]", unit="example", leave=False, miniters=100):
            if args.task == "blimp":
                good = normalize_whitespace(ex.get("sentence_good") or ex.get("good") or ex.get("s_good") or "")
                bad = normalize_whitespace(ex.get("sentence_bad") or ex.get("bad") or ex.get("s_bad") or "")
                for sent in tqdm((good, bad), desc="Sentences", unit="sent", leave=False, miniters=100):
                    if not sent:
                        continue
                    input_ids, parsed = _build_blimp_parsed(tokenizer, nlp, sent, max_length=args.max_length)
                    db.set(_hash_unpadded_input_ids(input_ids), parsed)
            elif args.task == "cola":
                input_ids, parsed = _build_cola_parsed(tokenizer, nlp, ex, include_label=do_label, max_length=args.max_length)
                db.set(_hash_unpadded_input_ids(input_ids), parsed)
            else:
                input_ids, parsed = _build_mcqa_parsed(tokenizer, nlp, args.task, ex, include_label=do_label, max_length=args.max_length)
                db.set(_hash_unpadded_input_ids(input_ids), parsed)

    db.close()


if __name__ == "__main__":
    main()
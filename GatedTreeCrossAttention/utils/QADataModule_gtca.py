"""
PyTorch Lightning DataModule for GTCA.

Properties:
- Model-agnostic chat formatting using tokenizer.apply_chat_template().
- Deterministic offline parse-tree retrieval through a SQLite cache keyed by a hash of unpadded input_ids.

"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from utils.gtca_format import (
    build_chat_prompt_text,
    format_mcqa_user_content,
    get_system_prompt,
    get_turn_end_token,
    label_index_to_letter,
    letter_to_label_index,
    normalize_whitespace,
)


def _hash_unpadded_input_ids(input_ids: Sequence[int]) -> str:
    """Stable hash for arbitrary token id values."""
    import struct
    m = hashlib.sha256()
    # Little-endian uint32 packing
    m.update(struct.pack(f"<{len(input_ids)}I", *list(map(int, input_ids))))
    return m.hexdigest()


class SQLiteParsedCache:
    """
    Simple SQLite key-value store:
      key: sha256 hash string
      value: JSON string (parsed tree dict)
    """

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

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute("SELECT value FROM parsed_cache WHERE key=?", (key,))
        row = cur.fetchone()
        if row is None:
            return None
        return json.loads(row[0])

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
    """
    If data_path is a directory, try to locate train/validation/test files.
    Supports .jsonl and .csv. Returns a dict: split -> filepath.
    """
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
    """
    Best-effort extraction of MCQA question/context and options from common formats.

    Supported sources:
    - MMLU/CLOTH-like: question + choices/options
    - HellaSwag-like: ctx or (ctx_a, ctx_b) + endings
    - Winogrande-like: sentence + option1/option2
    """
    # Question/context
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

    # Options
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
    """
    Extract label as an index [0..num_options-1] from common formats.
    """
    for k in ("label", "answer", "gold", "correct", "correct_answer"):
        if k in example and example[k] is not None:
            v = example[k]
            # Integer index
            if isinstance(v, (int, float)) and int(v) == v:
                idx = int(v)
                if 0 <= idx < num_options:
                    return idx
            # Winogrande often uses "1"/"2"
            if isinstance(v, str) and v.strip().isdigit():
                idx = int(v.strip()) - 1
                if 0 <= idx < num_options:
                    return idx
            # Letter label
            if isinstance(v, str) and v.strip().upper() in {"A", "B", "C", "D"}:
                idx = letter_to_label_index(v)
                if 0 <= idx < num_options:
                    return idx
    raise ValueError("Could not extract MCQA label from example.")


class QADataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer: Any,
        data_path: str,
        batch_size: int,
        max_length: int,
        task: str,
        cache_path: Optional[str] = None,
        num_workers: int = 0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.task = (task or "").lower()
        self.cache_path = cache_path
        self.num_workers = int(num_workers)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self._parsed_db: Optional[SQLiteParsedCache] = SQLiteParsedCache(cache_path) if cache_path else None

        self.train_data: List[Dict[str, Any]] = []
        self.val_data: List[Dict[str, Any]] = []
        self.test_data: List[Dict[str, Any]] = []

    def setup(self, stage: Optional[str] = None) -> None:
        split_files = _find_split_files(self.data_path)
        if "train" in split_files:
            self.train_data = _load_split_file(split_files["train"])
        if "validation" in split_files:
            self.val_data = _load_split_file(split_files["validation"])
        if "test" in split_files:
            self.test_data = _load_split_file(split_files["test"])

        if not self.val_data and self.train_data:
            n = max(1, int(0.01 * len(self.train_data)))
            self.val_data = self.train_data[-n:]

        if not self.test_data and self.val_data:
            self.test_data = self.val_data

    def _encode_text(self, text: str) -> Tuple[List[int], List[int]]:
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=True,
            truncation=False,
        )
        input_ids: List[int] = enc["input_ids"]
        attention_mask: List[int] = enc["attention_mask"]

        if len(input_ids) > self.max_length:
            input_ids = input_ids[-self.max_length :]
            attention_mask = attention_mask[-self.max_length :]
        return input_ids, attention_mask

    @staticmethod
    def _pad_left(seqs: List[List[int]], pad_value: int) -> torch.Tensor:
        max_len = max(len(x) for x in seqs) if seqs else 0
        out = []
        for x in seqs:
            pad_len = max_len - len(x)
            out.append(([pad_value] * pad_len) + x)
        return torch.tensor(out, dtype=torch.long)

    def _load_parsed_batch(self, input_ids_list: List[List[int]]) -> List[Optional[Dict[str, Any]]]:
        if not self._parsed_db:
            return [None for _ in input_ids_list]
        out: List[Optional[Dict[str, Any]]] = []
        for ids in input_ids_list:
            out.append(self._parsed_db.get(_hash_unpadded_input_ids(ids)))
        return out

    def _collate_mcqa(self, examples: List[Dict[str, Any]], include_label: bool) -> Dict[str, Any]:
        sys_prompt = get_system_prompt(self.task)
        turn_end = get_turn_end_token(self.tokenizer)

        input_ids_list: List[List[int]] = []
        attention_mask_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        meta_items: List[Dict[str, Any]] = []

        for ex in examples:
            question, options = _extract_mcqa_question_and_options(ex)
            mcqa = format_mcqa_user_content(self.task, question, options)
            prompt_text = build_chat_prompt_text(self.tokenizer, sys_prompt, mcqa.user_content)

            gold_idx = None
            gold_letter = None

            if include_label:
                gold_idx = _extract_mcqa_label(ex, len(options))
                if len(options) == 4:
                    gold_letter = label_index_to_letter(gold_idx)
                else:
                    gold_letter = "A" if gold_idx == 0 else "B"

                completion = " " + gold_letter
                full_text = prompt_text + completion + (turn_end or "")

                prompt_ids, _ = self._encode_text(prompt_text)
                full_ids, full_attn = self._encode_text(full_text)

                prompt_len = len(prompt_ids)
                labels = ([-100] * prompt_len) + full_ids[prompt_len:]
                labels = labels[-len(full_ids) :]

                input_ids_list.append(full_ids)
                attention_mask_list.append(full_attn)
                labels_list.append(labels)
            else:
                prompt_ids, prompt_attn = self._encode_text(prompt_text)
                input_ids_list.append(prompt_ids)
                attention_mask_list.append(prompt_attn)
                labels_list.append([-100] * len(prompt_ids))
                try:
                    gold_idx = _extract_mcqa_label(ex, len(options))
                    if len(options) == 4:
                        gold_letter = label_index_to_letter(gold_idx)
                    else:
                        gold_letter = "A" if gold_idx == 0 else "B"
                except Exception:
                    gold_idx, gold_letter = None, None

            meta_items.append(
                {
                    "question": question,
                    "options": options,
                    "gold_index": gold_idx,
                    "gold_letter": gold_letter,
                    "prompt_text": prompt_text,
                    "user_content": mcqa.user_content,
                    "question_span": mcqa.question_span,
                    "option_spans": mcqa.option_spans,
                    "options_header_span": mcqa.options_header_span,
                }
            )

        parsed_list = self._load_parsed_batch(input_ids_list)
        pad_id = int(self.tokenizer.pad_token_id)

        batch_input_ids = self._pad_left(input_ids_list, pad_id)
        batch_attention_mask = self._pad_left(attention_mask_list, 0)
        batch_labels = self._pad_left(labels_list, -100)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
            "parsed": parsed_list,
            "meta": meta_items,
            "task": self.task,
        }

    def _collate_cola(self, examples: List[Dict[str, Any]], include_label: bool) -> Dict[str, Any]:
        sys_prompt = get_system_prompt("cola")
        turn_end = get_turn_end_token(self.tokenizer)

        input_ids_list: List[List[int]] = []
        attention_mask_list: List[List[int]] = []
        labels_list: List[List[int]] = []
        meta_items: List[Dict[str, Any]] = []

        for ex in examples:
            sent = normalize_whitespace(ex.get("sentence") or ex.get("text") or "")
            prompt_text = build_chat_prompt_text(self.tokenizer, sys_prompt, sent)

            gold = None
            if include_label:
                gold = int(ex.get("label"))
                completion = " " + ("1" if gold == 1 else "0")
                full_text = prompt_text + completion + (turn_end or "")
                prompt_ids, _ = self._encode_text(prompt_text)
                full_ids, full_attn = self._encode_text(full_text)
                prompt_len = len(prompt_ids)
                labels = ([-100] * prompt_len) + full_ids[prompt_len:]
                labels = labels[-len(full_ids) :]
                input_ids_list.append(full_ids)
                attention_mask_list.append(full_attn)
                labels_list.append(labels)
            else:
                prompt_ids, prompt_attn = self._encode_text(prompt_text)
                input_ids_list.append(prompt_ids)
                attention_mask_list.append(prompt_attn)
                labels_list.append([-100] * len(prompt_ids))
                if ex.get("label") is not None:
                    gold = int(ex.get("label"))

            meta_items.append({"sentence": sent, "gold": gold, "prompt_text": prompt_text})

        parsed_list = self._load_parsed_batch(input_ids_list)
        pad_id = int(self.tokenizer.pad_token_id)

        batch_input_ids = self._pad_left(input_ids_list, pad_id)
        batch_attention_mask = self._pad_left(attention_mask_list, 0)
        batch_labels = self._pad_left(labels_list, -100)

        return {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "labels": batch_labels,
            "parsed": parsed_list,
            "meta": meta_items,
            "task": "cola",
        }

    def _collate_blimp(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        good_ids_list: List[List[int]] = []
        good_attn_list: List[List[int]] = []
        bad_ids_list: List[List[int]] = []
        bad_attn_list: List[List[int]] = []
        meta_items: List[Dict[str, Any]] = []

        for ex in examples:
            good = normalize_whitespace(ex.get("sentence_good") or ex.get("good") or ex.get("s_good") or "")
            bad = normalize_whitespace(ex.get("sentence_bad") or ex.get("bad") or ex.get("s_bad") or "")
            good_ids, good_attn = self._encode_text(good)
            bad_ids, bad_attn = self._encode_text(bad)
            good_ids_list.append(good_ids)
            good_attn_list.append(good_attn)
            bad_ids_list.append(bad_ids)
            bad_attn_list.append(bad_attn)
            meta_items.append({"sentence_good": good, "sentence_bad": bad, "uid": ex.get("UID") or ex.get("uid")})

        good_parsed = self._load_parsed_batch(good_ids_list)
        bad_parsed = self._load_parsed_batch(bad_ids_list)

        pad_id = int(self.tokenizer.pad_token_id)

        batch_good_ids = self._pad_left(good_ids_list, pad_id)
        batch_good_attn = self._pad_left(good_attn_list, 0)
        batch_bad_ids = self._pad_left(bad_ids_list, pad_id)
        batch_bad_attn = self._pad_left(bad_attn_list, 0)

        return {
            "good_input_ids": batch_good_ids,
            "good_attention_mask": batch_good_attn,
            "bad_input_ids": batch_bad_ids,
            "bad_attention_mask": batch_bad_attn,
            "good_parsed": good_parsed,
            "bad_parsed": bad_parsed,
            "meta": meta_items,
            "task": "blimp",
        }

    def train_dataloader(self) -> DataLoader:
        if self.task == "cola":
            collate_fn = lambda batch: self._collate_cola(batch, include_label=True)
        elif self.task in {"mmlu", "cloth"}:
            collate_fn = lambda batch: self._collate_mcqa(batch, include_label=True)
        else:
            collate_fn = lambda batch: self._collate_mcqa(batch, include_label=True)

        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        if self.task == "cola":
            collate_fn = lambda batch: self._collate_cola(batch, include_label=True)
        elif self.task in {"mmlu", "cloth"}:
            collate_fn = lambda batch: self._collate_mcqa(batch, include_label=True)
        else:
            collate_fn = lambda batch: self._collate_mcqa(batch, include_label=True)

        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        if self.task == "blimp":
            collate_fn = self._collate_blimp
        elif self.task == "cola":
            collate_fn = lambda batch: self._collate_cola(batch, include_label=False)
        else:
            collate_fn = lambda batch: self._collate_mcqa(batch, include_label=False)

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

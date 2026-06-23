#!/usr/bin/env python3
"""
Convert raw dataset files into the JSONL format expected by generate_tree_gtca.py.

Run from the repo root, i.e. the directory that contains both
  datasets/
  GatedTreeCrossAttention/

Usage:
  python prepare_gtca_data.py

Output layout (matches what generate_tree_gtca.py --data_path expects):
  GatedTreeCrossAttention/data/blimp/          test.jsonl
  GatedTreeCrossAttention/data/cloth/          train.jsonl  validation.jsonl  test.jsonl
  GatedTreeCrossAttention/data/glue/cola/      train.jsonl  validation.jsonl  test.jsonl
  GatedTreeCrossAttention/data/hellaswag/      train.jsonl  validation.jsonl  test.jsonl
  GatedTreeCrossAttention/data/mmlu/           train.jsonl  validation.jsonl  test.jsonl
  GatedTreeCrossAttention/data/winogrande/     train.jsonl  validation.jsonl  test.jsonl

Field contracts (what generate_tree_gtca.py reads):
  CLOTH / MMLU / HellaSwag   -> question, choices (list), answer (letter A-D or int)
  WinoGrande                 -> sentence, option1, option2, answer ("1"/"2")
  CoLA                       -> sentence, label (int 0/1)
  BLiMP                      -> sentence_good, sentence_bad
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterator, List

import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_jsonl(records: List[Dict[str, Any]], path: str) -> None:
    _mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(records):>7,}  ->  {path}")


def _read_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


def _glob_parquet(directory: str, prefix: str) -> str:
    """Return the first parquet file in directory whose name starts with prefix."""
    pattern = os.path.join(directory, f"{prefix}*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No parquet matching {pattern}")
    return files[0]


# ---------------------------------------------------------------------------
# BLiMP
# Concatenate all per-phenomenon JSONL files; test split only.
# Source fields already match: sentence_good, sentence_bad.
# ---------------------------------------------------------------------------

def convert_blimp(src: str, dst: str) -> None:
    print("\n[BLiMP]")
    data_dir = os.path.join(src, "blimp", "data")
    jsonl_files = sorted(glob.glob(os.path.join(data_dir, "*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under {data_dir}")

    records: List[Dict[str, Any]] = []
    for path in jsonl_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                records.append({
                    "sentence_good": ex["sentence_good"],
                    "sentence_bad":  ex["sentence_bad"],
                })

    # BLiMP has test only (no train/dev split in this distribution)
    _write_jsonl(records, os.path.join(dst, "blimp", "test.jsonl"))


# ---------------------------------------------------------------------------
# CLOTH
# Source: CLOTH_{train,valid,test}_cleaned.json
# Each file is a list of article dicts:
#   { "article": str, "questions": [str], "options": [[str,str,str,str]], "answers": [str] }
# Flatten to one example per question; use article+question as the "question" field.
# ---------------------------------------------------------------------------

def _cloth_flat_to_records(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Each item: { "sentence": "...  [MASK]  ...", "answer": str, "distractors": [str, str, str] }
    Shuffle answer into choices at a deterministic position (based on index) so
    the correct label isn't always index 0.
    """
    import random
    records: List[Dict[str, Any]] = []
    for i, item in enumerate(items):
        correct    = str(item["answer"])
        wrong      = [str(d) for d in item["distractors"]]
        choices    = [correct] + wrong          # start with correct at 0
        rng        = random.Random(i)           # deterministic per example
        rng.shuffle(choices)
        answer_idx = choices.index(correct)     # 0-3
        records.append({
            "question": str(item["sentence"]),  # contains [MASK] placeholder
            "choices":  choices,
            "answer":   answer_idx,
        })
    return records


def convert_cloth(src: str, dst: str) -> None:
    print("\n[CLOTH]")
    cloth_dir = os.path.join(src, "cloth")
    split_map = {
        "train":      "CLOTH_train_cleaned.json",
        "validation": "CLOTH_valid_cleaned.json",
        "test":       "CLOTH_test_cleaned.json",
    }
    for split, filename in split_map.items():
        path = os.path.join(cloth_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        records = _cloth_flat_to_records(items)
        _write_jsonl(records, os.path.join(dst, "cloth", f"{split}.jsonl"))


# ---------------------------------------------------------------------------
# CoLA  (GLUE)
# Source: {train,validation,test}-00000-of-00001.parquet
# Parquet columns: sentence, label  (test labels are -1 / missing in GLUE)
# ---------------------------------------------------------------------------

def convert_cola(src: str, dst: str) -> None:
    print("\n[CoLA]")
    cola_dir = os.path.join(src, "glue", "cola")
    split_map = {
        "train":      "train",
        "validation": "validation",
        "test":       "test",
    }
    for split, prefix in split_map.items():
        path = _glob_parquet(cola_dir, prefix)
        df   = _read_parquet(path)
        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            rec: Dict[str, Any] = {"sentence": str(row["sentence"])}
            # GLUE test set has no gold labels; omit 'label' so the script skips it
            if "label" in df.columns and int(row.get("label", -1)) >= 0:
                rec["label"] = int(row["label"])
            records.append(rec)
        _write_jsonl(records, os.path.join(dst, "cola", f"{split}.jsonl"))


# ---------------------------------------------------------------------------
# HellaSwag
# Source: {train,validation,test}-00000-of-00001.parquet
# Parquet columns: ctx, endings (list), label (str digit or int)
# ---------------------------------------------------------------------------

def convert_hellaswag(src: str, dst: str) -> None:
    print("\n[HellaSwag]")
    hs_dir = os.path.join(src, "hellaswag", "data")
    split_map = {
        "train":      "train",
        "validation": "validation",
        "test":       "test",
    }
    for split, prefix in split_map.items():
        path = _glob_parquet(hs_dir, prefix)
        df   = _read_parquet(path)
        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            endings = row["endings"]
            # endings may be stored as a list or a JSON string
            if isinstance(endings, str):
                endings = json.loads(endings)
            rec: Dict[str, Any] = {
                "question": str(row["ctx"]),
                "choices":  [str(e) for e in endings],
            }
            lbl = row.get("label", None)
            if lbl is not None:
                try:
                    idx = int(lbl)
                    if 0 <= idx <= 3:
                        rec["answer"] = idx
                except (ValueError, TypeError):
                    pass
            records.append(rec)
        _write_jsonl(records, os.path.join(dst, "hellaswag", f"{split}.jsonl"))


# ---------------------------------------------------------------------------
# MMLU
# Source layout:
#   mmlu/all/dev-00000-of-00001.parquet        (285 rows  -> validation)
#   mmlu/all/validation-00000-of-00001.parquet (might be empty or absent)
#   mmlu/all/test-00000-of-00001.parquet       (14042 rows)
#   mmlu/all/auxiliary_train-*.parquet         (99842 rows -> train)
# Parquet columns: question, choices (list), answer (int 0-3)
# ---------------------------------------------------------------------------

def _mmlu_parquet_to_records(path: str) -> List[Dict[str, Any]]:
    df = _read_parquet(path)
    records: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        choices = row["choices"]
        if isinstance(choices, str):
            choices = json.loads(choices)
        rec: Dict[str, Any] = {
            "question": str(row["question"]),
            "choices":  [str(c) for c in choices],
        }
        ans = row.get("answer", None)
        if ans is not None:
            try:
                rec["answer"] = int(ans)   # 0-3
            except (ValueError, TypeError):
                rec["answer"] = str(ans)
        records.append(rec)
    return records


def convert_mmlu(src: str, dst: str) -> None:
    print("\n[MMLU]")
    mmlu_all = os.path.join(src, "mmlu", "all")

    # train <- auxiliary_train (may be multiple shards)
    aux_files = sorted(glob.glob(os.path.join(mmlu_all, "auxiliary_train*.parquet")))
    if not aux_files:
        raise FileNotFoundError(f"No auxiliary_train parquet in {mmlu_all}")
    train_records: List[Dict[str, Any]] = []
    for p in aux_files:
        train_records.extend(_mmlu_parquet_to_records(p))
    _write_jsonl(train_records, os.path.join(dst, "mmlu", "train.jsonl"))

    # validation <- dev parquet (the 285-row split used as dev in the paper)
    val_path = _glob_parquet(mmlu_all, "dev")
    _write_jsonl(_mmlu_parquet_to_records(val_path), os.path.join(dst, "mmlu", "validation.jsonl"))

    # test <- test parquet
    test_path = _glob_parquet(mmlu_all, "test")
    _write_jsonl(_mmlu_parquet_to_records(test_path), os.path.join(dst, "mmlu", "test.jsonl"))


# ---------------------------------------------------------------------------
# WinoGrande  (use winogrande_xl: 40398 train rows)
# Source: winogrande_xl/{train,validation,test}-00000-of-00001.parquet
# Parquet columns: sentence, option1, option2, answer ("1" or "2")
# ---------------------------------------------------------------------------

def convert_winogrande(src: str, dst: str) -> None:
    print("\n[WinoGrande]")
    wg_dir = os.path.join(src, "winogrande", "winogrande_xl")
    split_map = {
        "train":      "train",
        "validation": "validation",
        "test":       "test",
    }
    for split, prefix in split_map.items():
        path = _glob_parquet(wg_dir, prefix)
        df   = _read_parquet(path)
        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            rec: Dict[str, Any] = {
                "sentence": str(row["sentence"]),
                "option1":  str(row["option1"]),
                "option2":  str(row["option2"]),
            }
            ans = row.get("answer", None)
            # Test set answer may be empty string; skip it
            if ans is not None and str(ans).strip() in ("1", "2"):
                rec["answer"] = str(ans).strip()
            records.append(rec)
        _write_jsonl(records, os.path.join(dst, "winogrande", f"{split}.jsonl"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Assume script is run from the repo root (parent of both datasets/ and GatedTreeCrossAttention/)
    src = "datasets"
    dst = os.path.join("GatedTreeCrossAttention", "data")

    convert_blimp(src, dst)
    convert_cloth(src, dst)
    convert_cola(src, dst)
    convert_hellaswag(src, dst)
    convert_mmlu(src, dst)
    convert_winogrande(src, dst)

    print("\nDone. Final layout:")
    for root, dirs, files in os.walk(dst):
        dirs.sort()
        level = root.replace(dst, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            size  = os.path.getsize(fpath)
            print(f"{indent}  {fname}  ({size:,} bytes)")


if __name__ == "__main__":
    main()
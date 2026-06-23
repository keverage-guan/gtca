#!/usr/bin/env python3
"""
Merge CLOTH and MMLU data splits into a single combined dataset directory.

Usage:
    python merge_data.py \
        --cloth_dir data/cloth \
        --mmlu_dir  data/mmlu \
        --out_dir   data/cloth_mmlu

The combined train split is shuffled (fixed seed=42) so examples from both
datasets are interleaved rather than block-ordered, which stabilises training.
Validation splits are concatenated without shuffling.
"""

import argparse
import json
import os
import random

SPLIT_ALIASES = {
    "train":      ["train"],
    "validation": ["validation", "valid", "dev"],
    "test":       ["test"],
}
EXTS = [".jsonl", ".csv"]


def find_split_file(data_dir: str, split: str) -> str | None:
    for alias in SPLIT_ALIASES[split]:
        for ext in EXTS:
            p = os.path.join(data_dir, alias + ext)
            if os.path.exists(p):
                return p
    return None


def read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv(path: str) -> list[dict]:
    import csv
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def read_split(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        return read_jsonl(path)
    if path.endswith(".csv"):
        return read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


def write_jsonl(rows: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_split(
    dir_a: str,
    dir_b: str,
    split: str,
    out_dir: str,
    shuffle: bool = False,
    seed: int = 42,
) -> int:
    path_a = find_split_file(dir_a, split)
    path_b = find_split_file(dir_b, split)

    rows_a = read_split(path_a) if path_a else []
    rows_b = read_split(path_b) if path_b else []

    if not rows_a and not rows_b:
        print(f"  [{split}] No files found in either directory — skipping.")
        return 0

    combined = rows_a + rows_b
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(combined)

    out_path = os.path.join(out_dir, f"{split}.jsonl")
    write_jsonl(combined, out_path)

    src_a = f"{len(rows_a)} from {os.path.basename(dir_a)}" if rows_a else "0"
    src_b = f"{len(rows_b)} from {os.path.basename(dir_b)}" if rows_b else "0"
    print(f"  [{split}] {src_a} + {src_b} = {len(combined)} total → {out_path}")
    return len(combined)


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge two MCQA dataset directories.")
    ap.add_argument("--cloth_dir", required=True, help="Path to CLOTH data directory")
    ap.add_argument("--mmlu_dir",  required=True, help="Path to MMLU data directory")
    ap.add_argument("--out_dir",   required=True, help="Output directory for merged data")
    ap.add_argument("--seed", type=int, default=42, help="Shuffle seed for train split")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Merging {args.cloth_dir} + {args.mmlu_dir} → {args.out_dir}\n")

    # Train: shuffle to interleave CLOTH and MMLU examples
    merge_split(args.cloth_dir, args.mmlu_dir, "train",      args.out_dir, shuffle=True,  seed=args.seed)
    # Validation: no shuffle needed
    merge_split(args.cloth_dir, args.mmlu_dir, "validation", args.out_dir, shuffle=False)
    # Test: included for completeness but not used during training
    merge_split(args.cloth_dir, args.mmlu_dir, "test",       args.out_dir, shuffle=False)

    print("\nDone.")


if __name__ == "__main__":
    main()
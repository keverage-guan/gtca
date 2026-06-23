#!/usr/bin/env python3
"""
GTCA evaluation script.

Supported tasks:
- hellaswag, winogrande, mmlu, cloth (MCQA): predict by comparing next-token probabilities of option letters.
- cola: predict by comparing next-token probabilities of " 0" vs " 1".
- blimp: compare sentence log-likelihood (good vs bad) if both are provided.

Expect a parse-tree cache generated for the same tokenizer and prompt formatting.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from peft import LoraConfig

from utils.QADataModule_gtca import QADataModule
from model.GTCA_Model import GTCAModel


def _parse_list_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _pick_single_token_id(tokenizer: Any, candidates: List[str]) -> Tuple[int, str]:
    """
    Choose a single token id for one of the candidate strings.
    Prefers candidates that tokenize to exactly one token.
    Returns (token_id, chosen_string).
    """
    for s in candidates:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], s
    # Fallback: use the first token of the first candidate
    ids = tokenizer.encode(candidates[0], add_special_tokens=False)
    if not ids:
        raise ValueError(f"Could not tokenize candidate: {candidates[0]!r}")
    return ids[0], candidates[0]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--checkpoint_path", type=str, required=False, default=None)
    ap.add_argument("--task", type=str, required=True, choices=["hellaswag", "winogrande", "mmlu", "cloth", "cola", "blimp"])
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--cache_path", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=1.0)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    ap.add_argument("--output_predictions", type=str, default=None)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=_parse_list_arg(args.lora_target_modules),
    )

    model = GTCAModel(
        model_name_or_path=args.model_name_or_path,
        lora_config=lora_config,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        attn_dropout=0.0,
        max_chunks_per_height=64,
    )

    if args.checkpoint_path:
        ckpt = torch.load(args.checkpoint_path, map_location="cpu")
        # Try Lightning-style state dict first
        state_dict = ckpt.get("state_dict", ckpt)
        # Lightning prefixes module parameters with "model."
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                cleaned[k[len("model.") :]] = v
            else:
                cleaned[k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if unexpected:
            print(f"Warning: unexpected keys: {len(unexpected)}")
        if missing:
            print(f"Warning: missing keys: {len(missing)}")

    model.eval()
    model.set_alpha(args.alpha)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    dm = QADataModule(
        tokenizer=tokenizer,
        data_path=args.data_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        task=args.task,
        cache_path=args.cache_path,
        num_workers=args.num_workers,
    )
    dm.setup()
    dl = dm.predict_dataloader()

    preds_out = []
    correct = 0
    total = 0

    device = next(model.parameters()).device

    if args.task in {"hellaswag", "winogrande", "mmlu", "cloth"}:
        # Candidate letters
        letters = ["A", "B", "C", "D"]
        # For 2-option tasks, we will restrict based on meta options length.

        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            parsed_list = batch.get("parsed")
            meta = batch.get("meta", [])

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, parsed_list=parsed_list)
            logits = out.logits  # (B, T, V)
            next_logits = logits[:, -1, :]
            next_logprobs = F.log_softmax(next_logits, dim=-1)

            for i in range(input_ids.size(0)):
                opts = meta[i].get("options") if meta else None
                num_opts = len(opts) if isinstance(opts, list) else 4
                cand_letters = letters[:num_opts]

                cand_token_ids = []
                chosen_forms = []
                for lab in cand_letters:
                    tok_id, form = _pick_single_token_id(tokenizer, [f" {lab}", lab])
                    cand_token_ids.append(tok_id)
                    chosen_forms.append(form)

                scores = next_logprobs[i, torch.tensor(cand_token_ids, device=device)]
                pred_idx = int(torch.argmax(scores).item())
                pred_letter = cand_letters[pred_idx]

                gold_idx = meta[i].get("gold_index") if meta else None
                gold_letter = meta[i].get("gold_letter") if meta else None

                if gold_idx is not None:
                    total += 1
                    if int(gold_idx) == pred_idx:
                        correct += 1

                preds_out.append(
                    {
                        "pred_index": pred_idx,
                        "pred_letter": pred_letter,
                        "gold_index": gold_idx,
                        "gold_letter": gold_letter,
                        "candidate_token_forms": chosen_forms,
                    }
                )

    elif args.task == "cola":
        for batch in dl:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            parsed_list = batch.get("parsed")
            meta = batch.get("meta", [])

            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None, parsed_list=parsed_list)
            logits = out.logits
            next_logits = logits[:, -1, :]
            next_logprobs = F.log_softmax(next_logits, dim=-1)

            tok0, form0 = _pick_single_token_id(tokenizer, [" 0", "0"])
            tok1, form1 = _pick_single_token_id(tokenizer, [" 1", "1"])

            for i in range(input_ids.size(0)):
                score0 = float(next_logprobs[i, tok0].item())
                score1 = float(next_logprobs[i, tok1].item())
                pred = 1 if score1 > score0 else 0
                gold = meta[i].get("gold") if meta else None

                if gold is not None:
                    total += 1
                    if int(gold) == int(pred):
                        correct += 1

                preds_out.append({"pred": pred, "gold": gold, "token_forms": [form0, form1]})

    elif args.task == "blimp":
        # BLiMP batches contain good/bad in the same batch item.
        for batch in dl:
            good_ids = batch["good_input_ids"].to(device)
            good_attn = batch["good_attention_mask"].to(device)
            bad_ids = batch["bad_input_ids"].to(device)
            bad_attn = batch["bad_attention_mask"].to(device)
            good_parsed = batch.get("good_parsed")
            bad_parsed = batch.get("bad_parsed")

            # Compute per-token log-likelihood for each sentence
            def seq_logprob(ids: torch.Tensor, attn: torch.Tensor, parsed_list):
                out = model(input_ids=ids, attention_mask=attn, labels=ids, parsed_list=parsed_list)
                loss = out.loss
                # Sum logprob is -loss * (num_tokens-1) averaged over batch; approximate:
                n = int(attn.sum().item())
                return -float(loss.item()) * max(1, n - 1)

            good_lp = seq_logprob(good_ids, good_attn, good_parsed)
            bad_lp = seq_logprob(bad_ids, bad_attn, bad_parsed)
            pred_good = good_lp > bad_lp

            total += 1
            if pred_good:
                correct += 1
            preds_out.append({"good_logprob": good_lp, "bad_logprob": bad_lp, "pred_good": pred_good})

    acc = None
    if total > 0:
        acc = correct / total
        print(f"Accuracy: {acc:.6f} ({correct}/{total})")
    else:
        print("No labeled examples found; wrote predictions only.")

    if args.output_predictions:
        os.makedirs(os.path.dirname(args.output_predictions), exist_ok=True)
        with open(args.output_predictions, "w", encoding="utf-8") as f:
            for row in preds_out:
                f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()

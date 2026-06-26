#!/usr/bin/env python3
"""
GTCA training script (PyTorch Lightning).

Implements the 3-stage training schedule described in the paper:
- stage1: LoRA only, structural pathway disabled (alpha=0), GTCA parameters frozen.
- stage2: Train GTCA parameters only, LoRA frozen, alpha warmup (default 10%).
- stage3: Jointly train LoRA + GTCA parameters, alpha fixed at alpha_max.

Compatible with:
- Qwen-2.5-7B
- Llama-3-8B

Prerequisite:
- A parse-tree cache (SQLite) generated for this tokenizer and dataset with generate_tree_gtca.py.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.optim import AdamW

from transformers import AutoTokenizer
from peft import LoraConfig

from utils.QADataModule_gtca import QADataModule
from model.GTCA_Model import GTCAModel

torch.set_float32_matmul_precision('high')

def _parse_list_arg(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


class GTCALightningModule(pl.LightningModule):
    def __init__(
        self,
        model: GTCAModel,
        train_stage: str,
        lr: float,
        weight_decay: float,
        alpha_max: float,
        alpha_warmup_ratio: float,
    ):
        super().__init__()
        self.model = model
        self.train_stage = train_stage
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.alpha_max = float(alpha_max)
        self.alpha_warmup_ratio = float(alpha_warmup_ratio)

        self._total_steps: Optional[int] = None

        self._apply_stage_freezing()

    def _apply_stage_freezing(self) -> None:
        stage = self.train_stage

        # Freeze everything by default
        for p in self.model.parameters():
            p.requires_grad = False

        # LoRA parameters (inside backbone) are typically named with "lora_" in PEFT.
        if stage in {"stage1", "stage3"}:
            for name, p in self.model.backbone.named_parameters():
                if "lora_" in name:
                    p.requires_grad = True

        # GTCA parameters
        if stage in {"stage2", "stage3"}:
            for p in self.model.parse_tree_encoder.parameters():
                p.requires_grad = True
            for p in self.model.cross_attn_layers.parameters():
                p.requires_grad = True

    def on_fit_start(self) -> None:
        # Estimated stepping batches is set by Lightning after dataloaders are ready.
        try:
            self._total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            self._total_steps = None
        print(f"[fit_start] total optimizer steps: {self._total_steps}", flush=True)

    def _current_alpha(self) -> float:
        stage = self.train_stage
        if stage == "stage1":
            return 0.0
        if stage == "stage2":
            # Linear warmup to alpha_max
            if not self._total_steps or self._total_steps <= 0:
                return min(self.alpha_max, float(self.global_step) * 1e-3)
            warmup_steps = max(1, int(self.alpha_warmup_ratio * self._total_steps))
            if self.global_step >= warmup_steps:
                return self.alpha_max
            return self.alpha_max * (float(self.global_step) / float(warmup_steps))
        # stage3
        return self.alpha_max

    def training_step(self, batch: Dict[str, Any], batch_idx: int):
        alpha = self._current_alpha()
        self.model.set_alpha(alpha)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        parsed_list = batch.get("parsed")

        # Filter out examples without parsed trees, if any
        if parsed_list is not None:
            keep = [i for i, p in enumerate(parsed_list) if p is not None]
            if len(keep) == 0:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                self.log("train_loss", loss, prog_bar=True)
                self.log("alpha", torch.tensor(alpha, device=self.device), prog_bar=True)
                return loss
            if len(keep) != len(parsed_list):
                input_ids = input_ids[keep]
                attention_mask = attention_mask[keep]
                labels = labels[keep]
                parsed_list = [parsed_list[i] for i in keep]

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            parsed_list=parsed_list,
        )
        loss = out.loss
        self.log("train_loss", loss, prog_bar=True)
        self.log("alpha", torch.tensor(alpha, device=self.device), prog_bar=True)
        current_lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", current_lr, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        alpha = self._current_alpha()
        self.model.set_alpha(alpha)

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        parsed_list = batch.get("parsed")

        if parsed_list is not None:
            keep = [i for i, p in enumerate(parsed_list) if p is not None]
            if len(keep) == 0:
                return
            if len(keep) != len(parsed_list):
                input_ids = input_ids[keep]
                attention_mask = attention_mask[keep]
                labels = labels[keep]
                parsed_list = [parsed_list[i] for i in keep]

        out = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            parsed_list=parsed_list,
        )
        loss = out.loss
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        opt = AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        return opt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--task", type=str, required=True, choices=["mmlu", "cloth", "hellaswag", "winogrande", "cola"])
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--cache_path", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)

    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--train_stage", type=str, default="stage3", choices=["stage1", "stage2", "stage3"])
    ap.add_argument("--alpha_max", type=float, default=1.0)
    ap.add_argument("--alpha_warmup_ratio", type=float, default=0.10)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)

    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )

    ap.add_argument("--precision", type=str, default="bf16-mixed", choices=["bf16-mixed", "16-mixed", "32-true"])
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--accumulate_grad_batches", type=int, default=1)
    ap.add_argument("--max_epochs", type=int, default=1)
    ap.add_argument("--gradient_clip_val", type=float, default=1.0)
    ap.add_argument("--tqdm", action="store_true", help="Enable tqdm progress bar")
    ap.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=(
            "Path to a Lightning checkpoint from a previous training stage. "
            "Only model weights are loaded; optimizer state and epoch counter "
            "are discarded so each stage begins with a fresh optimiser."
        ),
    )
    ap.add_argument("--tree_type", type=str, default="constituency")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

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
        torch_dtype=torch.bfloat16 if "bf16" in args.precision else torch.float16,
        attn_dropout=0.0,
        max_chunks_per_height=64,
        tree_type=args.tree_type
    )

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

    lit = GTCALightningModule(
        model=model,
        train_stage=args.train_stage,
        lr=args.lr,
        weight_decay=args.weight_decay,
        alpha_max=args.alpha_max,
        alpha_warmup_ratio=args.alpha_warmup_ratio,
    )

    if args.ckpt_path is not None:
        print(f"Loading weights from checkpoint: {args.ckpt_path}")
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        # Lightning prefixes every key with "model." (the attribute name inside
        # GTCALightningModule).  Strip that prefix to get GTCAModel keys.
        raw_sd = ckpt.get("state_dict", ckpt)
        cleaned_sd = {
            (k[len("model."):] if k.startswith("model.") else k): v
            for k, v in raw_sd.items()
        }
        missing, unexpected = lit.model.load_state_dict(cleaned_sd, strict=False)
        if missing:
            print(f"  Missing keys  ({len(missing)}): "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"  Unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        del ckpt, raw_sd, cleaned_sd
        print("Weights loaded. Starting fresh optimiser for this stage.")

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{val_loss:.4f}",
    )
    lr_cb = LearningRateMonitor(logging_interval="step")
    logger = CSVLogger(save_dir=args.output_dir, name="logs")

    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        precision=args.precision,
        devices=args.devices,
        accelerator="auto",
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[ckpt_cb],
        logger=logger,
        log_every_n_steps=1,
        enable_progress_bar=args.tqdm,
    )

    trainer.fit(lit, datamodule=dm)


if __name__ == "__main__":
    main()

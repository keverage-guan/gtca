#!/bin/bash
# =============================================================================
# GTCA Stage 2: Structural module specialisation
#
# What trains:   Parse-tree encoder + cross-attention + gating modules only
# What freezes:  Backbone + LoRA adapters (weights carried from stage 1)
# Alpha:         Linearly warmed up 0 → 0.15 over first 10% of steps
# Data:          Same CLOTH + MMLU combined
#
# Before running: set STAGE1_CKPT below to the best checkpoint from stage 1.
#   e.g.  outputs/stage1/checkpoints/epoch=0-val_loss=0.8123.ckpt
# =============================================================================
#SBATCH --job-name=gtca_stage2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kg8941@princeton.edu
#SBATCH --output=../logs/gtca_stage2.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

module purge
module load anaconda3/2025.6
conda activate GTCA

mkdir -p ../logs outputs/stage2/checkpoints

export HF_HOME="$(realpath ../hf_cache)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# ---------------------------------------------------------------------------
# SET THIS to the best checkpoint printed at the end of stage 1.
# ---------------------------------------------------------------------------
STAGE1_CKPT=$(ls outputs/stage1/checkpoints/*.ckpt \
    | sed 's/.*val_loss=\([0-9.]*\).*/\1 &/' \
    | sort -n | head -1 | awk '{print $2}')

echo "[stage2] Resuming weights from: $STAGE1_CKPT"
# ---------------------------------------------------------------------------

echo "[stage2] Starting GTCA-only structural training..."

python -m scripts.train_gtca \
    --model_name_or_path ../../hf_cache/Qwen2.5-7B \
    --task          cloth \
    --data_path     data/cloth_mmlu \
    --cache_path    cache/cloth_mmlu_qwen.sqlite \
    --output_dir    outputs/stage2 \
    --ckpt_path     "$STAGE1_CKPT" \
    --train_stage   stage2 \
    --lr            3e-5 \
    --lora_r        16 \
    --lora_alpha    32 \
    --lora_dropout  0.05 \
    --alpha_max     0.15 \
    --alpha_warmup_ratio 0.10 \
    --batch_size    2 \
    --accumulate_grad_batches 8 \
    --max_length    1024 \
    --max_epochs    1 \
    --gradient_clip_val 1.0 \
    --devices       1 \
    --precision     bf16-mixed \
    --num_workers   7 \
    --tqdm

echo ""
echo "[stage2] Done. Best checkpoint (lowest val_loss):"
ls outputs/stage2/checkpoints/*.ckpt \
    | sed 's/.*val_loss=\([0-9.]*\).*/\1 &/' \
    | sort -n \
    | head -1 \
    | awk '{print $2}'
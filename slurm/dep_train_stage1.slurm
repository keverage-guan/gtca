#!/bin/bash
# =============================================================================
# GTCA Stage 1: LoRA-only task adaptation
#
# What trains:   LoRA adapters only (lora_r=16, lora_alpha=32)
# What freezes:  GTCA cross-attention + parse-tree encoder (alpha=0 throughout)
# Data:          CLOTH + MMLU combined (unified MCQA format)
# After this:    Pick best checkpoint (lowest val_loss), pass to stage 2
# =============================================================================
#SBATCH --job-name=dep_gtca_stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kg8941@princeton.edu
#SBATCH --output=../logs/dep_gtca_stage1.out
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80

module purge
module load anaconda3/2025.6
conda activate GTCA

mkdir -p ../logs dep_outputs/stage1/checkpoints

export HF_HOME="$(realpath ../hf_cache)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[stage1] Starting LoRA-only training..."

python -m scripts.train_gtca \
    --model_name_or_path ../../hf_cache/Qwen2.5-7B \
    --task          cloth \
    --data_path     data/cloth_mmlu \
    --cache_path    dep_cache/cloth_mmlu_qwen.sqlite \
    --output_dir    dep_outputs/stage1 \
    --train_stage   stage1 \
    --lr            5e-5 \
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
    --tqdm \
    --tree_type     dependency

# Print the best checkpoint so you can copy it to the stage2 script
echo ""
echo "[stage1] Done. Best checkpoint (lowest val_loss):"
ls outputs/stage1/checkpoints/*.ckpt \
    | sed 's/.*val_loss=\([0-9.]*\).*/\1 &/' \
    | sort -n \
    | head -1 \
    | awk '{print $2}'
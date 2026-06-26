#!/bin/bash
#SBATCH --job-name=parse_trees
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --mail-type=fail
#SBATCH --mail-user=kg8941@princeton.edu
#SBATCH --output=../logs/parse_dep_2.out
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2025.6
conda activate GTCA

export HF_HOME="$(realpath ../hf_models)"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

for TASK in hellaswag blimp cloth cola; do
# for TASK in mmlu winogrande; do
  echo "=== Processing: $TASK ==="

  python -m scripts.generate_tree_gtca \
    --model_name_or_path ../../hf_cache/Qwen2.5-7B \
    --task "$TASK" \
    --data_path "data/$TASK" \
    --cache_path "dep_cache/${TASK}_qwen.sqlite" \
    --spacy_model en_core_web_sm \
    --benepar_model benepar_en3 \
    --splits train,validation,test \
    --include_label_splits train,validation \
    --tree_type dependency
done

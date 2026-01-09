#!/bin/bash
# Run evaluation with SOLIDER pretrained model
#
# Usage:
#   ./scripts/run_eval_pretrained.sh market1501
#   ./scripts/run_eval_pretrained.sh msmt17
#   ./scripts/run_eval_pretrained.sh cctv_reid

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate conda environment if available
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate torch  # or your environment name
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate torch
fi

# Default dataset
DATASET=${1:-cctv_reid}

echo "========================================"
echo "SOLIDER Pretrained Model Evaluation"
echo "Dataset: $DATASET"
echo "========================================"

# Run evaluation
python evaluate_pretrained.py --config configs/eval_${DATASET}.yaml

echo "========================================"
echo "Evaluation complete!"
echo "========================================"

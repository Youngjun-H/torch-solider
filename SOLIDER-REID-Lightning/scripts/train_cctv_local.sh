#!/bin/bash
# Local training script for CCTV ReID dataset
# Usage: ./scripts/train_cctv_local.sh

echo "=========================================="
echo "CCTV ReID Local Training"
echo "=========================================="

# Activate conda environment
# source ~/miniconda3/bin/activate
# conda activate reid-env

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Adjust based on available GPUs
export PYTHONUNBUFFERED=1

# Create output directories
mkdir -p logs
mkdir -p outputs

# Run training
# For single GPU, Lightning will automatically use non-DDP mode
# For multi-GPU, Lightning will automatically use DDP

python scripts/train.py \
    --config-name base_config \
    model.name=swin_base_patch4_window7_224 \
    model.pretrain_path=/purestorage/AILAB/AI_2/yjhwang/work/reid/torch-solider/important_checkpoints/phase2/phase2.pth \
    data.dataset=cctv_reid \
    data.root_dir=/purestorage/AILAB/AI_2/datasets/PersonReID/cctv_reid_dataset_v2 \
    data.batch_size=64 \
    data.num_instances=4 \
    data.num_workers=8 \
    data.sampler=softmax_triplet \
    training.max_epochs=120 \
    training.precision="16-mixed" \
    training.eval_period=5 \
    training.devices="auto" \
    optimizer.lr=0.0008 \
    optimizer.weight_decay=0.0001 \
    scheduler.warmup_epochs=10 \
    output_dir=./outputs/cctv_reid_swin_base

echo "=========================================="
echo "Training completed!"
echo "=========================================="

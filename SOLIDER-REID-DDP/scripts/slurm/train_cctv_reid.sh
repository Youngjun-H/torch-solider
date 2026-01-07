#!/bin/bash
#SBATCH --job-name=solider-ddp
#SBATCH --nodelist=nv170,nv172,nv174,nv176
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --output=logs/train_%A.out

# ============================================================
# SOLIDER-REID Native DDP Training Script
# ============================================================

echo "=========================================="
echo "SOLIDER-REID DDP Training"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Num nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "GPUs per node: 8"
echo "=========================================="

# Create directories
mkdir -p logs
mkdir -p outputs

# Set Python path
PYTHON_BIN=/purestorage/AILAB/AI_2/yjhwang/work/reid/torch-solider/.venv/bin/python
echo "Using Python: $PYTHON_BIN"

# Set working directory
cd /purestorage/AILAB/AI_2/yjhwang/work/reid/torch-solider/SOLIDER-REID-DDP

# Environment variables for DDP
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export NCCL_DEBUG=INFO

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE"

# Run training with srun
# srun automatically sets SLURM_PROCID, SLURM_LOCALID for each task
srun $PYTHON_BIN train.py \
    --config configs/config.yaml

echo "=========================================="
echo "Training completed!"
echo "Check outputs at: ./outputs/"
echo "=========================================="

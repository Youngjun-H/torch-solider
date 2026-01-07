#!/bin/bash
#SBATCH --job-name=solider-reid
##SBATCH --partition=hopper
#SBATCH --nodelist=nv170
#SBATCH --nodes=1                    # 노드 수 (필요시 수정)
#SBATCH --gres=gpu:8                 # 노드당 GPU 수 (필요시 수정)
#SBATCH --ntasks-per-node=8          # 노드당 태스크 수 (보통 GPU 수와 동일)
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --output=logs/train_%A.out

# Print job information
echo "=========================================="
echo "CCTV ReID Training Job"
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: 4"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "=========================================="

mkdir -p logs
mkdir -p outputs

# Activate conda environment
# Try multiple possible conda installation locations
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate reid-env
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate reid-env
elif [ -f ~/miniconda3/bin/activate ]; then
    source ~/miniconda3/bin/activate
    conda activate reid-env
elif [ -f ~/anaconda3/bin/activate ]; then
    source ~/anaconda3/bin/activate
    conda activate reid-env
else
    echo "WARNING: Conda not found, using system Python"
fi

echo "Python: $(which python || echo 'python not found, using venv')"
echo "Python version: $(python --version 2>&1 || echo 'using venv python')"

# Use the venv python directly
PYTHON_BIN=/purestorage/AILAB/AI_2/yjhwang/work/reid/torch-solider/.venv/bin/python
echo "Using Python: $PYTHON_BIN"

srun $PYTHON_BIN scripts/train.py \
    --config-name base_config \
    --config-path ../configs \
    model.pretrain_path=/purestorage/AILAB/AI_2/yjhwang/work/reid/torch-solider/important_checkpoints/phase2/phase2.pth \
    training.eval_period=5 \
    scheduler.warmup_epochs=10 \
    output_dir=./outputs/cctv_reid_swin_base

echo "=========================================="
echo "Training completed!"
echo "Check outputs at: ./outputs/cctv_reid_swin_base"
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=solider-reid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4        # Number of GPUs
#SBATCH --cpus-per-task=8          # CPU cores per GPU
#SBATCH --gres=gpu:4               # Request 4 GPUs
#SBATCH --partition=gpu            # Partition name (change as needed)
#SBATCH --time=48:00:00            # Max runtime
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Print job information
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

# Load modules (adjust for your cluster)
# module load cuda/11.3
# module load cudnn/8.2

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate reid-env  # Change to your environment name

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_DEBUG=INFO
export PYTHONUNBUFFERED=1

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p logs
mkdir -p outputs

# Run training with Lightning 2.6+
# ✅ Lightning automatically detects SLURM environment variables:
#    - SLURM_NTASKS_PER_NODE (4) → automatically sets devices=4
#    - SLURM_JOB_NUM_NODES (1) → automatically sets num_nodes=1
#    - Automatically configures DDP strategy for multi-GPU training
# No need to manually specify devices/num_nodes when using strategy="auto"!
srun python scripts/train.py \
    model.name=swin_base_patch4_window7_224 \
    model.pretrain_path=/path/to/solider/checkpoint_tea.pth \
    data.dataset=market1501 \
    data.root_dir=/path/to/data \
    data.batch_size=64 \
    training.max_epochs=120 \
    training.precision="16-mixed" \
    output_dir=./outputs/market1501_swin_base

echo "=========================================="
echo "Training completed!"
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=solider-reid-multi
#SBATCH --nodes=4                  # Number of nodes
#SBATCH --ntasks-per-node=4        # GPUs per node
#SBATCH --cpus-per-task=8          # CPU cores per GPU
#SBATCH --gres=gpu:4               # Request 4 GPUs per node
#SBATCH --partition=gpu            # Partition name (change as needed)
#SBATCH --time=72:00:00            # Max runtime
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# Print job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node(s): $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total GPUs: $(($SLURM_JOB_NUM_NODES * $SLURM_NTASKS_PER_NODE))"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "=========================================="

# Load modules (adjust for your cluster)
# module load cuda/11.3
# module load cudnn/8.2
# module load nccl/2.10

# Activate conda environment
source ~/miniconda3/bin/activate
conda activate reid-env  # Change to your environment name

# Set environment variables for multi-node training
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export PYTHONUNBUFFERED=1

# Change to project directory
cd $SLURM_SUBMIT_DIR

# Create output directories
mkdir -p logs
mkdir -p outputs

# Calculate effective batch size
# Total GPUs = nodes * GPUs_per_node = 4 * 4 = 16
# Batch size per GPU = 64 / 16 = 4 (will be adjusted automatically by Lightning)
# Or increase total batch size for better utilization
TOTAL_BATCH_SIZE=256  # 256 / 16 GPUs = 16 per GPU

# Run training with Lightning 2.6+
# ✅ Lightning automatically detects SLURM environment:
#    - SLURM_NTASKS_PER_NODE=4 → automatically sets devices=4
#    - SLURM_JOB_NUM_NODES=4 → automatically sets num_nodes=4
#    - Total GPUs: 4 × 4 = 16 (auto-calculated)
#    - Automatically enables multi-node DDP with proper NCCL setup
# You can still override in config if needed, but auto-detection is recommended!
srun python scripts/train.py \
    model.name=swin_base_patch4_window7_224 \
    model.pretrain_path=/path/to/solider/checkpoint_tea.pth \
    data.dataset=msmt17 \
    data.root_dir=/path/to/data \
    data.batch_size=${TOTAL_BATCH_SIZE} \
    training.max_epochs=120 \
    training.precision="16-mixed" \
    output_dir=./outputs/msmt17_swin_base_multinode

echo "=========================================="
echo "Multi-node training completed!"
echo "=========================================="

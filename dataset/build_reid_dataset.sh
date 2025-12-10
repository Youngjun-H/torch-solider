#!/bin/bash
#SBATCH --job-name=build_reid_dataset
#SBATCH --nodelist=hpe161
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="reid_dataset_generation"
#SBATCH --output=build_reid_dataset_%A.log

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date
hostname -I;

srun python build_reid_dataset.py \
  --train_dir "/purestorage/AILAB/AI_4/datasets/cctv/image/preprocessed/2025-10-15" \
  --val_dir "/purestorage/AILAB/AI_4/datasets/cctv/image/preprocessed/2025-10-16" \
  --output_dir "/purestorage/AILAB/AI_2/datasets/PersonReID/cctv_reid_dataset" \
  --query_per_floor 2 \
  --seed 42


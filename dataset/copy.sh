#!/bin/bash
#SBATCH --job-name=build_dataset
#SBATCH --nodelist=hpe161
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="dataset_generation"
#SBATCH --output=copy_%A.log

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date
hostname -I;

srun python copy_images.py \
  --input_path "/purestorage/AILAB/AI_4/datasets/cctv/image/preprocessed/2025-10-15" \
  --output_path "/purestorage/AILAB/AI_2/datasets/PersonReID/solider_step_pre/images"
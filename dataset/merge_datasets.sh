#!/bin/bash
#SBATCH --job-name=merge_datasets
#SBATCH --nodelist=hpe161
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="dataset_merge"
#SBATCH --output=merge_datasets_%A.log

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date
hostname -I;

srun python merge_datasets.py \
  --existing_path "/purestorage/AILAB/AI_2/datasets/PersonReID/solider_surv_pre/images" \
  --new_data_path "/purestorage/AILAB/AI_4/datasets/cctv/image/preprocessed/2025-10-15" \
  --output_path "/purestorage/AILAB/AI_2/datasets/PersonReID/solider_surv_pre_v2/images"
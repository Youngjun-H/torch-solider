#!/bin/bash
#SBATCH --job-name=DINO-SOLIDER
#SBATCH --nodelist=cubox01
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=8G
#SBATCH --comment="dataset_generation"
#SBATCH --output=model_%A.log

##### Number of total processes
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Job name:= " "$SLURM_JOB_NAME"
echo "Nodelist:= " "$SLURM_JOB_NODELIST"
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

echo "Run started at:- "
date
hostname -I;

python -W ignore train_dino.py \
--arch swin_base \
--data_path /purestorage/AILAB/AI_2/datasets/PersonReID/solider_surv_pre/images \
--output_dir ./log/lup/dino_base \
--height 256 --width 128 \
--crop_height 128 --crop_width 64 \
--epochs 100 \
--batch_size_per_gpu 112 \
--num_workers 8 \
--global_crops_scale 0.8 1. \
--local_crops_scale 0.05 0.8 \
--devices 8 \
--precision bf16-mixed
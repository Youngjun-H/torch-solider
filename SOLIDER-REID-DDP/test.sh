#!/bin/bash
#SBATCH --job-name=solider-ddp
#SBATCH --nodelist=nv170,nv172,nv174,nv176
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=14
#SBATCH --mem=0
#SBATCH --output=logs/test_%A.out

srun python evaluate_pretrained.py --config configs/eval_cctv_reid.yaml

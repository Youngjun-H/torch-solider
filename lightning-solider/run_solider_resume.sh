#!/bin/bash

# SOLIDER 학습을 위한 PyTorch Lightning 실행 스크립트 (DINO 체크포인트에서 resume)
# DINO 모델을 먼저 학습한 후, 그 모델을 초기화로 사용하여 SOLIDER를 fine-tuning합니다.

cd "$(dirname "$0")" || exit

python -W ignore train_solider.py \
--arch swin_base \
--data_path /home/yjhwang/work/cctv-solider/data/1119/crops_raw_2/cam_0 \
--output_dir ./log/lup/solider_base \
--height 256 --width 128 \
--crop_height 128 --crop_width 64 \
--epochs 100 \
--batch_size_per_gpu 48 \
--global_crops_scale 0.8 1. \
--local_crops_scale 0.05 0.8 \
--partnum 3 \
--parthead_nlayers 3 \
--semantic_loss 1.0 \
--warmup_epochs 1 \
--lr 0.00005 \
--devices 4 \
--precision bf16-mixed \
--resume true \
--init_model ./log/lup/dino_base/checkpoint.pth


#!/bin/bash

# DINO 학습을 위한 PyTorch Lightning 실행 스크립트
# 기존 run_dino.sh와 동일한 인자를 사용하되, Lightning의 자동 분산 처리를 활용합니다.

cd "$(dirname "$0")" || exit

python -W ignore train_dino.py \
--arch swin_base \
--data_path /home/yjhwang/work/cctv-solider/data/1119/crops_raw_2/cam_0 \
--output_dir ./log/lup/dino_base \
--height 256 --width 128 \
--crop_height 128 --crop_width 64 \
--epochs 100 \
--batch_size_per_gpu 48 \
--global_crops_scale 0.8 1. \
--local_crops_scale 0.05 0.8 \
--devices 4 \
--precision bf16-mixed
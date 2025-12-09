#!/bin/bash
# SOLIDER-REID Lightning Training Script

CUDA_VISIBLE_DEVICES=0 python train_reid.py \
    --transformer_type swin_base_patch4_window7_224 \
    --pretrain_path path/to/SOLIDER/log/lup/swin_base/checkpoint_tea.pth \
    --pretrain_choice self \
    --semantic_weight 0.2 \
    --dataset_name msmt17 \
    --root_dir path/to/msmt17/datasets \
    --base_lr 0.0002 \
    --optimizer_name SGD \
    --max_epochs 120 \
    --warmup_epochs 20 \
    --warmup_method cosine \
    --weight_decay 1e-4 \
    --ims_per_batch 64 \
    --num_instance 4 \
    --sampler softmax_triplet \
    --metric_loss_type triplet \
    --no_margin \
    --id_loss_weight 1.0 \
    --triplet_loss_weight 1.0 \
    --eval_period 10 \
    --checkpoint_period 120 \
    --log_period 20 \
    --output_dir ./log/msmt17/swin_base \
    --devices 1 \
    --precision 16-mixed


#!/usr/bin/env python
"""
Training script for SOLIDER-REID with PyTorch Lightning 2.6+

✅ Updated to Lightning 2.6+ best practices:
- import lightning as L
- precision="16-mixed" format
- strategy="auto" for SLURM auto-detection
- Explicit device/accelerator settings

Supports:
- Single/Multi GPU training
- SLURM cluster training (automatic detection)
- Automatic DDP setup
- Mixed precision training
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import lightning as L  # ✅ Lightning 2.6+ import
from lightning.pytorch.callbacks import (  # ✅ Lightning 2.6+ callbacks import
    ModelCheckpoint,
    LearningRateMonitor,
    RichProgressBar,
    TQDMProgressBar
)
from lightning.pytorch.loggers import WandbLogger  # ✅ Lightning 2.6+ WandB logger
from lightning.pytorch.strategies import DDPStrategy  # ✅ DDP strategy for find_unused_parameters
import hydra
from omegaconf import DictConfig, OmegaConf

from dotenv import load_dotenv

from models.lightning_model import ReIDLightningModule
from data.datamodule import ReIDDataModule
from callbacks.reid_eval import ReIDEvaluationCallback


@hydra.main(version_base=None, config_path="../configs", config_name="base_config")
def main(cfg: DictConfig):
    """
    ✅ Lightning 2.6+ Main training function

    Automatically detects SLURM environment and configures DDP
    """

    # Print config
    print("=" * 80)
    print("Training Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # ✅ Lightning 2.6+: Set seed for reproducibility
    L.seed_everything(cfg.training.seed, workers=True)

    # Initialize DataModule
    datamodule = ReIDDataModule(
        dataset_name=cfg.data.dataset,
        data_root=cfg.data.root_dir,
        batch_size=cfg.data.batch_size,
        num_instances=cfg.data.num_instances,
        num_workers=cfg.data.num_workers,
        sampler=cfg.data.sampler,
        img_size_train=tuple(cfg.data.img_size_train),
        img_size_test=tuple(cfg.data.img_size_test),
        random_flip_prob=cfg.data.augmentation.random_flip_prob,
        random_erase_prob=cfg.data.augmentation.random_erase_prob,
        padding=cfg.data.augmentation.padding,
        pixel_mean=tuple(cfg.data.augmentation.pixel_mean),
        pixel_std=tuple(cfg.data.augmentation.pixel_std),
    )

    # Setup to get dataset info
    datamodule.setup('fit')

    # Initialize Model
    model = ReIDLightningModule(
        # Model config
        model_name=cfg.model.name,
        num_classes=datamodule.num_classes,
        pretrain_path=cfg.model.pretrain_path,
        semantic_weight=cfg.model.semantic_weight,

        # Training config
        img_size=tuple(cfg.data.img_size_train),
        drop_path_rate=cfg.model.drop_path_rate,
        drop_rate=cfg.model.drop_rate,
        attn_drop_rate=cfg.model.attn_drop_rate,

        # Loss config
        id_loss_weight=cfg.loss.id_loss_weight,
        triplet_loss_weight=cfg.loss.triplet_loss_weight,
        triplet_margin=cfg.loss.triplet_margin,
        label_smoothing=cfg.loss.label_smoothing,

        # Optimizer config
        optimizer_name=cfg.optimizer.name,
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        momentum=cfg.optimizer.momentum,
        bias_lr_factor=cfg.optimizer.bias_lr_factor,

        # Scheduler config
        scheduler_name=cfg.scheduler.name,
        warmup_epochs=cfg.scheduler.warmup_epochs,
        max_epochs=cfg.training.max_epochs,
        steps=tuple(cfg.scheduler.steps),
        gamma=cfg.scheduler.gamma,

        # Other
        neck=cfg.model.neck,
        neck_feat=cfg.model.neck_feat,
    )

    # Setup callbacks
    callbacks = []

    # Model checkpoint - save best models
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(cfg.output_dir, 'checkpoints'),
        filename='epoch={epoch:03d}-mAP={val/mAP:.4f}',
        monitor='val/mAP',
        mode='max',
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Progress bar
    if cfg.training.get('use_rich_progress', False):
        callbacks.append(RichProgressBar())
    else:
        callbacks.append(TQDMProgressBar(refresh_rate=20))

    # ReID evaluation callback
    reid_eval_callback = ReIDEvaluationCallback(
        num_query=datamodule.num_query,
        eval_period=cfg.training.eval_period,
        feat_norm=cfg.model.feat_norm,
    )
    callbacks.append(reid_eval_callback)

    # Setup WandB logger
    logger = WandbLogger(
        project=cfg.logging.wandb_project,
        name=cfg.logging.experiment_name,
        save_dir=cfg.output_dir,
        log_model=True,  # Log model checkpoints to WandB
    )

    # ✅ Lightning 2.6+: Initialize Trainer with updated API
    trainer = L.Trainer(
        # ✅ Compute (auto-detection for SLURM)
        accelerator='auto',  # ✅ Auto-detect GPU/CPU
        devices='auto',      # ✅ Auto-detect available devices (SLURM aware)
        num_nodes=cfg.training.get('num_nodes', 1),
        # ✅ DDP with find_unused_parameters=True for SOLIDER model
        # Some parameters (e.g., semantic branch) may not be used in every training step
        strategy=DDPStrategy(find_unused_parameters=True),

        # Training
        max_epochs=cfg.training.max_epochs,
        precision=cfg.training.precision,  # ✅ Must be string: "16-mixed", "32-true", etc.
        gradient_clip_val=cfg.training.get('gradient_clip_val', 0.0),
        accumulate_grad_batches=cfg.training.get('accumulate_grad_batches', 1),

        # ✅ SyncBatchNorm for DDP - prevents batch size 1 errors
        sync_batchnorm=True,

        # Logging & Checkpointing
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.training.get('log_period', 50),

        # Validation
        check_val_every_n_epoch=cfg.training.get('eval_period', 10),

        # Performance
        benchmark=True,
        deterministic=False,
    )

    # Print training info
    print("\n" + "=" * 80)
    print("Training Information:")
    print(f"  Dataset: {cfg.data.dataset}")
    print(f"  Number of classes: {datamodule.num_classes}")
    print(f"  Number of cameras: {datamodule.num_cameras}")
    print(f"  Number of training samples: {len(datamodule.train_dataset)}")
    print(f"  Number of query samples: {datamodule.num_query}")
    print(f"  Number of gallery samples: {len(datamodule.val_dataset) - datamodule.num_query}")
    print(f"  Batch size: {cfg.data.batch_size}")
    print(f"  Number of GPUs: {trainer.num_devices}")
    print(f"  Number of nodes: {trainer.num_nodes}")
    print(f"  World size: {trainer.world_size}")
    print(f"  Effective batch size: {cfg.data.batch_size * trainer.world_size}")
    print(f"  Output directory: {cfg.output_dir}")
    print("=" * 80 + "\n")

    # Train
    trainer.fit(model, datamodule=datamodule)

    # Test on best model
    print("\n" + "=" * 80)
    print("Testing best model...")
    print("=" * 80 + "\n")

    trainer.test(model, datamodule=datamodule, ckpt_path='best')

    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best mAP: {checkpoint_callback.best_model_score:.4f}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    load_dotenv()
    main()

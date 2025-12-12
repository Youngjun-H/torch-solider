"""Training script for SOLIDER-REID with Lightning."""

import argparse
import random
from pathlib import Path

import lightning as L
import numpy as np
import torch
from config import get_args_parser, parse_args_to_config
from data.datamodule import ReIDDataModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from module import ReIDLightningModule


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    parser = get_args_parser()
    args = parser.parse_args()
    config = parse_args_to_config(args)

    # Set seed
    set_seed(config.seed)

    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    # Logger
    logger = WandbLogger(
        project="solider-reid",
        name=f"{config.dataset_name}_{config.model_name}",
        save_dir=config.output_dir,
    )

    # Data module
    datamodule = ReIDDataModule(config)

    # Model
    model = ReIDLightningModule(config)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(config.output_dir) / "checkpoints",
        filename="reid-{epoch:02d}-{val_mAP:.4f}",
        monitor=config.monitor,
        mode=config.monitor_mode,
        save_top_k=config.save_top_k,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Trainer
    trainer = L.Trainer(
        accelerator="gpu",
        devices=config.devices,
        num_nodes=config.num_nodes,
        strategy=(
            "ddp_find_unused_parameters_true"
            if config.devices and config.devices > 1
            else "auto"
        ),
        precision=config.precision,
        max_epochs=config.max_epochs,
        accumulate_grad_batches=config.accumulate_grad_batches,
        log_every_n_steps=config.log_every_n_steps,
        val_check_interval=config.val_check_interval,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        sync_batchnorm=True if config.devices and config.devices > 1 else False,
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    main()

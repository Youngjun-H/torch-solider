"""ReID Training Script with Lightning."""

import argparse
import datetime
import random
import sys
from pathlib import Path

import lightning as L
import numpy as np
import torch
import wandb
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


class SavePTHCallback(Callback):
    """Save model as .pth file in addition to Lightning checkpoint."""

    def __init__(self, output_dir, save_period=120):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.save_period = save_period
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        """Save .pth file at the end of each epoch if it's a checkpoint period."""
        if (trainer.current_epoch + 1) % self.save_period == 0:
            model_name = (
                pl_module.args.transformer_type
                if hasattr(pl_module.args, "transformer_type")
                else "model"
            )
            save_path = (
                self.output_dir / f"{model_name}_{trainer.current_epoch + 1}.pth"
            )
            torch.save(pl_module.model.state_dict(), save_path)
            print(f"Saved model state_dict to {save_path}")


# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from config.reid_args import get_args_parser
from reid_data.reid_data_module import ReIDDataModule
from reid_module.reid_lightning_module import ReIDLightningModule


def set_seed(seed):
    """Set random seed."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def main():
    parser = argparse.ArgumentParser("ReID Training", parents=[get_args_parser()])
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # wandb 설정
    wandb_logger = WandbLogger(
        project="solider-reid",
        name=f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    # Create data module
    dm = ReIDDataModule(args)
    dm.setup()

    # Create model (pass datamodule reference for early access to dataset info)
    model = ReIDLightningModule(args, datamodule=dm)

    # Callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")

    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.output_dir) / "checkpoints",
        filename="reid-{epoch:02d}-{val_mAP:.4f}",
        monitor="val_mAP",
        mode="max",
        save_top_k=3,
        every_n_epochs=args.checkpoint_period,
    )

    # Save .pth files
    save_pth_callback = SavePTHCallback(
        output_dir=Path(args.output_dir) / "checkpoints",
        save_period=args.checkpoint_period,
    )

    # Trainer
    devices = args.devices if args.devices is not None else "auto"

    trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=args.num_nodes,
        strategy="ddp" if devices != 1 and devices != "auto" else "auto",
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[lr_monitor, checkpoint_callback, save_pth_callback],
        logger=wandb_logger,
        sync_batchnorm=True,
        log_every_n_steps=args.log_period,
        check_val_every_n_epoch=args.eval_period,
        benchmark=True,
    )

    # Train
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    wandb.login(key="53f960c86b81377b89feb5d30c90ddc6c3810d3a")
    main()

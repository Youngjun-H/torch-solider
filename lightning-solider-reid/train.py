"""ReID Training Script with Lightning."""

import argparse
import datetime
import os
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
        log_model=False,  # 모델 체크포인트는 별도로 저장
    )

    # Create data module
    # 주의: Lightning의 표준 방식은 trainer.fit()이 자동으로 setup()을 호출하는 것입니다.
    # DDP 환경에서 수동으로 setup()을 호출하면 동기화 문제가 발생할 수 있습니다.
    # 따라서 수동 호출을 제거하고 Lightning이 자동으로 호출하도록 합니다.
    dm = ReIDDataModule(args)

    # Create model (pass datamodule reference for early access to dataset info)
    # 모델 초기화는 datamodule 정보 없이도 가능하도록 수정됨 (setup()에서 처리)
    # model.setup()도 Lightning이 자동으로 호출하므로 수동 호출하지 않음
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
    # SLURM 환경에서 자동 감지하거나 argument로 전달된 값 사용
    if args.devices is not None:
        devices = args.devices
    else:
        # SLURM 환경 변수 확인
        if "SLURM_NTASKS_PER_NODE" in os.environ:
            devices = int(os.environ["SLURM_NTASKS_PER_NODE"])
        else:
            devices = "auto"

    # num_nodes 설정
    if args.num_nodes is not None and args.num_nodes > 1:
        num_nodes = args.num_nodes
    else:
        # SLURM 환경 변수 확인
        if "SLURM_NNODES" in os.environ:
            num_nodes = int(os.environ["SLURM_NNODES"])
        else:
            num_nodes = 1

    print("Training configuration:")
    print(f"  - Devices: {devices}")
    print(f"  - Nodes: {num_nodes}")
    total_gpus = num_nodes * (devices if isinstance(devices, int) else 1)
    print(f"  - Total GPUs: {total_gpus}")
    print(f"  - Accumulate grad batches: {args.accumulate_grad_batches}")
    effective_batch_size = (
        total_gpus
        * args.accumulate_grad_batches
        * (args.ims_per_batch // total_gpus if total_gpus > 0 else args.ims_per_batch)
    )
    print(
        f"  - Effective batch size: {effective_batch_size} (world_size={total_gpus} * accumulate={args.accumulate_grad_batches} * mini_batch_size={args.ims_per_batch // total_gpus if total_gpus > 0 else args.ims_per_batch})"
    )

    # Note: We use custom DDP samplers (StratifiedIdentitySampler_DDP, etc.)
    # which already handle DDP distribution. When using batch_sampler in DataLoader,
    # Lightning automatically does not add DistributedSampler, so no need to disable it.
    # For validation, all ranks process the same data (validation_step handles this).
    #
    # Use find_unused_parameters=True because:
    # 1. ReID model may have conditional parameters (camera_num=0, view_num=0)
    # 2. center_criterion may not be used in all training steps
    # 3. Some model parameters may be conditionally used based on input
    trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=num_nodes,
        strategy="ddp_find_unused_parameters_true",  # Enable unused parameter detection
        precision=args.precision,
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,  # Gradient accumulation
        callbacks=[lr_monitor, checkpoint_callback, save_pth_callback],
        logger=wandb_logger,
        sync_batchnorm=True,
        log_every_n_steps=args.log_period,
        check_val_every_n_epoch=args.eval_period,
        benchmark=True,
        num_sanity_val_steps=0,  # Skip sanity check to avoid DDP issues
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # Train
    # Lightning 표준: trainer.fit()이 자동으로 다음을 호출합니다:
    # 1. dm.setup(stage="fit") - 데이터셋 준비
    # 2. model.setup(stage="fit") - 모델 초기화
    # 3. dm.train_dataloader() - 학습 데이터로더 생성
    # 4. dm.val_dataloader() - 검증 데이터로더 생성
    # DDP 환경에서는 모든 프로세스가 자동으로 동기화됩니다.
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    wandb.login(key="53f960c86b81377b89feb5d30c90ddc6c3810d3a")
    main()

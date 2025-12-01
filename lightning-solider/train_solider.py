"""SOLIDER 학습 스크립트."""

import argparse
import datetime
import sys
from pathlib import Path

import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# lightning-solider 루트 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.args import get_args_parser
from solider.solider_data_module import SOLIDERDataModule
from solider.solider_lightning_module import SOLIDERLightningModule


def main():
    parser = argparse.ArgumentParser(
        "SOLIDER PyTorch Lightning", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    # wandb 설정
    wandb_logger = WandbLogger(
        project="dino-solider",
        name=f"phase2_batchsize_{args.batch_size_per_gpu}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.data_path}",
    )

    # 데이터 모듈 생성
    dm = SOLIDERDataModule(args)

    # 모델 생성
    model = SOLIDERLightningModule(args)

    # Resume from DINO checkpoint if specified
    if args.resume and args.init_model:
        print(f"Resuming from DINO checkpoint: {args.init_model}")
        model.load_from_dino_checkpoint(args.init_model)

    # Learning Rate Monitor 설정
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Checkpoint Callback 설정
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="solider-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        every_n_epochs=args.saveckp_freq,
    )

    # Trainer 설정
    devices = args.devices if args.devices is not None else "auto"

    trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,
        num_nodes=args.num_nodes,
        strategy="ddp_find_unused_parameters_true",  # SOLIDER는 Teacher 파라미터가 grad 계산에서 제외되므로 필요
        precision=args.precision,
        max_epochs=args.epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=wandb_logger,
        sync_batchnorm=True,  # 멀티 GPU간 BatchNorm 동기화
        use_distributed_sampler=True,  # DDP 환경에서 자동으로 DistributedSampler 사용
        log_every_n_steps=10,
        benchmark=True,  # cudnn.benchmark = True
    )

    device_info = (
        f"{devices} GPU(s)" if isinstance(devices, int) else "auto-detected GPU(s)"
    )
    print(f"Starting SOLIDER training with PyTorch Lightning on {device_info}...")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    wandb.login(key="your_wandb_api_key")
    main()

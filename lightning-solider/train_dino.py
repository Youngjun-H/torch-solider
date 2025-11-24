"""DINO 학습 스크립트."""

import argparse
import sys
from pathlib import Path

import lightning as L
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# lightning-solider 루트 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config.args import get_args_parser
from dino.dino_data_module import DINODataModule
from dino.dino_lightning_module import DINOLightningModule


def main():
    parser = argparse.ArgumentParser(
        "DINO PyTorch Lightning", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    # wandb 설정
    wandb_logger = WandbLogger(project="dino-solider")

    # 데이터 모듈 생성
    dm = DINODataModule(args)

    # 모델 생성
    model = DINOLightningModule(args)

    # Learning Rate Monitor 설정
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Checkpoint Callback 설정
    # 최상위 모델을 저장하고, 주기적으로 체크포인트를 저장합니다.
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        filename="dino-{epoch:02d}-{train_loss:.2f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        every_n_epochs=args.saveckp_freq,
    )

    # Trainer 설정
    # devices가 지정되지 않으면 자동으로 사용 가능한 GPU 감지
    devices = args.devices if args.devices is not None else "auto"

    trainer = L.Trainer(
        accelerator="gpu",
        devices=devices,
        strategy="ddp_find_unused_parameters_true",  # DINO는 Teacher 파라미터가 grad 계산에서 제외되므로 필요할 수 있음
        precision=args.precision,  # args에서 precision 설정 사용
        max_epochs=args.epochs,
        callbacks=[lr_monitor, checkpoint_callback],
        logger=wandb_logger,
        sync_batchnorm=True,  # DINO 필수: 멀티 GPU간 BatchNorm 동기화
        use_distributed_sampler=True,  # DDP 환경에서 자동으로 DistributedSampler 사용
        log_every_n_steps=10,
        benchmark=True,  # cudnn.benchmark = True (입력 크기가 고정된 경우 성능 향상)
        # gradient_clip_val=args.clip_grad if args.clip_grad > 0 else None,
    )

    device_info = (
        f"{devices} GPU(s)" if isinstance(devices, int) else "auto-detected GPU(s)"
    )
    print(f"Starting DINO training with PyTorch Lightning on {device_info}...")
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()

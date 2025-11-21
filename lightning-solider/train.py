import argparse

import lightning as L
from args import get_args_parser
from dino_data_modeule import DINODataModule
from dino_lightining_module import DINO
from lightning.pytorch.callbacks import ModelCheckpoint


def main():
    parser = argparse.ArgumentParser(
        "DINO PyTorch Lightning", parents=[get_args_parser()]
    )
    args = parser.parse_args()

    # 데이터 모듈 생성
    dm = DINODataModule(args)

    # 모델 생성
    model = DINO(args)

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
        callbacks=[checkpoint_callback],
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

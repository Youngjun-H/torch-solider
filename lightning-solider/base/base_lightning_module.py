"""Base LightningModule for DINO and SOLIDER."""
import math
import sys
from pathlib import Path

import lightning as L
import torch
import torch.optim as optim

# lightning-solider 루트 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import utils


class BaseDINOLightningModule(L.LightningModule):
    """
    DINO와 SOLIDER의 공통 기능을 담은 Base 클래스.
    
    공통 기능:
    - 모델 아키텍처 빌드 (하위 클래스에서 구현)
    - Optimizer 설정
    - Scheduler 생성 및 업데이트
    - 공통 학습 로직
    """

    def __init__(self, args):
        super().__init__()
        # args가 객체인 경우 dict로 변환하여 저장
        if hasattr(args, "__dict__"):
            self.save_hyperparameters(vars(args))
        else:
            self.save_hyperparameters(args)
        self.args = args

        # 내부 변수 (Schedules 등)
        self.automatic_optimization = (
            False  # DINO는 수동 최적화가 더 적합 (Custom Schedulers 때문)
        )
        
        # Schedules는 on_fit_start에서 초기화
        self.lr_schedule = None
        self.wd_schedule = None
        self.momentum_schedule = None
        self.iters_per_epoch = None

    def _build_models(self):
        """
        모델 아키텍처를 구축합니다.
        하위 클래스에서 구현해야 합니다.
        """
        raise NotImplementedError("Subclasses must implement _build_models()")

    def on_fit_start(self):
        """
        학습 시작 시 스케줄러를 생성합니다.
        데이터로더 길이를 안전하게 가져와서 cosine scheduler를 생성합니다.
        """
        # 데이터로더 길이를 안전하게 가져오기
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            train_dataloader = self.trainer.datamodule.train_dataloader()
            if train_dataloader is not None:
                self.iters_per_epoch = len(train_dataloader)
            else:
                raise RuntimeError("Cannot get train_dataloader from datamodule")
        elif hasattr(self.trainer, "train_dataloader"):
            train_dataloader = self.trainer.train_dataloader
            if train_dataloader is not None:
                if callable(train_dataloader):
                    train_dataloader = train_dataloader()
                self.iters_per_epoch = len(train_dataloader)
            else:
                raise RuntimeError("Cannot get train_dataloader from trainer")
        elif (
            hasattr(self.trainer, "num_training_batches")
            and self.trainer.num_training_batches is not None
            and not (
                isinstance(self.trainer.num_training_batches, float)
                and math.isinf(self.trainer.num_training_batches)
            )
        ):
            self.iters_per_epoch = int(self.trainer.num_training_batches)
        else:
            raise RuntimeError(
                "Cannot determine number of training batches. "
                "Please ensure datamodule is properly set up."
            )

        # 안전성 체크
        if self.iters_per_epoch <= 0 or (
            isinstance(self.iters_per_epoch, float) and math.isinf(self.iters_per_epoch)
        ):
            raise RuntimeError(
                f"Invalid iters_per_epoch: {self.iters_per_epoch}. "
                "Please check your dataloader configuration."
            )

        print(f"Training batches per epoch: {self.iters_per_epoch}")

        # Cosine Schedulers 생성
        self.lr_schedule = utils.cosine_scheduler(
            self.args.lr
            * (self.args.batch_size_per_gpu * self.trainer.world_size)
            / 256.0,
            self.args.min_lr,
            self.args.epochs,
            self.iters_per_epoch,
            warmup_epochs=self.args.warmup_epochs,
        )
        self.wd_schedule = utils.cosine_scheduler(
            self.args.weight_decay,
            self.args.weight_decay_end,
            self.args.epochs,
            self.iters_per_epoch,
        )
        self.momentum_schedule = utils.cosine_scheduler(
            self.args.momentum_teacher, 1, self.args.epochs, self.iters_per_epoch
        )

    def configure_optimizers(self):
        """공통 optimizer 설정"""
        params_groups = utils.get_params_groups(self.student)
        if self.args.optimizer == "adamw":
            optimizer = optim.AdamW(params_groups)
        elif self.args.optimizer == "sgd":
            optimizer = optim.SGD(params_groups, lr=0, momentum=0.9)
        elif self.args.optimizer == "lars":
            optimizer = utils.LARS(params_groups)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
        return optimizer

    def _update_schedulers(self, optimizer):
        """
        매 스텝마다 LR, WD를 업데이트합니다.
        
        Args:
            optimizer: 현재 optimizer
        """
        it = self.global_step
        if it >= len(self.lr_schedule):
            it = len(self.lr_schedule) - 1  # Safety check

        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = self.lr_schedule[it]
            if i == 0:  # Only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[it]

    def _update_teacher_ema(self):
        """
        Teacher 모델을 EMA로 업데이트합니다.
        하위 클래스에서 student와 teacher를 정의해야 합니다.
        """
        it = self.global_step
        if it >= len(self.momentum_schedule):
            it = len(self.momentum_schedule) - 1

        with torch.no_grad():
            m = self.momentum_schedule[it]
            for param_q, param_k in zip(
                self.student.parameters(), self.teacher.parameters()
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

    def _apply_gradient_clipping(self, optimizer):
        """Gradient clipping을 적용합니다."""
        if self.args.clip_grad > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.args.clip_grad,
                gradient_clip_algorithm="norm",
            )

    def _cancel_gradients_last_layer(self):
        """Early epoch에서 마지막 레이어의 gradient를 취소합니다."""
        utils.cancel_gradients_last_layer(
            self.current_epoch, self.student, self.args.freeze_last_layer
        )


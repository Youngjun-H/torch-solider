"""DINO LightningModule."""

import sys
from pathlib import Path

import lightning as L
import torch
from torchvision import models as torchvision_models

# lightning-solider 루트 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from base.base_lightning_module import BaseDINOLightningModule
from models import swin_transformer as swin
from models import vision_transformer as vits
from models.vision_transformer import DINOHead
from shared import utils

from dino.dino_criterion import DINOLoss


class DINOLightningModule(BaseDINOLightningModule):
    """
    DINO 학습을 위한 PyTorch Lightning Module.

    BaseDINOLightningModule을 상속받아 DINO 특화 기능을 구현합니다.
    """

    def __init__(self, args):
        super().__init__(args)

        # 1. 모델 아키텍처 빌드 (Student & Teacher)
        self._build_models()

        # 2. Loss 함수 초기화
        self.dino_loss = DINOLoss(
            args.out_dim,
            args.local_crops_number + 2,  # Total crops
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        )

    def _build_models(self):
        """DINO 모델 아키텍처를 구축합니다."""
        args = self.args

        # Backbone 선택 로직
        # Vision Transformer (ViT) 아키텍처
        if args.arch in vits.__dict__.keys():
            student = vits.__dict__[args.arch](
                img_size=(args.height, args.width),
                patch_size=args.patch_size,
                drop_path_rate=0.1,  # stochastic depth
            )
            teacher = vits.__dict__[args.arch](
                img_size=(args.height, args.width), patch_size=args.patch_size
            )
            embed_dim = student.embed_dim
        # Swin Transformer 아키텍처
        elif args.arch == "swin_base":
            student = swin.swin_base_patch4_window7_224(
                img_size=(args.height, args.width),
                drop_path_rate=0.1,  # stochastic depth
            )
            teacher = swin.swin_base_patch4_window7_224(
                img_size=(args.height, args.width)
            )
            embed_dim = student.num_features[-1]
        elif args.arch == "swin_small":
            student = swin.swin_small_patch4_window7_224(
                img_size=(args.height, args.width),
                drop_path_rate=0.1,  # stochastic depth
            )
            teacher = swin.swin_small_patch4_window7_224(
                img_size=(args.height, args.width)
            )
            embed_dim = student.num_features[-1]
        elif args.arch == "swin_tiny":
            student = swin.swin_tiny_patch4_window7_224(
                img_size=(args.height, args.width),
                drop_path_rate=0.1,  # stochastic depth
            )
            teacher = swin.swin_tiny_patch4_window7_224(
                img_size=(args.height, args.width)
            )
            embed_dim = student.num_features[-1]
        # Torchvision 모델 (ResNet, EfficientNet 등)
        elif args.arch in torchvision_models.__dict__.keys():
            student = torchvision_models.__dict__[args.arch]()
            teacher = torchvision_models.__dict__[args.arch]()
            embed_dim = student.fc.weight.shape[1]
        else:
            raise ValueError(f"Unknown architecture: {args.arch}")

        # Wrapper & Head 적용
        self.student = utils.MultiCropWrapper(
            student,
            DINOHead(
                embed_dim,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            ),
        )
        self.teacher = utils.MultiCropWrapper(
            teacher,
            DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        )

        # Teacher 초기화 (Student와 동일하게 시작, Gradient 계산 안 함)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

    def training_step(self, batch, batch_idx):
        """DINO 학습 스텝."""
        opt = self.optimizers()
        images, _ = batch  # Label은 무시

        # 1. Update Hyperparameters (LR, WD) manually per step
        self._update_schedulers(opt)

        # 2. Forward Pass
        # Teacher: Global Crops (first 2) only
        with torch.no_grad():
            teacher_output = self.teacher(images[:2])

        # Student: All crops
        student_output = self.student(images)

        # 3. Loss Calculation
        loss = self.dino_loss(student_output, teacher_output, self.current_epoch)

        if not torch.isfinite(loss):
            self.print(f"Loss is {loss}, stopping training")
            if hasattr(self.trainer, "should_stop"):
                self.trainer.should_stop = True
            return None

        # 4. Backward & Optimization (Manual)
        opt.zero_grad()
        self.manual_backward(loss)

        # Gradient Clipping
        self._apply_gradient_clipping(opt)

        # Cancel gradients for last layer (Freeze strategy in early epochs)
        self._cancel_gradients_last_layer()

        opt.step()

        # 5. EMA Update (Teacher)
        self._update_teacher_ema()

        # 6. Logging - loss.item()로 스칼라만 전달하여 메모리 절약
        loss_value = loss.item()
        self.log(
            "train_loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log("lr", opt.param_groups[0]["lr"], on_step=True, logger=True)
        self.log("wd", opt.param_groups[0]["weight_decay"], on_step=True, logger=True)

        # 7. 명시적 메모리 해제 (메모리 누수 방지)
        del student_output, teacher_output

        # 주기적으로 메모리 캐시 정리 (매 100 스텝마다)
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

        # Manual optimization 모드에서는 Tensor를 반환해야 함 (detach로 계산 그래프 분리)
        return loss.detach()

    def compute_feature_variance(self, dataloader, num_batches=5):
        """
        Student backbone의 feature variance를 계산합니다.
        Projection head 이전의 encoder output을 사용합니다.

        Args:
            dataloader: 데이터로더
            num_batches: 계산에 사용할 배치 수

        Returns:
            mean_var: 평균 feature variance
        """
        self.eval()
        vars_list = []

        # 모델의 device 확인
        device = next(self.parameters()).device

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                images, _ = batch  # Label은 무시
                # 첫 번째 global crop만 사용 (일관성을 위해)
                if isinstance(images, list):
                    images = images[0]

                # 이미지를 모델과 같은 device로 이동
                images = images.to(device)

                # Student backbone의 encoder output (projection head 이전)
                # Backbone은 (output, _) 형태 또는 단일 output을 반환할 수 있음
                backbone_output = self.student.backbone(images)
                if isinstance(backbone_output, tuple):
                    feats = backbone_output[0]  # 첫 번째 출력이 encoder output
                else:
                    feats = backbone_output

                # Feature variance 계산: 각 feature dimension에 대한 variance의 평균
                var_dim = feats.var(dim=0).mean().item()
                vars_list.append(var_dim)

        self.train()  # 다시 training 모드로 전환
        return sum(vars_list) / len(vars_list) if len(vars_list) > 0 else 0.0

    def on_train_epoch_end(self):
        """
        Epoch 종료 시 feature variance를 계산하고 로깅합니다.
        """
        # 매 epoch마다 feature variance 계산
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            train_dataloader = self.trainer.datamodule.train_dataloader()
            if train_dataloader is not None:
                mean_var = self.compute_feature_variance(
                    train_dataloader, num_batches=5
                )
                self.print(
                    f"[Epoch {self.current_epoch}] FeatureVariance = {mean_var:.4f}"
                )
                self.log(
                    "feature_variance",
                    mean_var,
                    on_epoch=True,
                    logger=True,
                    sync_dist=True,
                )

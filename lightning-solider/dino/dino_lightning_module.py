"""DINO LightningModule."""
import sys
from pathlib import Path

import lightning as L
import swin_transformer as swin
import torch
import utils
import vision_transformer as vits
from torchvision import models as torchvision_models
from vision_transformer import DINOHead

# lightning-solider 루트 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from base.base_lightning_module import BaseDINOLightningModule
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

        # 6. Logging
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log("lr", opt.param_groups[0]["lr"], on_step=True, logger=True)
        self.log("wd", opt.param_groups[0]["weight_decay"], on_step=True, logger=True)

        return loss


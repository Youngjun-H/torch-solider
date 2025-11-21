import math

import lightning as L
import swin_transformer as swin
import torch
import torch.optim as optim
import utils
import vision_transformer as vits
from criterion import DINOLoss
from torchvision import models as torchvision_models
from vision_transformer import DINOHead


class DINO(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        # args가 객체인 경우 dict로 변환하여 저장
        if hasattr(args, "__dict__"):
            self.save_hyperparameters(vars(args))
        else:
            self.save_hyperparameters(args)
        self.args = args

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

        # 3. 내부 변수 (Schedules 등)
        self.automatic_optimization = (
            False  # DINO는 수동 최적화가 더 적합 (Custom Schedulers 때문)
        )

    def _build_models(self):
        args = self.args
        # Backbone 선택 로직 (원본 코드와 동일)
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

    def setup(self, stage=None):
        # 데이터 로더 길이를 기반으로 스케줄러 생성 (PL의 setup 단계에서 수행)
        # 주의: train_dataloader가 아직 setup 되지 않았을 수 있으므로 on_fit_start에서 할 수도 있음
        pass

    def on_fit_start(self):
        """
        학습 시작 시 스케줄러를 생성합니다.
        데이터로더 길이를 안전하게 가져와서 cosine scheduler를 생성합니다.
        """
        # 데이터로더 길이를 안전하게 가져오기
        # Lightning의 num_training_batches는 때때로 inf를 반환할 수 있으므로
        # 직접 데이터로더를 가져와서 길이를 계산하는 것이 더 안전합니다.

        # 방법 1: datamodule에서 직접 가져오기 (가장 안전)
        if hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            train_dataloader = self.trainer.datamodule.train_dataloader()
            if train_dataloader is not None:
                self.iters_per_epoch = len(train_dataloader)
            else:
                raise RuntimeError("Cannot get train_dataloader from datamodule")
        # 방법 2: trainer의 train_dataloader 사용
        elif hasattr(self.trainer, "train_dataloader"):
            train_dataloader = self.trainer.train_dataloader
            if train_dataloader is not None:
                # train_dataloader가 property인 경우 호출
                if callable(train_dataloader):
                    train_dataloader = train_dataloader()
                self.iters_per_epoch = len(train_dataloader)
            else:
                raise RuntimeError("Cannot get train_dataloader from trainer")
        # 방법 3: num_training_batches 사용 (inf 체크 포함)
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
        params_groups = utils.get_params_groups(self.student)
        if self.args.optimizer == "adamw":
            optimizer = optim.AdamW(params_groups)
        elif self.args.optimizer == "sgd":
            optimizer = optim.SGD(params_groups, lr=0, momentum=0.9)
        elif self.args.optimizer == "lars":
            optimizer = utils.LARS(params_groups)
        return optimizer

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        images, _ = batch  # Label은 무시

        # 1. Update Hyperparameters (LR, WD) manually per step
        # Global Step 계산
        it = self.global_step
        if it >= len(self.lr_schedule):
            it = len(self.lr_schedule) - 1  # Safety check

        for i, param_group in enumerate(opt.param_groups):
            param_group["lr"] = self.lr_schedule[it]
            if i == 0:  # Only the first group is regularized
                param_group["weight_decay"] = self.wd_schedule[it]

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
            # 최신 Lightning에서는 should_stop 대신 다른 방식 사용 가능
            # 하지만 should_stop도 여전히 작동함
            if hasattr(self.trainer, "should_stop"):
                self.trainer.should_stop = True
            return None

        # 4. Backward & Optimization (Manual)
        opt.zero_grad()
        self.manual_backward(loss)

        # Gradient Clipping
        if self.args.clip_grad:
            self.clip_gradients(
                opt,
                gradient_clip_val=self.args.clip_grad,
                gradient_clip_algorithm="norm",
            )

        # Cancel gradients for last layer (Freeze strategy in early epochs)
        utils.cancel_gradients_last_layer(
            self.current_epoch, self.student, self.args.freeze_last_layer
        )

        opt.step()

        # 5. EMA Update (Teacher)
        with torch.no_grad():
            m = self.momentum_schedule[it]
            for param_q, param_k in zip(
                self.student.parameters(), self.teacher.parameters()
            ):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

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

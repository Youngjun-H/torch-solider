"""SOLIDER LightningModule."""

import random
import sys
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as torchvision_models

# lightning-solider 루트 디렉토리를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from base.base_lightning_module import BaseDINOLightningModule
from dino.dino_criterion import DINOLoss
from models import swin_transformer as swin
from models import vision_transformer as vits
from models.vision_transformer import DINOHead
from shared import utils
from shared.semantic import build_onehot_semantic_weight, get_mask


class SOLIDERLightningModule(BaseDINOLightningModule):
    """
    SOLIDER 학습을 위한 PyTorch Lightning Module.

    BaseDINOLightningModule을 상속받아 SOLIDER 특화 기능을 구현합니다.
    - Semantic clustering 기반 part segmentation
    - MultiCropCondWrapper 사용
    - Part classifier 추가
    """

    def __init__(self, args):
        super().__init__(args)

        # SOLIDER 특화 파라미터
        self.partnum = args.partnum
        self.semantic_loss_weight = args.semantic_loss

        # 1. 모델 아키텍처 빌드 (Student & Teacher)
        self._build_models()

        # 2. Part Classifier 초기화
        self._build_part_classifier()

        # 3. Loss 함수 초기화
        self.dino_loss = DINOLoss(
            args.out_dim,
            args.local_crops_number + 2,  # Total crops
            args.warmup_teacher_temp,
            args.teacher_temp,
            args.warmup_teacher_temp_epochs,
            args.epochs,
        )
        self.part_loss = nn.CrossEntropyLoss(reduction="none")

    def _build_models(self):
        """SOLIDER 모델 아키텍처를 구축합니다."""
        args = self.args

        # Backbone 선택 로직 (DINO와 동일)
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
        elif args.arch in torchvision_models.__dict__.keys():
            student = torchvision_models.__dict__[args.arch]()
            teacher = torchvision_models.__dict__[args.arch]()
            embed_dim = student.fc.weight.shape[1]
        else:
            raise ValueError(f"Unknown architecture: {args.arch}")

        # MultiCropCondWrapper 사용 (DINO와 다름)
        self.student = utils.MultiCropCondWrapper(
            student,
            DINOHead(
                embed_dim,
                args.out_dim,
                use_bn=args.use_bn_in_head,
                norm_last_layer=args.norm_last_layer,
            ),
        )
        self.teacher = utils.MultiCropCondWrapper(
            teacher,
            DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
        )

        # Teacher 초기화 (Student와 동일하게 시작, Gradient 계산 안 함)
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.embed_dim = embed_dim

    def _build_part_classifier(self):
        """Part classifier를 구축합니다."""
        self.part_classifier = DINOHead(
            self.embed_dim,
            self.partnum + 1,  # partnum parts + background
            nlayers=self.args.parthead_nlayers,
        )

    def load_from_dino_checkpoint(self, checkpoint_path):
        """
        DINO 체크포인트에서 student와 teacher 가중치를 로드합니다.
        part_classifier는 새로 초기화된 상태로 유지됩니다.

        Args:
            checkpoint_path: DINO 체크포인트 파일 경로
        """
        import os

        if not os.path.isfile(checkpoint_path):
            print(f"Cannot find checkpoint at {checkpoint_path}")
            return

        print(f"Loading DINO checkpoint from {checkpoint_path}")

        # Load checkpoint
        # weights_only=False: PyTorch 2.6+에서 numpy 객체를 포함한 체크포인트 로드를 위해 필요
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Lightning checkpoint 형식인지 확인
        is_lightning_checkpoint = (
            "state_dict" in checkpoint and "pytorch-lightning_version" in checkpoint
        )

        def extract_state_dict(state_dict, prefix=""):
            """체크포인트에서 실제 state_dict를 추출합니다."""
            if isinstance(state_dict, dict):
                # Lightning checkpoint의 경우
                if "state_dict" in state_dict:
                    return extract_state_dict(state_dict["state_dict"], prefix)
                # 일반 체크포인트의 경우
                result = {}
                for k, v in state_dict.items():
                    # DDP wrapper 제거
                    new_key = k.replace("module.", "") if k.startswith("module.") else k
                    # Lightning 모듈 prefix 제거 (예: "student.backbone.", "teacher.head.")
                    if prefix and new_key.startswith(prefix):
                        new_key = new_key[len(prefix) :]
                    result[new_key] = v
                return result
            return state_dict

        # Load student weights
        if is_lightning_checkpoint:
            # Lightning checkpoint: state_dict에 모든 모델이 저장됨
            student_state = {}
            teacher_state = {}
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("student."):
                    student_state[k[8:]] = v  # "student." 제거
                elif k.startswith("teacher."):
                    teacher_state[k[8:]] = v  # "teacher." 제거
        else:
            # 원본 DINO 체크포인트 형식
            student_state = checkpoint.get("student", {})
            teacher_state = checkpoint.get("teacher", {})

        # Load student
        if student_state:
            try:
                student_state = extract_state_dict(student_state)
                msg = self.student.load_state_dict(student_state, strict=False)
                print(f"=> Loaded 'student' from checkpoint with msg: {msg}")
            except Exception as e:
                print(f"=> Failed to load 'student' from checkpoint: {e}")
        else:
            print("=> Key 'student' not found in checkpoint")

        # Load teacher
        if teacher_state:
            try:
                teacher_state = extract_state_dict(teacher_state)
                msg = self.teacher.load_state_dict(teacher_state, strict=False)
                print(f"=> Loaded 'teacher' from checkpoint with msg: {msg}")
            except Exception as e:
                print(f"=> Failed to load 'teacher' from checkpoint: {e}")
        else:
            print("=> Key 'teacher' not found in checkpoint")

        print("=> Part classifier remains newly initialized")

    def training_step(self, batch, batch_idx):
        """SOLIDER 학습 스텝."""
        opt = self.optimizers()
        images, _ = batch  # Label은 무시

        # 1. Update Hyperparameters (LR, WD) manually per step
        self._update_schedulers(opt)

        # 2. Forward Pass with Semantic Weight
        # Get device from images
        device = images[0].device

        # Teacher forward, semantic weight = 1 (for mask generation)
        weight_unit_1 = build_onehot_semantic_weight(
            self.args.batch_size_per_gpu, setVal=1, device=device
        )
        semantic_weight_1 = [weight_unit_1, weight_unit_1]

        with torch.no_grad():
            _, teacher_feats = self.teacher(
                images[:2], semantic_weight_1
            )  # only the 2 global views pass through the teacher

        # Teacher forward, semantic weight = randint (for DINO loss)
        weight_unit_rand = build_onehot_semantic_weight(
            self.args.batch_size_per_gpu, setVal=-1, device=device
        )
        semantic_weight_rand = [weight_unit_rand, weight_unit_rand]

        with torch.no_grad():
            teacher_output, _ = self.teacher(images[:2], semantic_weight_rand)

        # 3. Build semantic labels with teacher feats
        mask, mask_idxs = get_mask(teacher_feats, self.partnum)
        nimages = torch.cat(images[:2])

        if len(mask_idxs) != 0:
            # Random part selection for masking
            k = random.randint(1, self.partnum)
            nmask = np.where(mask == k, 0.0, 1.0)
            scale_factor = int(images[0].shape[-1] / nmask.shape[-1])
            nmask = torch.from_numpy(nmask).unsqueeze(1).to(images[0].device)
            nmask = F.interpolate(nmask, scale_factor=scale_factor, mode="nearest")
            nimages = nimages[torch.from_numpy(mask_idxs)]
            nimages = nimages * nmask

        # 4. Student forward, semantic weight = randint
        if len(mask_idxs) != 0:
            semantic_weight_student = [
                torch.cat(semantic_weight_rand)[torch.from_numpy(mask_idxs)]
            ] + [weight_unit_rand for _ in range(2 + self.args.local_crops_number)]
            student_output, student_feats = self.student(
                [nimages.float()] + images, semantic_weight_student
            )
        else:
            semantic_weight_student = [
                weight_unit_rand for _ in range(2 + self.args.local_crops_number)
            ]
            student_output, student_feats = self.student(
                images, semantic_weight_student
            )

        # 5. Extract features for part classification
        if len(mask_idxs) != 0:
            feats = student_feats[
                : (nimages.shape[0] + self.args.batch_size_per_gpu * 2)
            ].permute(0, 2, 3, 1)
            feats1 = feats[: nimages.shape[0]]
            feats2 = feats[nimages.shape[0] :][torch.from_numpy(mask_idxs)]
            feats = torch.cat([feats1, feats2])
        else:
            feats = student_feats[: self.args.batch_size_per_gpu * 2].permute(
                0, 2, 3, 1
            )

        # 6. Compute semantic classification loss
        feats = feats.reshape(-1, feats.shape[-1])
        pred = self.part_classifier(feats)
        labels = torch.zeros(pred.shape[0], dtype=torch.long, device=pred.device)

        if len(mask_idxs) != 0:
            labels = torch.from_numpy(mask.flatten()).long().to(pred.device)
            labels = torch.cat([labels, labels])

        loss2 = self.part_loss(pred, labels)
        acc = (pred.max(1)[1] == labels).float().mean()

        # 7. Compute DINO cross-entropy loss
        if len(mask_idxs) != 0:
            dino_student_output = student_output[nimages.shape[0] :]
        else:
            dino_student_output = student_output

        loss1 = self.dino_loss(dino_student_output, teacher_output, self.current_epoch)

        # 8. Sum all losses with semantic weight
        if len(mask_idxs) != 0:
            semantic_weight1 = semantic_weight_student[0]
            semantic_weight2 = torch.cat(semantic_weight_student[1:3])[
                torch.from_numpy(mask_idxs)
            ]
            semantic_weight = torch.cat([semantic_weight1, semantic_weight2])[:, 0]
            semantic_weight = semantic_weight.repeat_interleave(
                int(loss2.shape[0] / semantic_weight.shape[0])
            ).flatten()
            loss2 = torch.mean(self.semantic_loss_weight * semantic_weight * loss2)
        else:
            loss2 = torch.tensor(0.0, device=loss1.device)
            acc = torch.tensor(0.0, device=loss1.device)

        loss = loss1 + loss2

        if not torch.isfinite(loss):
            self.print(f"Loss is {loss}, stopping training")
            if hasattr(self.trainer, "should_stop"):
                self.trainer.should_stop = True
            return None

        # 9. Backward & Optimization (Manual)
        opt.zero_grad()
        self.manual_backward(loss)

        # Gradient Clipping
        self._apply_gradient_clipping(opt)

        # Cancel gradients for last layer (Freeze strategy in early epochs)
        self._cancel_gradients_last_layer()

        opt.step()

        # 10. EMA Update (Teacher)
        self._update_teacher_ema()

        # 11. Logging - loss.item()로 스칼라만 전달하여 메모리 절약
        loss_value = loss.item()
        loss1_value = loss1.item()
        self.log(
            "train_loss",
            loss_value,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log("loss1", loss1_value, on_step=True, logger=True, sync_dist=True)
        self.log("fgnum", len(mask_idxs), on_step=True, logger=True, sync_dist=True)

        if len(mask_idxs) != 0:
            loss2_value = loss2.item()
            acc_value = acc.item()
            self.log("loss2", loss2_value, on_step=True, logger=True, sync_dist=True)
            self.log("acc", acc_value, on_step=True, logger=True, sync_dist=True)

        self.log("lr", opt.param_groups[0]["lr"], on_step=True, logger=True)
        self.log("wd", opt.param_groups[0]["weight_decay"], on_step=True, logger=True)

        # 12. 명시적 메모리 해제 (메모리 누수 방지)
        del student_output, student_feats, teacher_output, teacher_feats
        if len(mask_idxs) != 0:
            del feats, pred, labels, nimages, nmask

        # 주기적으로 메모리 캐시 정리 (매 100 스텝마다)
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()

        # Manual optimization 모드에서는 Tensor를 반환해야 함 (detach로 계산 그래프 분리)
        return loss.detach()

"""
PyTorch Lightning 2.6+ Module for Person Re-Identification

Updated to follow Lightning 2.6+ best practices:
- import lightning as L
- Explicit logging with sync_dist=True
- Manual validation output management
- Dict-based optimizer configuration
"""

import torch
import torch.nn as nn
import lightning as L  # ✅ Lightning 2.6+ import
from typing import Dict, List, Tuple, Optional, Any
import torch.nn.functional as F

from .backbones.swin_transformer import (
    swin_base_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_tiny_patch4_window7_224
)
from .backbones.vit_pytorch import (
    vit_base_patch16_224_TransReID,
    vit_small_patch16_224_TransReID
)
from .backbones.resnet_ibn_a import resnet50_ibn_a, resnet101_ibn_a
from .backbones.resnet import ResNet, Bottleneck


def weights_init_kaiming(m):
    """Initialize weights with Kaiming normalization"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """Initialize classifier weights"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class ReIDLightningModule(L.LightningModule):  # ✅ L.LightningModule (Lightning 2.6+)
    """
    PyTorch Lightning 2.6+ Module for Person Re-Identification

    Features:
    - Multi-loss training (ID loss + Triplet loss)
    - Metric learning with triplet sampling
    - Feature extraction for evaluation
    - Support for Swin Transformer, ViT, ResNet backbones

    Lightning 2.6+ Updates:
    - Manual validation output management
    - Explicit logging with sync_dist
    - Dict-based optimizer configuration
    """

    def __init__(
        self,
        # Model architecture
        model_name: str = 'swin_base_patch4_window7_224',
        num_classes: int = 751,
        pretrain_path: str = '',
        semantic_weight: float = 0.2,

        # Training config
        img_size: Tuple[int, int] = (384, 128),
        drop_path_rate: float = 0.1,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,

        # Loss config
        id_loss_weight: float = 1.0,
        triplet_loss_weight: float = 1.0,
        triplet_margin: float = 0.3,
        label_smoothing: float = 0.0,

        # Optimizer config
        optimizer_name: str = 'SGD',
        lr: float = 0.0008,
        weight_decay: float = 0.0001,
        momentum: float = 0.9,
        bias_lr_factor: float = 2.0,

        # Scheduler config
        scheduler_name: str = 'cosine',
        warmup_epochs: int = 20,
        max_epochs: int = 120,
        steps: Tuple[int, ...] = (40, 70),
        gamma: float = 0.1,

        # Other
        feat_dim: int = 768,
        neck: str = 'bnneck',
        neck_feat: str = 'before',
        **kwargs
    ):
        super().__init__()

        # ✅ Lightning 2.6+: Save all hyperparameters
        self.save_hyperparameters()

        # ✅ Lightning 2.6+: Manual validation output management
        self.validation_step_outputs = []

        # Build backbone
        self.backbone = self._build_backbone()
        self.in_planes = self.backbone.num_features[-1] if hasattr(self.backbone, 'num_features') else 2048

        # Build neck and classifier
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.dropout = nn.Dropout(drop_rate)

        # Build loss functions
        if label_smoothing > 0:
            from losses import LabelSmoothingCrossEntropy
            self.ce_loss = LabelSmoothingCrossEntropy(epsilon=label_smoothing)
        else:
            self.ce_loss = nn.CrossEntropyLoss()

        from losses import TripletLoss
        self.triplet_loss = TripletLoss(margin=triplet_margin)

        # Load pretrained weights if provided
        if pretrain_path and pretrain_path != '':
            self._load_pretrained_weights(pretrain_path)

    def _build_backbone(self):
        """Build backbone network based on model_name"""
        model_name = self.hparams.model_name
        img_size = self.hparams.img_size
        drop_path_rate = self.hparams.drop_path_rate
        drop_rate = self.hparams.drop_rate
        attn_drop_rate = self.hparams.attn_drop_rate
        pretrain_path = self.hparams.pretrain_path
        semantic_weight = self.hparams.semantic_weight

        backbone_factory = {
            'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
            'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
            'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
            'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
            'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
        }

        if model_name in backbone_factory:
            # Transformer-based models
            convert_weights = True if pretrain_path else False
            backbone = backbone_factory[model_name](
                img_size=img_size,
                drop_path_rate=drop_path_rate,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                pretrained=pretrain_path if pretrain_path else None,
                convert_weights=convert_weights,
                semantic_weight=semantic_weight
            )
        elif model_name == 'resnet50_ibn_a':
            backbone = resnet50_ibn_a(last_stride=1)
        elif model_name == 'resnet50':
            backbone = ResNet(last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3])
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        return backbone

    def _load_pretrained_weights(self, pretrain_path: str):
        """Load pretrained weights from checkpoint"""
        try:
            checkpoint = torch.load(pretrain_path, map_location='cpu')
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace('module.', '')
                # Skip classifier weights (different num_classes)
                if 'classifier' not in new_key:
                    new_state_dict[new_key] = v

            # Load into backbone only
            msg = self.backbone.load_state_dict(new_state_dict, strict=False)
            print(f"✅ Loaded pretrained weights from {pretrain_path}")
            if msg.missing_keys:
                print(f"   Missing keys: {len(msg.missing_keys)}")
            if msg.unexpected_keys:
                print(f"   Unexpected keys: {len(msg.unexpected_keys)}")
        except Exception as e:
            print(f"⚠️  Could not load pretrained weights: {e}")

    def forward(self, x, label=None, cam_label=None, view_label=None):
        """
        Forward pass for inference

        Returns:
            features: Normalized features for ReID matching
            featmaps: Feature maps (for visualization)
        """
        # Extract features from backbone
        global_feat, featmaps = self.backbone(x)

        # Apply bottleneck
        feat = self.bottleneck(global_feat)

        if self.training:
            # Training mode: return classification scores and features
            feat_cls = self.dropout(feat)
            cls_score = self.classifier(feat_cls)
            return cls_score, global_feat, featmaps
        else:
            # Inference mode: return features for evaluation
            if self.hparams.neck_feat == 'after':
                return feat, featmaps
            else:
                return global_feat, featmaps

    def training_step(self, batch, batch_idx):
        """
        Training step with multi-loss

        ✅ Lightning 2.6+: Explicit logging with sync_dist=True
        """
        imgs, pids, cam_ids, view_ids = batch

        # Forward pass
        cls_score, global_feat, featmaps = self(imgs, label=pids, cam_label=cam_ids, view_label=view_ids)

        # Compute ID loss
        id_loss = self.ce_loss(cls_score, pids)

        # Compute triplet loss (returns loss, dist_ap, dist_an)
        triplet_loss, dist_ap, dist_an = self.triplet_loss(global_feat, pids)

        # Calculate triplet precision: percentage of triplets where dist_an > dist_ap
        # This indicates how well the model separates positives from negatives
        if dist_ap.sum() > 0:  # Check for valid triplets (not dummy zeros)
            triplet_prec = (dist_an > dist_ap).float().mean()
        else:
            triplet_prec = torch.tensor(0.0, device=global_feat.device)

        # Combined loss
        loss = (
            self.hparams.id_loss_weight * id_loss +
            self.hparams.triplet_loss_weight * triplet_loss
        )

        # Compute accuracy
        preds = torch.argmax(cls_score, dim=1)
        acc = (preds == pids).float().mean()

        # ✅ Lightning 2.6+: Explicit logging with all parameters
        self.log('train/loss', loss,
                 on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log('train/id_loss', id_loss,
                 on_step=True, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log('train/triplet_loss', triplet_loss,
                 on_step=True, on_epoch=True,
                 logger=True, sync_dist=True)
        self.log('train/acc', acc,
                 on_step=True, on_epoch=True, prog_bar=True,
                 logger=True, sync_dist=True)
        self.log('train/triplet_prec', triplet_prec,
                 on_step=True, on_epoch=True,
                 logger=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step - extract features for evaluation

        ✅ Lightning 2.6+: Manual output management
        """
        imgs, pids, cam_ids, cam_ids_batch, view_ids, img_paths = batch

        # Extract features
        feat, featmaps = self(imgs, cam_label=cam_ids_batch, view_label=view_ids)

        # ✅ Lightning 2.6+: Store outputs manually
        output = {
            'feat': feat.cpu(),
            'pids': pids,
            'cam_ids': cam_ids,
        }
        self.validation_step_outputs.append(output)

        return output

    def on_validation_epoch_end(self):
        """
        ✅ Lightning 2.6+: Use manually collected outputs

        Called at the end of validation epoch
        """
        # Note: Actual mAP/CMC computation is handled by ReIDEvaluationCallback
        # This is just for housekeeping

        # ✅ Lightning 2.6+: Clear outputs for next epoch
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        """
        ✅ Lightning 2.6+: Dict-based configuration (recommended)

        Configure optimizer and learning rate scheduler
        """
        # Build parameter groups with different learning rates
        params = []
        for key, value in self.named_parameters():
            if not value.requires_grad:
                continue

            lr = self.hparams.lr
            weight_decay = self.hparams.weight_decay

            # Different LR for bias
            if "bias" in key:
                lr = self.hparams.lr * self.hparams.bias_lr_factor
                weight_decay = 0

            params.append({
                "params": [value],
                "lr": lr,
                "weight_decay": weight_decay
            })

        # Build optimizer
        if self.hparams.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(params, momentum=self.hparams.momentum)
        elif self.hparams.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(params)
        elif self.hparams.optimizer_name == 'AdamW':
            optimizer = torch.optim.AdamW(params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")

        # Build scheduler
        if self.hparams.scheduler_name == 'cosine':
            # Warmup scheduler
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                total_iters=self.hparams.warmup_epochs
            )

            # Main cosine scheduler
            main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs - self.hparams.warmup_epochs,
                eta_min=1e-7
            )

            # ✅ Combine with SequentialLR
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[self.hparams.warmup_epochs]
            )

        elif self.hparams.scheduler_name == 'step':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=list(self.hparams.steps),
                gamma=self.hparams.gamma
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.hparams.scheduler_name}")

        # ✅ Lightning 2.6+: Return dict format (recommended)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
            }
        }

    def on_train_epoch_end(self):
        """Called at the end of training epoch"""
        # Log current learning rate
        current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
        self.log('train/lr', current_lr,
                 on_epoch=True, prog_bar=False,
                 logger=True, sync_dist=True)

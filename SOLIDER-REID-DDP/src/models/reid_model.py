"""
Native PyTorch ReID Model (without Lightning)
Person Re-Identification with SOLIDER backbone
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

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


class ReIDModel(nn.Module):
    """
    Native PyTorch ReID Model

    Features:
    - Multi-loss training (ID loss + Triplet loss)
    - Support for Swin Transformer, ViT, ResNet backbones
    - BN Neck for better generalization
    """

    def __init__(
        self,
        model_name: str = 'swin_base_patch4_window7_224',
        num_classes: int = 751,
        pretrain_path: str = '',
        semantic_weight: float = 0.2,
        img_size: Tuple[int, int] = (384, 128),
        drop_path_rate: float = 0.1,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        neck: str = 'bnneck',
        neck_feat: str = 'before',
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.neck = neck
        self.neck_feat = neck_feat

        # Build backbone
        self.backbone = self._build_backbone(
            model_name, img_size, drop_path_rate, drop_rate,
            attn_drop_rate, pretrain_path, semantic_weight
        )

        # Get feature dimension
        if hasattr(self.backbone, 'num_features'):
            self.in_planes = self.backbone.num_features[-1]
        else:
            self.in_planes = 2048

        # Build neck and classifier
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.dropout = nn.Dropout(drop_rate)

        # Load pretrained weights if provided
        if pretrain_path and pretrain_path != '':
            self._load_pretrained_weights(pretrain_path)

    def _build_backbone(
        self, model_name, img_size, drop_path_rate,
        drop_rate, attn_drop_rate, pretrain_path, semantic_weight
    ):
        """Build backbone network based on model_name"""
        backbone_factory = {
            'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
            'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
            'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
            'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
            'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
        }

        if model_name in backbone_factory:
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
            print(f"Loaded pretrained weights from {pretrain_path}")
            if msg.missing_keys:
                print(f"  Missing keys: {len(msg.missing_keys)}")
            if msg.unexpected_keys:
                print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")

    def forward(
        self,
        x: torch.Tensor,
        label: Optional[torch.Tensor] = None,
        cam_label: Optional[torch.Tensor] = None,
        view_label: Optional[torch.Tensor] = None
    ):
        """
        Forward pass

        Args:
            x: Input images [B, C, H, W]
            label: Person IDs (for training)
            cam_label: Camera IDs
            view_label: View IDs

        Returns:
            Training mode: (cls_score, global_feat, featmaps)
            Inference mode: (feat, featmaps)
        """
        # Extract features from backbone
        global_feat, featmaps = self.backbone(x)

        # Apply bottleneck (BN Neck)
        feat = self.bottleneck(global_feat)

        if self.training:
            # Training mode: return classification scores and features
            feat_cls = self.dropout(feat)
            cls_score = self.classifier(feat_cls)
            return cls_score, global_feat, featmaps
        else:
            # Inference mode: return features for evaluation
            if self.neck_feat == 'after':
                return feat, featmaps
            else:
                return global_feat, featmaps


def build_reid_model(cfg) -> ReIDModel:
    """Build ReID model from config"""
    model = ReIDModel(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        pretrain_path=cfg.model.pretrain_path,
        semantic_weight=cfg.model.semantic_weight,
        img_size=tuple(cfg.data.img_size_train),
        drop_path_rate=cfg.model.drop_path_rate,
        drop_rate=cfg.model.drop_rate,
        attn_drop_rate=cfg.model.attn_drop_rate,
        neck=cfg.model.neck,
        neck_feat=cfg.model.neck_feat,
    )
    return model

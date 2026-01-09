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
        frozen_stages: int = -1,  # -1 = no freezing, 4 = freeze entire backbone
    ):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.img_size = img_size
        self.neck = neck
        self.neck_feat = neck_feat
        self.frozen_stages = frozen_stages

        # Build backbone
        self.backbone = self._build_backbone(
            model_name, img_size, drop_path_rate, drop_rate,
            attn_drop_rate, pretrain_path, semantic_weight, frozen_stages
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
        # NOTE: Weight may have already been loaded in backbone.init_weights()
        # This is a secondary loading attempt for the full model
        if pretrain_path and pretrain_path != '':
            print("\n[DEBUG] ReIDModel.__init__: Attempting to load pretrained weights...")
            print("[DEBUG] NOTE: Backbone may have already loaded weights via init_weights()")
            self._load_pretrained_weights(pretrain_path)

    def _build_backbone(
        self, model_name, img_size, drop_path_rate,
        drop_rate, attn_drop_rate, pretrain_path, semantic_weight, frozen_stages
    ):
        """Build backbone network based on model_name"""
        print("\n" + "=" * 80)
        print("[DEBUG] Building Backbone")
        print("=" * 80)
        print(f"[DEBUG] Model name: {model_name}")
        print(f"[DEBUG] Image size: {img_size}")
        print(f"[DEBUG] Pretrain path: {pretrain_path}")
        print(f"[DEBUG] Semantic weight: {semantic_weight}")
        print(f"[DEBUG] Frozen stages: {frozen_stages}")

        backbone_factory = {
            'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
            'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
            'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
            'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
            'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
        }

        if model_name in backbone_factory:
            convert_weights = True if pretrain_path else False
            print(f"[DEBUG] Convert weights: {convert_weights}")

            backbone = backbone_factory[model_name](
                img_size=img_size,
                drop_path_rate=drop_path_rate,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                pretrained=pretrain_path if pretrain_path else None,
                convert_weights=convert_weights,
                semantic_weight=semantic_weight,
                frozen_stages=frozen_stages,
            )
            print(f"[DEBUG] Backbone created: {type(backbone).__name__}")

            # Explicitly call init_weights for SwinTransformer
            if hasattr(backbone, 'init_weights') and pretrain_path:
                print(f"[DEBUG] Calling backbone.init_weights('{pretrain_path}')")
                backbone.init_weights(pretrain_path)
            else:
                print(f"[DEBUG] init_weights not called (no pretrain_path or method)")

        elif model_name == 'resnet50_ibn_a':
            backbone = resnet50_ibn_a(last_stride=1)
            print(f"[DEBUG] Backbone created: ResNet50-IBN-A")
        elif model_name == 'resnet50':
            backbone = ResNet(last_stride=1, block=Bottleneck, layers=[3, 4, 6, 3])
            print(f"[DEBUG] Backbone created: ResNet50")
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        print("[DEBUG] Backbone building completed!")
        print("=" * 80 + "\n")

        return backbone

    def _load_pretrained_weights(self, pretrain_path: str):
        """Load pretrained weights from checkpoint"""
        try:
            print("\n" + "=" * 80)
            print("[DEBUG] Pretrain Weight Loading")
            print("=" * 80)
            print(f"[DEBUG] Loading pretrained weights from: {pretrain_path}")

            checkpoint = torch.load(pretrain_path, map_location='cpu')

            # Debug: Show checkpoint structure
            print(f"[DEBUG] Checkpoint type: {type(checkpoint)}")
            if isinstance(checkpoint, dict):
                print(f"[DEBUG] Checkpoint keys: {list(checkpoint.keys())}")

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                print("[DEBUG] Using 'state_dict' key from checkpoint")
            else:
                state_dict = checkpoint
                print("[DEBUG] Using checkpoint directly as state_dict")

            print(f"[DEBUG] Total keys in pretrain checkpoint: {len(state_dict)}")

            # Show sample keys from checkpoint
            sample_keys = list(state_dict.keys())[:10]
            print(f"[DEBUG] Sample checkpoint keys (first 10): {sample_keys}")

            # Remove 'module.' prefix if present
            new_state_dict = {}
            module_prefix_count = 0
            classifier_skip_count = 0
            for k, v in state_dict.items():
                new_key = k.replace('module.', '')
                if new_key != k:
                    module_prefix_count += 1
                # Skip classifier weights (different num_classes)
                if 'classifier' not in new_key:
                    new_state_dict[new_key] = v
                else:
                    classifier_skip_count += 1

            print(f"[DEBUG] Keys with 'module.' prefix removed: {module_prefix_count}")
            print(f"[DEBUG] Classifier keys skipped: {classifier_skip_count}")
            print(f"[DEBUG] Keys to load after filtering: {len(new_state_dict)}")

            # Get backbone state dict before loading
            backbone_state_before = {k: v.clone() for k, v in self.backbone.state_dict().items()}
            backbone_keys = set(self.backbone.state_dict().keys())
            pretrain_keys = set(new_state_dict.keys())

            print(f"\n[DEBUG] Backbone total keys: {len(backbone_keys)}")
            print(f"[DEBUG] Pretrain total keys: {len(pretrain_keys)}")

            # Check key matching
            matched_keys = backbone_keys & pretrain_keys
            missing_in_pretrain = backbone_keys - pretrain_keys
            unexpected_in_pretrain = pretrain_keys - backbone_keys

            print(f"[DEBUG] Matched keys: {len(matched_keys)}")
            print(f"[DEBUG] Missing in pretrain (backbone has, pretrain doesn't): {len(missing_in_pretrain)}")
            print(f"[DEBUG] Unexpected in pretrain (pretrain has, backbone doesn't): {len(unexpected_in_pretrain)}")

            if missing_in_pretrain:
                print(f"[DEBUG] Sample missing keys: {list(missing_in_pretrain)[:5]}")
            if unexpected_in_pretrain:
                print(f"[DEBUG] Sample unexpected keys: {list(unexpected_in_pretrain)[:5]}")

            # Load into backbone only
            msg = self.backbone.load_state_dict(new_state_dict, strict=False)

            print(f"\n[DEBUG] Load state dict result:")
            print(f"  - Missing keys: {len(msg.missing_keys)}")
            print(f"  - Unexpected keys: {len(msg.unexpected_keys)}")

            if msg.missing_keys:
                print(f"[DEBUG] Missing keys (first 10): {msg.missing_keys[:10]}")
            if msg.unexpected_keys:
                print(f"[DEBUG] Unexpected keys (first 10): {msg.unexpected_keys[:10]}")

            # Verify weights actually changed
            backbone_state_after = self.backbone.state_dict()
            changed_count = 0
            unchanged_count = 0

            for key in matched_keys:
                before = backbone_state_before[key]
                after = backbone_state_after[key]
                if torch.equal(before, after):
                    unchanged_count += 1
                else:
                    changed_count += 1

            print(f"\n[DEBUG] Weight verification:")
            print(f"  - Keys with changed weights: {changed_count}")
            print(f"  - Keys with unchanged weights: {unchanged_count}")

            # Sample weight comparison for first matched key
            if matched_keys:
                sample_key = list(matched_keys)[0]
                before_sample = backbone_state_before[sample_key].flatten()[:5]
                after_sample = backbone_state_after[sample_key].flatten()[:5]
                pretrain_sample = new_state_dict[sample_key].flatten()[:5]
                print(f"\n[DEBUG] Sample weight comparison for '{sample_key}':")
                print(f"  - Before loading (first 5 values): {before_sample.tolist()}")
                print(f"  - After loading (first 5 values): {after_sample.tolist()}")
                print(f"  - Pretrain values (first 5 values): {pretrain_sample.tolist()}")

                if torch.allclose(after_sample, pretrain_sample):
                    print(f"  - [OK] Weights match pretrain!")
                else:
                    print(f"  - [WARNING] Weights do NOT match pretrain!")

            print("\n" + "=" * 80)
            print(f"[DEBUG] Pretrain weight loading completed!")
            print("=" * 80 + "\n")

        except Exception as e:
            print(f"[ERROR] Could not load pretrained weights: {e}")
            import traceback
            traceback.print_exc()

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
        frozen_stages=getattr(cfg.model, 'frozen_stages', -1),
    )
    return model

"""ReID Model with SOLIDER backbone."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone


def weights_init_kaiming(m):
    """Kaiming initialization."""
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
    """Classifier initialization."""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class ReIDModel(nn.Module):
    """ReID Model with SOLIDER backbone."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Build backbone
        self.backbone = build_backbone(
            model_name=config.model_name,
            img_size=config.image_size,
            semantic_weight=config.semantic_weight,
            drop_path_rate=config.drop_path_rate,
            drop_rate=config.drop_rate,
            attn_drop_rate=config.attn_drop_rate,
            pretrained=config.pretrain_path,
        )
        
        # Get feature dimension from backbone
        # Swin outputs: (batch, num_features[-1])
        # We need to get the last stage feature dim
        # For Swin: tiny=768, small=768, base=1024
        if config.model_name == "swin_base":
            in_planes = 1024
        else:
            in_planes = 768
        
        # Reduce feature dimension if needed
        if config.feat_dim < in_planes:
            self.fcneck = nn.Linear(in_planes, config.feat_dim, bias=False)
            self.fcneck.apply(weights_init_kaiming)
            in_planes = config.feat_dim
        else:
            self.fcneck = None
        
        # BNNeck
        self.bottleneck = nn.BatchNorm1d(in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        # Dropout
        if config.drop_rate > 0:
            self.dropout = nn.Dropout(config.drop_rate)
        else:
            self.dropout = None
        
        # Classifier
        self.classifier = nn.Linear(in_planes, config.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        
    def forward(self, x, label=None, cam_label=None, view_label=None):
        """
        Forward pass.
        
        Args:
            x: Input images (B, C, H, W)
            label: Person IDs (B,) - for training
            cam_label: Camera IDs (B,) - optional
            view_label: View IDs (B,) - optional
        """
        # Backbone forward
        global_feat, featmaps = self.backbone(x)
        
        # Reduce dimension if needed
        if self.fcneck is not None:
            global_feat = self.fcneck(global_feat)
        
        # BNNeck
        feat = self.bottleneck(global_feat)
        
        # Dropout
        if self.dropout is not None:
            feat = self.dropout(feat)
        
        if self.training:
            # Training: return classification score and global feature
            cls_score = self.classifier(feat)
            return cls_score, global_feat, featmaps
        else:
            # Inference: return feature after BN
            return feat, featmaps




"""Backbone models for ReID."""
import sys
from pathlib import Path
import torch
import torch.nn as nn

# Import Swin Transformer from original SOLIDER-REID
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "SOLIDER-REID"))
from model.backbones.swin_transformer import (
    swin_tiny_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_base_patch4_window7_224,
)


def build_backbone(model_name: str, img_size: tuple, semantic_weight: float, 
                   drop_path_rate: float, drop_rate: float, attn_drop_rate: float,
                   pretrained: str = ""):
    """
    Build backbone model.
    
    Args:
        model_name: Model name (swin_tiny, swin_small, swin_base)
        img_size: Image size (H, W)
        semantic_weight: SOLIDER semantic weight
        drop_path_rate: Drop path rate
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        pretrained: Path to pretrained weights
    """
    img_size = max(img_size)  # Use max dimension for Swin
    
    factory = {
        "swin_tiny": swin_tiny_patch4_window7_224,
        "swin_small": swin_small_patch4_window7_224,
        "swin_base": swin_base_patch4_window7_224,
    }
    
    if model_name not in factory:
        raise ValueError(f"Unknown model: {model_name}")
    
    model = factory[model_name](
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        semantic_weight=semantic_weight,
        pretrained=pretrained if pretrained else None,
        convert_weights=True if pretrained else False,
    )
    
    # Load pretrained weights if provided
    if pretrained:
        model.init_weights(pretrained)
    
    return model




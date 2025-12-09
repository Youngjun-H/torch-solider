"""Utility functions for Swin Transformer (replacing mmcv)."""

import logging

import torch


def load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    """
    Load checkpoint from file.
    Replaces mmcv.runner.load_checkpoint.

    Args:
        model: Model to load checkpoint into
        filename: Path to checkpoint file
        map_location: Device to map checkpoint to
        strict: Whether to strictly enforce that the keys match
        logger: Optional logger for warnings

    Returns:
        missing_keys, unexpected_keys
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    checkpoint = torch.load(filename, map_location=map_location)

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Remove 'module.' prefix if present
    if state_dict and len(state_dict) > 0:
        first_key = list(state_dict.keys())[0]
        if first_key.startswith("module."):
            state_dict = {k[7:]: v for k, v in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")

    return missing_keys, unexpected_keys

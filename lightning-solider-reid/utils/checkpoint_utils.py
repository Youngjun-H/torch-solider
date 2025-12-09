"""Utility functions for loading checkpoints (including Lightning checkpoints)."""

import logging

import torch

logger = logging.getLogger(__name__)


def load_checkpoint_state_dict(checkpoint_path, map_location="cpu"):
    """
    Load state_dict from checkpoint file.
    Supports both regular PyTorch checkpoints and Lightning checkpoints (.ckpt).

    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device to map checkpoint to

    Returns:
        state_dict: Model state dictionary
    """
    ckpt = torch.load(checkpoint_path, map_location=map_location)

    # Handle Lightning checkpoint format
    if isinstance(ckpt, dict):
        # Lightning checkpoint has 'state_dict' key
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            logger.info("Loading from Lightning checkpoint (.ckpt format)")
        # Handle 'teacher' key (for SOLIDER)
        elif "teacher" in ckpt:
            state_dict = ckpt["teacher"]
            logger.info("Loading from checkpoint with 'teacher' key")
        # Handle 'model' key
        elif "model" in ckpt:
            state_dict = ckpt["model"]
            logger.info("Loading from checkpoint with 'model' key")
        # Direct state_dict
        else:
            state_dict = ckpt
            logger.info("Loading from direct state_dict checkpoint")
    else:
        state_dict = ckpt
        logger.info("Loading from direct state_dict checkpoint")

    # Remove Lightning-specific prefixes
    # Lightning may add 'model.' prefix to all keys
    if state_dict and len(state_dict) > 0:
        first_key = list(state_dict.keys())[0]

        # Remove 'model.' prefix if present (Lightning adds this)
        if first_key.startswith("model."):
            state_dict = {k[6:]: v for k, v in state_dict.items()}
            logger.info("Removed 'model.' prefix from state_dict keys")
            # Update first_key after removing prefix
            if state_dict:
                first_key = list(state_dict.keys())[0]

        # Remove 'base.' prefix if present (for transformer backbone in Lightning)
        if state_dict and first_key.startswith("base."):
            state_dict = {k[5:]: v for k, v in state_dict.items()}
            logger.info("Removed 'base.' prefix from state_dict keys")

    return state_dict

"""
Logging Utilities
"""

import os
import sys
import logging
from typing import Optional


def setup_logger(
    name: str = 'reid',
    save_dir: Optional[str] = None,
    distributed_rank: int = 0,
    filename: str = 'train.log'
) -> logging.Logger:
    """
    Setup logger for training

    Args:
        name: Logger name
        save_dir: Directory to save log file
        distributed_rank: Process rank (only rank 0 logs to file)
        filename: Log file name

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Don't add handlers if already configured
    if logger.hasHandlers():
        return logger

    # Create formatters
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler (all ranks)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (only rank 0)
    if save_dir and distributed_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

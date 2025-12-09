"""ReID Training Arguments."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def get_args_parser():
    parser = argparse.ArgumentParser("SOLIDER-REID Lightning", add_help=False)

    # Model parameters
    parser.add_argument(
        "--transformer_type",
        default="swin_base_patch4_window7_224",
        type=str,
        choices=[
            "swin_base_patch4_window7_224",
            "swin_small_patch4_window7_224",
            "swin_tiny_patch4_window7_224",
        ],
        help="Transformer backbone type",
    )
    parser.add_argument(
        "--pretrain_path",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint",
    )
    parser.add_argument(
        "--pretrain_choice",
        default="self",
        type=str,
        choices=["self", "imagenet"],
        help="Pretrain choice: self (SOLIDER) or imagenet",
    )
    parser.add_argument(
        "--semantic_weight",
        default=0.2,
        type=float,
        help="Semantic weight for SOLIDER",
    )
    parser.add_argument(
        "--id_loss_type",
        default="softmax",
        type=str,
        choices=["softmax", "arcface", "cosface", "amsoftmax", "circle"],
        help="ID loss type",
    )

    # Input parameters
    parser.add_argument(
        "--size_train",
        nargs=2,
        type=int,
        default=[384, 128],
        help="Training image size [height, width]",
    )
    parser.add_argument(
        "--size_test",
        nargs=2,
        type=int,
        default=[384, 128],
        help="Test image size [height, width]",
    )
    parser.add_argument(
        "--pixel_mean",
        nargs=3,
        type=float,
        default=[0.5, 0.5, 0.5],
        help="Pixel mean for normalization",
    )
    parser.add_argument(
        "--pixel_std",
        nargs=3,
        type=float,
        default=[0.5, 0.5, 0.5],
        help="Pixel std for normalization",
    )
    parser.add_argument(
        "--prob",
        default=0.5,
        type=float,
        help="Random horizontal flip probability",
    )
    parser.add_argument(
        "--re_prob",
        default=0.5,
        type=float,
        help="Random erasing probability",
    )
    parser.add_argument(
        "--padding",
        default=10,
        type=int,
        help="Padding for random crop",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_name",
        default="msmt17",
        type=str,
        choices=["msmt17", "market1501", "mm"],
        help="Dataset name",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory of dataset",
    )
    parser.add_argument(
        "--sampler",
        default="softmax_triplet",
        type=str,
        choices=["softmax", "softmax_triplet", "id_triplet", "id"],
        help="Data sampler type",
    )
    parser.add_argument(
        "--num_instance",
        default=4,
        type=int,
        help="Number of instances per identity in a batch",
    )

    # Solver parameters
    parser.add_argument(
        "--optimizer_name",
        default="SGD",
        type=str,
        choices=["SGD", "AdamW", "Adam"],
        help="Optimizer name",
    )
    parser.add_argument(
        "--base_lr",
        default=0.0002,
        type=float,
        help="Base learning rate",
    )
    parser.add_argument(
        "--max_epochs",
        default=120,
        type=int,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=20,
        type=int,
        help="Number of warmup epochs",
    )
    parser.add_argument(
        "--warmup_method",
        default="cosine",
        type=str,
        choices=["cosine", "linear", "constant"],
        help="Warmup method",
    )
    parser.add_argument(
        "--warmup_factor",
        default=0.01,
        type=float,
        help="Warmup factor",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-4,
        type=float,
        help="Weight decay",
    )
    parser.add_argument(
        "--weight_decay_bias",
        default=1e-4,
        type=float,
        help="Weight decay for bias",
    )
    parser.add_argument(
        "--bias_lr_factor",
        default=2,
        type=float,
        help="Learning rate factor for bias",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        help="Momentum for SGD",
    )
    parser.add_argument(
        "--large_fc_lr",
        action="store_true",
        help="Use 2x learning rate for FC layer",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=[40, 70],
        help="Milestones for step scheduler",
    )
    parser.add_argument(
        "--gamma",
        default=0.1,
        type=float,
        help="Gamma for step scheduler",
    )

    # Loss parameters
    parser.add_argument(
        "--metric_loss_type",
        default="triplet",
        type=str,
        choices=["triplet"],
        help="Metric loss type",
    )
    parser.add_argument(
        "--no_margin",
        action="store_true",
        help="Use soft triplet loss (no margin)",
    )
    parser.add_argument(
        "--margin",
        default=0.3,
        type=float,
        help="Margin for triplet loss",
    )
    parser.add_argument(
        "--id_loss_weight",
        default=1.0,
        type=float,
        help="Weight for ID loss",
    )
    parser.add_argument(
        "--triplet_loss_weight",
        default=1.0,
        type=float,
        help="Weight for triplet loss",
    )
    parser.add_argument(
        "--if_labelsmooth",
        default="off",
        type=str,
        choices=["on", "off"],
        help="Use label smoothing",
    )
    parser.add_argument(
        "--cosine_scale",
        default=30,
        type=float,
        help="Scale for cosine-based losses",
    )
    parser.add_argument(
        "--cosine_margin",
        default=0.5,
        type=float,
        help="Margin for cosine-based losses",
    )

    # Training parameters
    parser.add_argument(
        "--ims_per_batch",
        default=64,
        type=int,
        help="Images per batch",
    )
    parser.add_argument(
        "--test_ims_per_batch",
        default=256,
        type=int,
        help="Images per batch for testing",
    )
    parser.add_argument(
        "--num_workers",
        default=8,
        type=int,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--seed",
        default=1234,
        type=int,
        help="Random seed",
    )

    # Validation parameters
    parser.add_argument(
        "--eval_period",
        default=10,
        type=int,
        help="Validation period (epochs)",
    )
    parser.add_argument(
        "--feat_norm",
        default="yes",
        type=str,
        choices=["yes", "no"],
        help="Normalize features for evaluation",
    )
    parser.add_argument(
        "--neck_feat",
        default="before",
        type=str,
        choices=["before", "after"],
        help="Use feature before or after BN neck",
    )

    # Output parameters
    parser.add_argument(
        "--output_dir",
        default="./log/reid",
        type=str,
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--checkpoint_period",
        default=120,
        type=int,
        help="Checkpoint save period (epochs)",
    )
    parser.add_argument(
        "--log_period",
        default=20,
        type=int,
        help="Logging period (iterations)",
    )

    # Lightning parameters
    parser.add_argument(
        "--devices",
        default=None,
        type=int,
        help="Number of GPUs",
    )
    parser.add_argument(
        "--num_nodes",
        default=1,
        type=int,
        help="Number of nodes",
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        type=str,
        choices=["32", "16-mixed", "bf16-mixed"],
        help="Training precision",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="Resume from checkpoint",
    )

    return parser

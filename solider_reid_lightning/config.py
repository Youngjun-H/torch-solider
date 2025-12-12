"""Configuration management for SOLIDER-REID."""
import argparse
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    """Configuration class for SOLIDER-REID training."""
    # Data
    dataset_name: str = "market1501"
    data_root: str = "../data"
    image_size: tuple = (384, 128)
    num_workers: int = 8
    
    # DataLoader
    batch_size: int = 64
    num_instances: int = 16  # number of instances per identity
    
    # Model
    model_name: str = "swin_tiny"  # swin_tiny, swin_small, swin_base
    pretrain_path: str = ""
    semantic_weight: float = 1.0  # SOLIDER semantic weight
    num_classes: Optional[int] = None  # auto-determined from dataset
    feat_dim: int = 768
    drop_path_rate: float = 0.1
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    
    # Loss
    id_loss_weight: float = 1.0
    triplet_loss_weight: float = 1.0
    triplet_margin: float = 0.3
    label_smooth: bool = True
    
    # Optimizer
    optimizer: str = "AdamW"
    base_lr: float = 3e-4
    weight_decay: float = 0.0005
    momentum: float = 0.9
    
    # Scheduler
    max_epochs: int = 100
    warmup_epochs: int = 5
    warmup_method: str = "cosine"  # cosine or linear
    warmup_factor: float = 0.01
    
    # Training
    seed: int = 1234
    precision: str = "16-mixed"
    accumulate_grad_batches: int = 1
    log_every_n_steps: int = 100
    val_check_interval: float = 1.0  # validate every N epochs
    
    # Checkpoint
    output_dir: str = "./outputs"
    save_top_k: int = 3
    monitor: str = "val_mAP"
    monitor_mode: str = "max"
    
    # DDP
    devices: Optional[int] = None  # None for auto
    num_nodes: int = 1
    
    # Test
    test_weight: str = ""
    rerank: bool = False


def get_args_parser():
    """Get argument parser with all configuration options."""
    parser = argparse.ArgumentParser("SOLIDER-REID Training")
    
    # Data
    parser.add_argument("--dataset_name", type=str, default="market1501", choices=["market1501", "msmt17"])
    parser.add_argument("--data_root", type=str, default="../data")
    parser.add_argument("--image_size", type=int, nargs=2, default=[384, 128])
    parser.add_argument("--num_workers", type=int, default=8)
    
    # DataLoader
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_instances", type=int, default=16)
    
    # Model
    parser.add_argument("--model_name", type=str, default="swin_tiny", 
                       choices=["swin_tiny", "swin_small", "swin_base"])
    parser.add_argument("--pretrain_path", type=str, default="")
    parser.add_argument("--semantic_weight", type=float, default=1.0)
    parser.add_argument("--feat_dim", type=int, default=768)
    parser.add_argument("--drop_path_rate", type=float, default=0.1)
    parser.add_argument("--drop_rate", type=float, default=0.0)
    parser.add_argument("--attn_drop_rate", type=float, default=0.0)
    
    # Loss
    parser.add_argument("--id_loss_weight", type=float, default=1.0)
    parser.add_argument("--triplet_loss_weight", type=float, default=1.0)
    parser.add_argument("--triplet_margin", type=float, default=0.3)
    parser.add_argument("--label_smooth", action="store_true", default=True)
    parser.add_argument("--no_label_smooth", dest="label_smooth", action="store_false")
    
    # Optimizer
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=["AdamW", "Adam", "SGD"])
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--momentum", type=float, default=0.9)
    
    # Scheduler
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--warmup_method", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--warmup_factor", type=float, default=0.01)
    
    # Training
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--precision", type=str, default="16-mixed", choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=100)
    parser.add_argument("--val_check_interval", type=float, default=1.0)
    
    # Checkpoint
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--monitor", type=str, default="val_mAP")
    parser.add_argument("--monitor_mode", type=str, default="max", choices=["max", "min"])
    
    # DDP
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--num_nodes", type=int, default=1)
    
    # Test
    parser.add_argument("--test_weight", type=str, default="")
    parser.add_argument("--rerank", action="store_true")
    
    return parser


def parse_args_to_config(args) -> Config:
    """Convert parsed arguments to Config object."""
    return Config(**vars(args))




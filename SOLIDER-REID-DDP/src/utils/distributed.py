"""
Distributed Training Utilities for Native PyTorch DDP
"""

import os
import torch
import torch.distributed as dist
from typing import Optional


def init_distributed(backend: str = 'nccl') -> bool:
    """
    Initialize distributed training environment

    Automatically detects SLURM or torchrun environment

    Returns:
        bool: True if distributed training is enabled
    """
    # Check if already initialized
    if dist.is_initialized():
        return True

    # Check for SLURM environment
    if 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])

        # Set master address and port
        if 'MASTER_ADDR' not in os.environ:
            node_list = os.environ.get('SLURM_NODELIST', 'localhost')
            # Parse first node from SLURM_NODELIST
            if '[' in node_list:
                # Format: node[001-004] -> node001
                import re
                match = re.match(r'(\w+)\[(\d+)', node_list)
                if match:
                    os.environ['MASTER_ADDR'] = f"{match.group(1)}{match.group(2)}"
                else:
                    os.environ['MASTER_ADDR'] = 'localhost'
            else:
                os.environ['MASTER_ADDR'] = node_list.split(',')[0]

        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'

        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['WORLD_SIZE'] = str(world_size)

    # Check for torchrun/torch.distributed.launch environment
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Single GPU mode
        return False

    # Set CUDA device
    torch.cuda.set_device(local_rank)

    # Initialize process group
    dist.init_process_group(
        backend=backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )

    # Synchronize all processes
    dist.barrier()

    return True


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)"""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def get_rank() -> int:
    """Get current process rank"""
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes"""
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_local_rank() -> int:
    """Get local rank (GPU index on this node)"""
    return int(os.environ.get('LOCAL_RANK', 0))


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduce tensor across all processes

    Args:
        tensor: Tensor to reduce
        average: If True, return average; otherwise return sum

    Returns:
        Reduced tensor
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    # Clone to avoid modifying original
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)

    if average:
        rt = rt / world_size

    return rt


def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Gather tensors from all processes

    Args:
        tensor: Tensor to gather

    Returns:
        Concatenated tensor from all processes
    """
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    if world_size == 1:
        return tensor

    # Create placeholder for gathered tensors
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    return torch.cat(tensor_list, dim=0)


def broadcast_object(obj, src: int = 0):
    """
    Broadcast object from source rank to all processes

    Args:
        obj: Object to broadcast
        src: Source rank

    Returns:
        Broadcasted object
    """
    if not dist.is_initialized():
        return obj

    object_list = [obj]
    dist.broadcast_object_list(object_list, src=src)
    return object_list[0]


def cleanup():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

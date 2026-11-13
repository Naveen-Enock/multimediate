"""DDP helpers — init from torchrun env, rank-0 gating.

Works single-process (no env vars) and across multiple GPUs under
`torchrun --standalone --nproc_per_node=N`.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup() -> dict:
    """Initialize the process group if launched under torchrun. Returns info."""
    if is_distributed():
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0

    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank % max(torch.cuda.device_count(), 1))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    return {"rank": rank, "world_size": world_size,
            "local_rank": local_rank, "device": device}


def is_rank0(info: dict) -> bool:
    return info["rank"] == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

"""Small host-side helpers shared by distributed tests and benchmarks."""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

import nvshmem.core as nvshmem


def broadcast_unique_id(rank: int) -> nvshmem.UniqueID:
    """Create one NVSHMEM unique ID and broadcast it through NCCL."""
    unique_id = (
        nvshmem.get_unique_id()
        if rank == 0
        else nvshmem.get_unique_id(empty=True)
    )
    unique_id_bytes = unique_id._data.view(np.uint8).reshape(-1)
    encoded = torch.empty(unique_id_bytes.size, dtype=torch.uint8, device="cuda")
    if rank == 0:
        encoded.copy_(torch.from_numpy(unique_id_bytes.copy()).to(device="cuda"))
    dist.broadcast(encoded, src=0)
    unique_id_bytes[:] = encoded.cpu().numpy()
    return unique_id


__all__ = ["broadcast_unique_id"]

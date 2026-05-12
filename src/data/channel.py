from __future__ import annotations

import torch

from core.complex_ops import complex_normal


def iid_rayleigh_channel(
    batch_size: int,
    num_users: int,
    num_tx_antennas: int,
    device: str | torch.device = "cpu",
    normalize: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Generate H with shape [B, K, Nt]."""
    h = complex_normal((batch_size, num_users, num_tx_antennas), device=device, dtype=dtype)
    if normalize:
        h = h / (num_tx_antennas**0.5)
    return h


class RayleighDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples: int, num_users: int, num_tx_antennas: int, seed: int = 0):
        gen = torch.Generator().manual_seed(seed)
        scale = (2 * num_tx_antennas) ** -0.5
        real = torch.randn(num_samples, num_users, num_tx_antennas, generator=gen) * scale
        imag = torch.randn(num_samples, num_users, num_tx_antennas, generator=gen) * scale
        self.h = torch.complex(real, imag)

    def __len__(self) -> int:
        return self.h.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.h[idx]

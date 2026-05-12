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


def pathloss_rayleigh_channel(
    batch_size: int,
    num_users: int,
    num_tx_antennas: int,
    pathloss_db_low: float = -5.0,
    pathloss_db_high: float = 5.0,
    device: str | torch.device = "cpu",
    normalize: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Rayleigh channel with per-user large-scale pathloss shift.

    Each user k receives an independent multiplicative factor sqrt(d_k),
    where d_k is drawn uniformly in dB. This mirrors common domain-shift
    tests in WMMSE unfolding papers.
    """
    h = iid_rayleigh_channel(
        batch_size,
        num_users,
        num_tx_antennas,
        device=device,
        normalize=normalize,
        dtype=dtype,
    )
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    pl_db = torch.empty(batch_size, num_users, device=device, dtype=real_dtype).uniform_(
        pathloss_db_low,
        pathloss_db_high,
    )
    gain = 10.0 ** (pl_db / 10.0)
    return h * torch.sqrt(gain).to(h.dtype)[:, :, None]


def benchmark_channel(
    *,
    batch_size: int,
    num_users: int,
    num_tx_antennas: int,
    channel: str = "iid_rayleigh",
    pathloss_db_low: float = 0.0,
    pathloss_db_high: float = 0.0,
    device: str | torch.device = "cpu",
    normalize: bool = True,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Factory for benchmark channel distributions."""
    if channel == "iid_rayleigh":
        return iid_rayleigh_channel(
            batch_size,
            num_users,
            num_tx_antennas,
            device=device,
            normalize=normalize,
            dtype=dtype,
        )
    if channel == "pathloss_rayleigh":
        return pathloss_rayleigh_channel(
            batch_size,
            num_users,
            num_tx_antennas,
            pathloss_db_low=pathloss_db_low,
            pathloss_db_high=pathloss_db_high,
            device=device,
            normalize=normalize,
            dtype=dtype,
        )
    raise ValueError(f"Unknown channel distribution: {channel}")


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

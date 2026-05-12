from __future__ import annotations

import torch


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert real-imag last dimension representation to a complex tensor."""
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError("Expected last dimension of size 2 for real-imag representation.")
    return torch.view_as_complex(x.contiguous())


def to_real_imag(x: torch.Tensor) -> torch.Tensor:
    """Convert a complex tensor to real-imag last dimension representation."""
    if not torch.is_complex(x):
        return x
    return torch.view_as_real(x)


def complex_normal(shape: tuple[int, ...], device=None, dtype=torch.complex64) -> torch.Tensor:
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    scale = 1.0 / 2.0**0.5
    real = torch.randn(*shape, device=device, dtype=real_dtype) * scale
    imag = torch.randn(*shape, device=device, dtype=real_dtype) * scale
    return torch.complex(real, imag).to(dtype)


def total_power(v: torch.Tensor) -> torch.Tensor:
    """Return per-sample total transmit power for V with shape [B, K, Nt]."""
    return torch.sum(torch.abs(v) ** 2, dim=(-1, -2))


def project_sum_power(v: torch.Tensor, p_max: float | torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Project V onto the sum-power ball ||V||_F^2 <= p_max, batch-wise."""
    p = total_power(v).clamp_min(eps)
    pmax = torch.as_tensor(p_max, device=v.device, dtype=p.real.dtype)
    scale = torch.sqrt(torch.minimum(torch.ones_like(p), pmax / p))
    return v * scale[:, None, None]


def effective_channels(h: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Return G[b,k,j] = h_{b,k}^H v_{b,j}."""
    return torch.einsum("bkn,bjn->bkj", h.conj(), v)


def sinr(h: torch.Tensor, v: torch.Tensor, noise_var: float | torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    g = effective_channels(h, v)
    power = torch.abs(g) ** 2
    desired = torch.diagonal(power, dim1=-2, dim2=-1)
    interference = power.sum(dim=-1) - desired
    sigma = torch.as_tensor(noise_var, device=h.device, dtype=desired.dtype)
    return desired / (interference + sigma + eps)


def weighted_sum_rate(
    h: torch.Tensor,
    v: torch.Tensor,
    noise_var: float | torch.Tensor,
    weights: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return per-sample weighted sum-rate in bit/s/Hz."""
    gamma = sinr(h, v, noise_var, eps=eps)
    if weights is None:
        weights = torch.ones_like(gamma)
    elif weights.ndim == 1:
        weights = weights[None, :].to(device=h.device, dtype=gamma.dtype)
    return torch.sum(weights * torch.log2(1.0 + gamma), dim=-1)

from __future__ import annotations

import torch

from core.complex_ops import project_sum_power


def mrt(h: torch.Tensor, p_max: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """Maximum-ratio transmission: v_k proportional to h_k."""
    v = h / torch.linalg.norm(h, dim=-1, keepdim=True).clamp_min(eps)
    return project_sum_power(v, p_max)


def zf(h: torch.Tensor, p_max: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    """Zero-forcing beamformer for K <= Nt."""
    gram = torch.matmul(h, h.conj().transpose(-1, -2))
    eye = torch.eye(gram.shape[-1], device=h.device, dtype=h.dtype)[None]
    inv = torch.linalg.pinv(gram + eps * eye)
    v_cols = torch.matmul(h.conj().transpose(-1, -2), inv)
    v = v_cols.transpose(-1, -2).contiguous()
    return project_sum_power(v, p_max)


def rzf(h: torch.Tensor, p_max: float = 1.0, reg: float = 1e-2) -> torch.Tensor:
    """Regularized zero-forcing beamformer."""
    gram = torch.matmul(h, h.conj().transpose(-1, -2))
    eye = torch.eye(gram.shape[-1], device=h.device, dtype=h.dtype)[None]
    inv = torch.linalg.solve(gram + reg * eye, eye.expand(h.shape[0], -1, -1))
    v_cols = torch.matmul(h.conj().transpose(-1, -2), inv)
    v = v_cols.transpose(-1, -2).contiguous()
    return project_sum_power(v, p_max)

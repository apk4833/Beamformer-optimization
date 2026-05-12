from __future__ import annotations

import torch

from core.complex_ops import weighted_sum_rate


def negative_wsr_loss(
    h: torch.Tensor,
    v: torch.Tensor,
    noise_var: float = 1e-3,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return -weighted_sum_rate(h, v, noise_var=noise_var, weights=weights).mean()


def supervised_beamformer_loss(
    pred_v: torch.Tensor,
    target_v: torch.Tensor,
    *,
    phase_align: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """MSE loss for complex beamformers.

    A per-stream common phase rotation does not change the stream power and is often
    weakly identifiable in supervised labels. Phase alignment avoids penalizing this
    equivalent representation when training a direct beamformer from WMMSE labels.
    """
    if not phase_align:
        return torch.mean(torch.abs(pred_v - target_v) ** 2)

    inner = torch.sum(target_v.conj() * pred_v, dim=-1, keepdim=True)
    phase = inner / inner.abs().clamp_min(eps)
    aligned_target = phase * target_v
    return torch.mean(torch.abs(pred_v - aligned_target) ** 2)

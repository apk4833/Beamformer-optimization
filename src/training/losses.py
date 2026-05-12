from __future__ import annotations

import torch

from mu_miso_bf_lab.core.complex_ops import weighted_sum_rate


def negative_wsr_loss(
    h: torch.Tensor,
    v: torch.Tensor,
    noise_var: float = 1e-3,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    return -weighted_sum_rate(h, v, noise_var=noise_var, weights=weights).mean()


def supervised_beamformer_loss(pred_v: torch.Tensor, target_v: torch.Tensor) -> torch.Tensor:
    # Phase ambiguity is not generally free for multiuser beamforming, so use direct complex MSE.
    return torch.mean(torch.abs(pred_v - target_v) ** 2)

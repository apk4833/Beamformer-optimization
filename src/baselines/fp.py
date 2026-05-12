from __future__ import annotations

import torch

from mu_miso_bf_lab.baselines.wmmse import wmmse


def fp_quadratic_transform(
    h: torch.Tensor,
    p_max: float = 1.0,
    noise_var: float = 1e-3,
    weights: torch.Tensor | None = None,
    max_iter: int = 50,
) -> torch.Tensor:
    """FP baseline entry point.

    For the single-cell MU-MISO WSR problem, WMMSE and quadratic-transform FP
    lead to closely related alternating auxiliary-variable updates. This scaffold
    keeps a dedicated FP API so FastFP/DeepFP variants can later be inserted
    without changing the experiment manager.
    """
    return wmmse(h, p_max=p_max, noise_var=noise_var, weights=weights, max_iter=max_iter)

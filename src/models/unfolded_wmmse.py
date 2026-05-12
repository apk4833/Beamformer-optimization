from __future__ import annotations

import torch
from torch import nn

from mu_miso_bf_lab.baselines.linear import rzf
from mu_miso_bf_lab.core.complex_ops import effective_channels, project_sum_power


class UnfoldedWMMSE(nn.Module):
    """Matrix-inverse-free unfolded WMMSE with trainable PGD step sizes.

    This follows the common unfoldable-WMMSE idea: exact auxiliary updates for
    receiver and MSE weights, followed by a small number of projected gradient
    steps for the transmit beamformer.
    """

    def __init__(self, num_layers: int = 5, pgd_steps: int = 3, init_step: float = 0.05):
        super().__init__()
        init = torch.full((num_layers, pgd_steps), float(init_step)).log()
        self.log_steps = nn.Parameter(init)
        self.num_layers = num_layers
        self.pgd_steps = pgd_steps

    def forward(
        self,
        h: torch.Tensor,
        p_max: float = 1.0,
        noise_var: float = 1e-3,
        weights: torch.Tensor | None = None,
        v0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, k, _ = h.shape
        if weights is None:
            alpha = torch.ones(batch, k, device=h.device, dtype=h.real.dtype)
        elif weights.ndim == 1:
            alpha = weights[None, :].expand(batch, -1).to(h.device, h.real.dtype)
        else:
            alpha = weights.to(h.device, h.real.dtype)

        v = rzf(h, p_max=p_max, reg=noise_var) if v0 is None else project_sum_power(v0, p_max)
        sigma = torch.as_tensor(noise_var, device=h.device, dtype=h.real.dtype)

        for ell in range(self.num_layers):
            g = effective_channels(h, v)
            recv_power = (torch.abs(g) ** 2).sum(dim=-1) + sigma
            desired = torch.diagonal(g, dim1=-2, dim2=-1)
            u = desired / recv_power.clamp_min(1e-9)
            mse = 1.0 - 2.0 * torch.real(u.conj() * desired) + (torch.abs(u) ** 2) * recv_power
            w = alpha / mse.clamp_min(1e-9)

            coeff = alpha * w * torch.abs(u) ** 2
            a = torch.einsum("bk,bkn,bkm->bnm", coeff.to(h.real.dtype), h, h.conj())
            target = (alpha * w * u.conj()).to(h.dtype)[:, :, None] * h

            for step_idx in range(self.pgd_steps):
                # grad f(v_k) = -2 alpha_k w_k u_k^* h_k + 2 A v_k
                av = torch.einsum("bnm,bkm->bkn", a, v)
                grad = -2.0 * target + 2.0 * av
                eta = torch.nn.functional.softplus(self.log_steps[ell, step_idx])
                v = project_sum_power(v - eta * grad, p_max)
        return v

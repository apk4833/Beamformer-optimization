from __future__ import annotations

import torch
from torch import nn

from baselines.linear import rzf
from core.complex_ops import effective_channels, project_sum_power


class UnfoldedWMMSE(nn.Module):
    """Matrix-inverse-free unfolded WMMSE with trainable PGD step sizes.

    The auxiliary receiver and MSE-weight updates follow WMMSE, while the
    beamformer update is replaced by a small number of projected-gradient steps.
    User weights alpha are applied exactly once in the PGD objective.
    """

    def __init__(self, num_layers: int = 5, pgd_steps: int = 3, init_step: float = 0.05):
        super().__init__()
        init = torch.full((num_layers, pgd_steps), float(init_step)).log()
        self.log_steps = nn.Parameter(init)
        self.num_layers = num_layers
        self.pgd_steps = pgd_steps

    def step_sizes(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.log_steps.detach())

    def forward(
        self,
        h: torch.Tensor,
        p_max: float = 1.0,
        noise_var: float = 1e-3,
        weights: torch.Tensor | None = None,
        v0: torch.Tensor | None = None,
        return_all_layers: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        batch, k, _ = h.shape
        if weights is None:
            alpha = torch.ones(batch, k, device=h.device, dtype=h.real.dtype)
        elif weights.ndim == 1:
            alpha = weights[None, :].expand(batch, -1).to(h.device, h.real.dtype)
        else:
            alpha = weights.to(h.device, h.real.dtype)

        v = rzf(h, p_max=p_max, reg=noise_var) if v0 is None else project_sum_power(v0, p_max)
        sigma = torch.as_tensor(noise_var, device=h.device, dtype=h.real.dtype)
        layer_outputs: list[torch.Tensor] = []

        for ell in range(self.num_layers):
            g = effective_channels(h, v)
            recv_power = (torch.abs(g) ** 2).sum(dim=-1) + sigma
            desired = torch.diagonal(g, dim1=-2, dim2=-1)
            u = desired / recv_power.clamp_min(1e-9)
            mse = 1.0 - 2.0 * torch.real(u.conj() * desired) + (torch.abs(u) ** 2) * recv_power

            # w is inverse MSE only; alpha is applied once below.
            w = 1.0 / mse.clamp_min(1e-9)

            coeff = alpha * w * torch.abs(u) ** 2
            a = torch.einsum("bk,bkn,bkm->bnm", coeff.to(h.dtype), h, h.conj())
            target = (alpha * w * u.conj()).to(h.dtype)[:, :, None] * h

            for step_idx in range(self.pgd_steps):
                av = torch.einsum("bnm,bkm->bkn", a, v)
                grad = -2.0 * target + 2.0 * av
                eta = torch.nn.functional.softplus(self.log_steps[ell, step_idx])
                v = project_sum_power(v - eta * grad, p_max)

            if return_all_layers:
                layer_outputs.append(v)

        if return_all_layers:
            return v, layer_outputs
        return v

from __future__ import annotations

import torch

from baselines.linear import rzf
from core.complex_ops import effective_channels, project_sum_power, total_power, weighted_sum_rate


def _solve_beamformer(
    a: torch.Tensor,
    b: torch.Tensor,
    p_max: float,
    eps: float = 1e-8,
    bisection_steps: int = 30,
) -> torch.Tensor:
    """Solve V(mu)=(A+mu I)^-1 B and pick mu to satisfy sum-power."""
    batch, nt, _ = a.shape
    eye = torch.eye(nt, device=a.device, dtype=a.dtype)[None]

    def solve_for(mu: torch.Tensor) -> torch.Tensor:
        mat = a + (mu[:, None, None] + eps) * eye
        v_cols = torch.linalg.solve(mat, b)
        return v_cols.transpose(-1, -2).contiguous()

    mu0 = torch.zeros(batch, device=a.device, dtype=a.real.dtype)
    v0 = solve_for(mu0)
    ok = total_power(v0) <= p_max
    if bool(ok.all()):
        return v0

    lo = torch.zeros_like(mu0)
    hi = torch.ones_like(mu0)
    for _ in range(30):
        v_hi = solve_for(hi)
        enough = total_power(v_hi) <= p_max
        if bool(enough.all()):
            break
        hi = torch.where(enough, hi, hi * 2.0)

    for _ in range(bisection_steps):
        mid = 0.5 * (lo + hi)
        v_mid = solve_for(mid)
        too_much = total_power(v_mid) > p_max
        lo = torch.where(too_much, mid, lo)
        hi = torch.where(too_much, hi, mid)

    v = solve_for(hi)
    return torch.where(ok[:, None, None], v0, v)


@torch.no_grad()
def wmmse(
    h: torch.Tensor,
    p_max: float = 1.0,
    noise_var: float = 1e-3,
    weights: torch.Tensor | None = None,
    max_iter: int = 50,
    tol: float = 1e-5,
    eps: float = 1e-9,
    init: str = "rzf",
) -> torch.Tensor:
    """Classical weighted MMSE solver for single-cell MU-MISO WSR maximization.

    Shape convention: H and V are [B, K, Nt]. User weights alpha are applied
    exactly once in the beamformer update. The auxiliary variable w is the
    inverse MSE, not alpha / MSE.
    """
    batch, k, _ = h.shape
    if weights is None:
        alpha = torch.ones(batch, k, device=h.device, dtype=h.real.dtype)
    elif weights.ndim == 1:
        alpha = weights[None, :].expand(batch, -1).to(h.device, h.real.dtype)
    else:
        alpha = weights.to(h.device, h.real.dtype)

    if init == "rzf":
        v = rzf(h, p_max=p_max, reg=noise_var)
    else:
        v = project_sum_power(torch.randn_like(h), p_max)

    prev = weighted_sum_rate(h, v, noise_var, alpha).mean()
    sigma = torch.as_tensor(noise_var, device=h.device, dtype=h.real.dtype)

    for _ in range(max_iter):
        g = effective_channels(h, v)
        recv_power = (torch.abs(g) ** 2).sum(dim=-1) + sigma
        desired = torch.diagonal(g, dim1=-2, dim2=-1)
        u = desired / recv_power.clamp_min(eps)
        mse = 1.0 - 2.0 * torch.real(u.conj() * desired) + (torch.abs(u) ** 2) * recv_power

        # w is inverse MSE only; alpha is applied once below.
        w = 1.0 / mse.clamp_min(eps)

        coeff = alpha * w * torch.abs(u) ** 2
        a = torch.einsum("bk,bkn,bkm->bnm", coeff.to(h.dtype), h, h.conj())
        rhs = (alpha * w * u.conj()).to(h.dtype)[:, :, None] * h
        b = rhs.transpose(-1, -2).contiguous()
        v = _solve_beamformer(a, b, p_max=p_max, eps=eps)

        cur = weighted_sum_rate(h, v, noise_var, alpha).mean()
        if torch.abs(cur - prev) <= tol * torch.maximum(torch.ones_like(prev), torch.abs(prev)):
            break
        prev = cur
    return project_sum_power(v, p_max)

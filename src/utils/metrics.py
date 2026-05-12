from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from core.complex_ops import sinr, total_power


def gap_to_reference(value: float, reference: float, eps: float = 1e-12) -> float:
    return float((reference - value) / max(abs(reference), eps))


def relative_to_reference_percent(value: float, reference: float, eps: float = 1e-12) -> float:
    return float(100.0 * value / max(abs(reference), eps))


def tensor_summary(x: torch.Tensor, prefix: str) -> dict[str, float]:
    """Return plot-friendly summary statistics for a tensor."""
    x = x.detach().flatten().to(torch.float64).cpu()
    if x.numel() == 0:
        return {
            f"{prefix}_n": 0.0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_p50": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_n": float(x.numel()),
        f"{prefix}_mean": x.mean().item(),
        f"{prefix}_std": x.std(unbiased=False).item(),
        f"{prefix}_min": x.min().item(),
        f"{prefix}_p10": torch.quantile(x, 0.10).item(),
        f"{prefix}_p50": torch.quantile(x, 0.50).item(),
        f"{prefix}_p90": torch.quantile(x, 0.90).item(),
        f"{prefix}_max": x.max().item(),
    }


def per_user_rates(
    h: torch.Tensor,
    v: torch.Tensor,
    noise_var: float | torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return per-user spectral efficiency in bit/s/Hz with shape [B, K]."""
    gamma = sinr(h, v, noise_var, eps=eps)
    return torch.log2(1.0 + gamma)


def jain_fairness(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    x = x.to(torch.float64)
    n = x.shape[dim]
    return (x.sum(dim=dim) ** 2) / (n * torch.sum(x**2, dim=dim).clamp_min(eps))


def beamformer_summary(v: torch.Tensor, p_max: float, eps: float = 1e-9) -> dict[str, float]:
    p = total_power(v).detach()
    return {
        "mean_power": p.mean().item(),
        "std_power": p.to(torch.float64).std(unbiased=False).item(),
        "max_power": p.max().item(),
        "power_utilization": (p / max(float(p_max), eps)).mean().item(),
        "power_violation_ratio": (p > p_max + eps).float().mean().item(),
    }


def evaluate_beamformer(
    h: torch.Tensor,
    v: torch.Tensor,
    noise_var: float | torch.Tensor,
    p_max: float,
    weights: torch.Tensor | None = None,
    reference_wsr: torch.Tensor | float | None = None,
    rate_threshold: float | None = None,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Compute scalar metrics for publication-style beamforming plots."""
    rates = per_user_rates(h, v, noise_var, eps=eps)
    if weights is None:
        weights = torch.ones_like(rates)
    elif weights.ndim == 1:
        weights = weights[None, :].to(device=h.device, dtype=rates.dtype)
    else:
        weights = weights.to(device=h.device, dtype=rates.dtype)

    wsr = torch.sum(weights * rates, dim=-1)
    gamma = sinr(h, v, noise_var, eps=eps)
    min_rate = rates.min(dim=-1).values
    fairness = jain_fairness(rates, dim=-1)

    metrics: dict[str, float] = {}
    metrics.update(tensor_summary(wsr, "wsr"))
    metrics.update(tensor_summary(rates, "user_rate"))
    metrics.update(tensor_summary(min_rate, "min_user_rate"))
    metrics.update(tensor_summary(10.0 * torch.log10(gamma.clamp_min(eps)), "sinr_db"))
    metrics.update(tensor_summary(fairness, "jain"))
    metrics.update(beamformer_summary(v, p_max=p_max))

    metrics["mean_wsr"] = metrics["wsr_mean"]
    metrics["std_wsr"] = metrics["wsr_std"]
    metrics["p10_wsr"] = metrics["wsr_p10"]
    metrics["p50_wsr"] = metrics["wsr_p50"]
    metrics["p90_wsr"] = metrics["wsr_p90"]
    metrics["mean_user_rate"] = metrics["user_rate_mean"]
    metrics["mean_min_user_rate"] = metrics["min_user_rate_mean"]
    metrics["mean_jain_fairness"] = metrics["jain_mean"]

    if rate_threshold is not None:
        metrics["user_rate_outage_ratio"] = (rates < rate_threshold).float().mean().item()
        metrics["sample_min_rate_outage_ratio"] = (min_rate < rate_threshold).float().mean().item()

    if reference_wsr is not None:
        if not isinstance(reference_wsr, torch.Tensor):
            ref = torch.full_like(wsr, float(reference_wsr))
        else:
            ref = reference_wsr.to(device=wsr.device, dtype=wsr.dtype)
            if ref.ndim == 0:
                ref = torch.full_like(wsr, float(ref.item()))
        ref_mean = ref.mean().item()
        diff = ref - wsr
        metrics["reference_mean_wsr"] = ref_mean
        metrics["gap_to_reference"] = gap_to_reference(wsr.mean().item(), ref_mean, eps=eps)
        metrics["relative_wsr_percent"] = relative_to_reference_percent(wsr.mean().item(), ref_mean, eps=eps)
        metrics["mean_abs_wsr_gap"] = diff.abs().mean().item()
        metrics["p90_abs_wsr_gap"] = torch.quantile(diff.abs().detach().flatten().cpu(), 0.90).item()
        metrics["win_rate_vs_reference"] = (wsr > ref).float().mean().item()

    return metrics


def compact_metric_row(metrics: dict, keep_prefixes: tuple[str, ...] = ()) -> dict:
    common = {
        "mean_wsr",
        "std_wsr",
        "p10_wsr",
        "p50_wsr",
        "p90_wsr",
        "mean_user_rate",
        "mean_min_user_rate",
        "mean_jain_fairness",
        "mean_power",
        "power_utilization",
        "power_violation_ratio",
        "gap_to_reference",
        "relative_wsr_percent",
        "mean_abs_wsr_gap",
        "p90_abs_wsr_gap",
        "win_rate_vs_reference",
        "runtime_ms_per_sample",
    }
    return {
        k: v
        for k, v in metrics.items()
        if k in common or any(k.startswith(prefix) for prefix in keep_prefixes)
    }


def parameter_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    sq_sum = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        sq_sum += float(torch.sum(p.grad.detach() ** 2).item())
    return math.sqrt(sq_sum)

from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from core.complex_ops import total_power


def gap_to_reference(value: float, reference: float, eps: float = 1e-12) -> float:
    return float((reference - value) / max(abs(reference), eps))


def beamformer_summary(v: torch.Tensor, p_max: float, eps: float = 1e-9) -> dict[str, float]:
    p = total_power(v).detach()
    return {
        "mean_power": p.mean().item(),
        "max_power": p.max().item(),
        "power_violation_ratio": (p > p_max + eps).float().mean().item(),
    }


def parameter_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    sq_sum = 0.0
    for p in parameters:
        if p.grad is None:
            continue
        sq_sum += float(torch.sum(p.grad.detach() ** 2).item())
    return math.sqrt(sq_sum)

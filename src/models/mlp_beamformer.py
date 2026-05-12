from __future__ import annotations

import torch
from torch import nn

from core.complex_ops import project_sum_power, to_real_imag


class DirectMLPBeamformer(nn.Module):
    """Simple end-to-end baseline: H -> V, with power projection."""

    def __init__(self, num_users: int, num_tx_antennas: int, hidden_dim: int = 512, depth: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_tx_antennas = num_tx_antennas
        dim = 2 * num_users * num_tx_antennas
        layers: list[nn.Module] = [nn.Linear(dim, hidden_dim), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
        layers.append(nn.Linear(hidden_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, h: torch.Tensor, p_max: float = 1.0) -> torch.Tensor:
        batch_size = h.shape[0]
        x = to_real_imag(h).reshape(batch_size, -1)
        y = self.net(x).reshape(batch_size, self.num_users, self.num_tx_antennas, 2)
        v = torch.view_as_complex(y.contiguous())
        return project_sum_power(v, p_max)

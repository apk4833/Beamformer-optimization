from __future__ import annotations

import torch

from mu_miso_bf_lab.core.complex_ops import project_sum_power, weighted_sum_rate
from mu_miso_bf_lab.data.channel import iid_rayleigh_channel


class StaticMISOBeamformingEnv:
    """Minimal continuous-action environment for direct beamformer policies.

    This is intentionally simple: every step samples a new iid channel and the
    reward is WSR. It is useful only as a direct-BF RL baseline, not as a queue or
    hybrid-action environment.
    """

    def __init__(
        self,
        num_users: int = 4,
        num_tx_antennas: int = 8,
        p_max: float = 1.0,
        noise_var: float = 1e-3,
        device: str | torch.device = "cpu",
    ):
        self.k = num_users
        self.nt = num_tx_antennas
        self.p_max = p_max
        self.noise_var = noise_var
        self.device = torch.device(device)
        self.h = None

    @property
    def state_dim(self) -> int:
        return 2 * self.k * self.nt

    @property
    def action_dim(self) -> int:
        return 2 * self.k * self.nt

    def reset(self) -> torch.Tensor:
        self.h = iid_rayleigh_channel(1, self.k, self.nt, self.device)
        return torch.view_as_real(self.h).flatten()

    def step(self, action: torch.Tensor):
        if self.h is None:
            self.reset()
        v = action.reshape(1, self.k, self.nt, 2)
        v = torch.view_as_complex(v.contiguous())
        v = project_sum_power(v, self.p_max)
        reward = weighted_sum_rate(self.h, v, self.noise_var).item()
        next_state = self.reset()
        done = False
        info = {"wsr": reward}
        return next_state, reward, done, info

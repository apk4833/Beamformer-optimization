from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mu_miso_bf_lab.baselines.linear import mrt, rzf, zf
from mu_miso_bf_lab.baselines.wmmse import wmmse
from mu_miso_bf_lab.core.complex_ops import weighted_sum_rate
from mu_miso_bf_lab.data.channel import iid_rayleigh_channel
from mu_miso_bf_lab.models.unfolded_wmmse import UnfoldedWMMSE


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h = iid_rayleigh_channel(batch_size=32, num_users=4, num_tx_antennas=8, device=device)
    for name, fn in [("MRT", mrt), ("ZF", zf), ("RZF", rzf)]:
        v = fn(h, p_max=1.0)
        print(f"{name:>5s} WSR:", weighted_sum_rate(h, v, 1e-3).mean().item())
    v_wmmse = wmmse(h, p_max=1.0, noise_var=1e-3, max_iter=10)
    print("WMMSE WSR:", weighted_sum_rate(h, v_wmmse, 1e-3).mean().item())
    model = UnfoldedWMMSE(num_layers=2, pgd_steps=2).to(device)
    v_unfold = model(h, p_max=1.0, noise_var=1e-3)
    print("Unfolded init WSR:", weighted_sum_rate(h, v_unfold, 1e-3).mean().item())


if __name__ == "__main__":
    main()

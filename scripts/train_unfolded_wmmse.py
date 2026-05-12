from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import argparse

import torch
from torch.optim import Adam
from tqdm import trange

from mu_miso_bf_lab.core.complex_ops import weighted_sum_rate
from mu_miso_bf_lab.data.channel import iid_rayleigh_channel
from mu_miso_bf_lab.models.unfolded_wmmse import UnfoldedWMMSE
from mu_miso_bf_lab.training.losses import negative_wsr_loss
from mu_miso_bf_lab.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--num-tx-antennas", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--p-max", type=float, default=1.0)
    parser.add_argument("--noise-var", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UnfoldedWMMSE(num_layers=5, pgd_steps=3, init_step=0.05).to(device)
    opt = Adam(model.parameters(), lr=args.lr)

    for step in trange(args.steps):
        h = iid_rayleigh_channel(args.batch_size, args.num_users, args.num_tx_antennas, device=device)
        v = model(h, p_max=args.p_max, noise_var=args.noise_var)
        loss = negative_wsr_loss(h, v, noise_var=args.noise_var)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % 100 == 0:
            print({"step": step, "wsr": weighted_sum_rate(h, v, args.noise_var).mean().item()})


if __name__ == "__main__":
    main()

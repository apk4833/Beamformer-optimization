from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.optim import Adam
from tqdm import trange

from baselines.wmmse import wmmse
from core.complex_ops import weighted_sum_rate
from data.channel import iid_rayleigh_channel
from models.mlp_beamformer import DirectMLPBeamformer
from training.losses import supervised_beamformer_loss
from utils.logger import CSVLogger, make_run_dir, save_json
from utils.metrics import gap_to_reference, parameter_grad_norm
from utils.seed import set_seed


def evaluate(
    model: DirectMLPBeamformer,
    *,
    batch_size: int,
    num_users: int,
    num_tx_antennas: int,
    p_max: float,
    noise_var: float,
    wmmse_iters: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        h = iid_rayleigh_channel(batch_size, num_users, num_tx_antennas, device=device)
        target = wmmse(h, p_max=p_max, noise_var=noise_var, max_iter=wmmse_iters)
        pred = model(h, p_max=p_max)
        mlp_wsr = weighted_sum_rate(h, pred, noise_var).mean().item()
        wmmse_wsr = weighted_sum_rate(h, target, noise_var).mean().item()
    model.train()
    return {
        "eval_mlp_mean_wsr": mlp_wsr,
        "eval_wmmse_mean_wsr": wmmse_wsr,
        "eval_gap_to_wmmse": gap_to_reference(mlp_wsr, wmmse_wsr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--num-tx-antennas", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--p-max", type=float, default=1.0)
    parser.add_argument("--noise-var", type=float, default=1e-3)
    parser.add_argument("--wmmse-iters", type=int, default=30)
    parser.add_argument("--loss-mode", choices=["mse", "mixed"], default="mse")
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--wsr-weight", type=float, default=0.1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--output-root", type=str, default="outputs/runs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_dir = make_run_dir(
        args.output_root,
        prefix=f"mlp_k{args.num_users}_nt{args.num_tx_antennas}_seed{args.seed}",
    )
    save_json(run_dir / "config.json", vars(args))
    logger = CSVLogger(run_dir / "metrics.csv")

    model = DirectMLPBeamformer(args.num_users, args.num_tx_antennas).to(device)
    opt = Adam(model.parameters(), lr=args.lr)

    for step in trange(args.steps + 1):
        h = iid_rayleigh_channel(
            args.batch_size,
            args.num_users,
            args.num_tx_antennas,
            device=device,
        )
        with torch.no_grad():
            target = wmmse(
                h,
                p_max=args.p_max,
                noise_var=args.noise_var,
                max_iter=args.wmmse_iters,
            )

        pred = model(h, p_max=args.p_max)
        mse_loss = supervised_beamformer_loss(pred, target, phase_align=True)
        pred_wsr = weighted_sum_rate(h, pred, args.noise_var).mean()
        wmmse_wsr = weighted_sum_rate(h, target, args.noise_var).mean()

        if args.loss_mode == "mse":
            loss = mse_loss
        else:
            loss = args.mse_weight * mse_loss - args.wsr_weight * pred_wsr

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % args.log_every == 0:
            row = {
                "step": step,
                "loss": loss.item(),
                "mse_loss": mse_loss.item(),
                "train_mlp_mean_wsr": pred_wsr.item(),
                "train_wmmse_mean_wsr": wmmse_wsr.item(),
                "train_gap_to_wmmse": gap_to_reference(pred_wsr.item(), wmmse_wsr.item()),
                "grad_norm": parameter_grad_norm(model.parameters()),
            }
            if step % args.eval_every == 0:
                row.update(
                    evaluate(
                        model,
                        batch_size=args.eval_batch_size,
                        num_users=args.num_users,
                        num_tx_antennas=args.num_tx_antennas,
                        p_max=args.p_max,
                        noise_var=args.noise_var,
                        wmmse_iters=args.wmmse_iters,
                        device=device,
                    )
                )
            logger.write(row)
            print(row)

        if step > 0 and step % args.checkpoint_every == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "model_type": "mlp",
                },
                run_dir / "checkpoint.pt",
            )


if __name__ == "__main__":
    main()

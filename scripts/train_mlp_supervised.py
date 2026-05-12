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
from data.channel import benchmark_channel
from experiments.scenarios import get_scenario_set
from models.mlp_beamformer import DirectMLPBeamformer
from training.losses import supervised_beamformer_loss
from utils.logger import CSVLogger, make_run_dir, save_json
from utils.metrics import compact_metric_row, evaluate_beamformer, gap_to_reference, parameter_grad_norm
from utils.seed import set_seed


def _select_scenario(args: argparse.Namespace):
    scenarios = get_scenario_set(args.scenario_set)
    for scenario in scenarios:
        if scenario.name == args.scenario_name:
            return scenario
    available = ", ".join(s.name for s in scenarios)
    raise ValueError(f"Unknown scenario_name={args.scenario_name}. Available: {available}")


def evaluate(
    model: DirectMLPBeamformer,
    scenario,
    *,
    batch_size: int,
    wmmse_iters: int,
    device: str,
    rate_threshold: float,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        h = benchmark_channel(
            batch_size=batch_size,
            num_users=scenario.num_users,
            num_tx_antennas=scenario.num_tx_antennas,
            channel=scenario.channel,
            pathloss_db_low=scenario.pathloss_db_low,
            pathloss_db_high=scenario.pathloss_db_high,
            device=device,
        )
        target = wmmse(h, p_max=scenario.p_max, noise_var=scenario.noise_var, max_iter=wmmse_iters)
        pred = model(h, p_max=scenario.p_max)
        pred_wsr_tensor = weighted_sum_rate(h, pred, scenario.noise_var)
        target_wsr_tensor = weighted_sum_rate(h, target, scenario.noise_var)
        pred_wsr = pred_wsr_tensor.mean().item()
        target_wsr = target_wsr_tensor.mean().item()
        row = {
            "eval_mlp_mean_wsr": pred_wsr,
            "eval_wmmse_mean_wsr": target_wsr,
            "eval_gap_to_wmmse": gap_to_reference(pred_wsr, target_wsr),
            "eval_relative_wsr_percent": 100.0 * pred_wsr / max(abs(target_wsr), 1e-12),
        }
        detailed = evaluate_beamformer(
            h,
            pred,
            scenario.noise_var,
            p_max=scenario.p_max,
            reference_wsr=target_wsr_tensor,
            rate_threshold=rate_threshold,
        )
        row.update({f"eval_{k}": v for k, v in compact_metric_row(detailed).items()})
    model.train()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-set", choices=["literature_core", "sweep", "all"], default="literature_core")
    parser.add_argument("--scenario-name", type=str, default="fully_loaded_snr10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--wmmse-iters", type=int, default=30)
    parser.add_argument("--loss-mode", choices=["mse", "mixed"], default="mixed")
    parser.add_argument("--mse-weight", type=float, default=1.0)
    parser.add_argument("--wsr-weight", type=float, default=0.1)
    parser.add_argument("--rate-threshold", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--output-root", type=str, default="outputs/runs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenario = _select_scenario(args)

    run_dir = make_run_dir(
        args.output_root,
        prefix=f"mlp_{scenario.name}_seed{args.seed}",
    )
    save_json(
        run_dir / "config.json",
        {
            **vars(args),
            "device": device,
            "scenario": scenario.to_dict(),
            "model_type": "mlp",
        },
    )
    logger = CSVLogger(run_dir / "training_metrics.csv")

    model = DirectMLPBeamformer(scenario.num_users, scenario.num_tx_antennas).to(device)
    opt = Adam(model.parameters(), lr=args.lr)

    best_eval_wsr = float("-inf")
    for step in trange(args.steps + 1):
        h = benchmark_channel(
            batch_size=args.batch_size,
            num_users=scenario.num_users,
            num_tx_antennas=scenario.num_tx_antennas,
            channel=scenario.channel,
            pathloss_db_low=scenario.pathloss_db_low,
            pathloss_db_high=scenario.pathloss_db_high,
            device=device,
        )
        with torch.no_grad():
            target = wmmse(
                h,
                p_max=scenario.p_max,
                noise_var=scenario.noise_var,
                max_iter=args.wmmse_iters,
            )

        pred = model(h, p_max=scenario.p_max)
        mse_loss = supervised_beamformer_loss(pred, target, phase_align=True)
        pred_wsr = weighted_sum_rate(h, pred, scenario.noise_var).mean()
        wmmse_wsr = weighted_sum_rate(h, target, scenario.noise_var).mean()

        if args.loss_mode == "mse":
            loss = args.mse_weight * mse_loss
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
                eval_row = evaluate(
                    model,
                    scenario,
                    batch_size=args.eval_batch_size,
                    wmmse_iters=args.wmmse_iters,
                    device=device,
                    rate_threshold=args.rate_threshold,
                )
                row.update(eval_row)
                if row["eval_mlp_mean_wsr"] > best_eval_wsr:
                    best_eval_wsr = row["eval_mlp_mean_wsr"]
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "args": vars(args),
                            "scenario": scenario.to_dict(),
                            "model_type": "mlp",
                            "best_eval_wsr": best_eval_wsr,
                        },
                        run_dir / "checkpoint_best.pt",
                    )
            logger.write(row)
            print(row)

        if step > 0 and step % args.checkpoint_every == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "args": vars(args),
                    "scenario": scenario.to_dict(),
                    "model_type": "mlp",
                },
                run_dir / "checkpoint_last.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "scenario": scenario.to_dict(),
            "model_type": "mlp",
        },
        run_dir / "checkpoint_last.pt",
    )


if __name__ == "__main__":
    main()

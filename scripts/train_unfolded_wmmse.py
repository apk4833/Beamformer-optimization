from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.optim import Adam
from tqdm import trange

from baselines.linear import rzf
from baselines.wmmse import wmmse
from core.complex_ops import weighted_sum_rate
from data.channel import benchmark_channel
from experiments.scenarios import get_scenario_set
from models.unfolded_wmmse import UnfoldedWMMSE
from training.losses import negative_wsr_loss
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


def _step_size_stats(model: UnfoldedWMMSE) -> dict[str, float]:
    eta = model.step_sizes().detach().cpu()
    return {
        "eta_min": eta.min().item(),
        "eta_mean": eta.mean().item(),
        "eta_max": eta.max().item(),
    }


def evaluate(
    model: UnfoldedWMMSE,
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
        v_unfolded, layer_outputs = model(
            h,
            p_max=scenario.p_max,
            noise_var=scenario.noise_var,
            return_all_layers=True,
        )
        v_ref = wmmse(
            h,
            p_max=scenario.p_max,
            noise_var=scenario.noise_var,
            max_iter=wmmse_iters,
        )
        v_rzf = rzf(h, p_max=scenario.p_max, reg=scenario.noise_var)
        ref_wsr = weighted_sum_rate(h, v_ref, scenario.noise_var)
        unfolded_wsr = weighted_sum_rate(h, v_unfolded, scenario.noise_var).mean().item()
        ref_mean = ref_wsr.mean().item()
        rzf_wsr = weighted_sum_rate(h, v_rzf, scenario.noise_var).mean().item()

        row = {
            "eval_unfolded_mean_wsr": unfolded_wsr,
            "eval_wmmse_mean_wsr": ref_mean,
            "eval_rzf_mean_wsr": rzf_wsr,
            "eval_gap_to_wmmse": gap_to_reference(unfolded_wsr, ref_mean),
            "eval_relative_wsr_percent": 100.0 * unfolded_wsr / max(abs(ref_mean), 1e-12),
            **_step_size_stats(model),
        }
        detailed = evaluate_beamformer(
            h,
            v_unfolded,
            scenario.noise_var,
            p_max=scenario.p_max,
            reference_wsr=ref_wsr,
            rate_threshold=rate_threshold,
        )
        row.update({f"eval_{k}": v for k, v in compact_metric_row(detailed).items()})
        for idx, v_layer in enumerate(layer_outputs, start=1):
            layer_wsr = weighted_sum_rate(h, v_layer, scenario.noise_var).mean().item()
            row[f"eval_layer_{idx}_wsr"] = layer_wsr
            row[f"eval_layer_{idx}_relative_percent"] = 100.0 * layer_wsr / max(abs(ref_mean), 1e-12)
    model.train()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-set", choices=["literature_core", "sweep", "all"], default="literature_core")
    parser.add_argument("--scenario-name", type=str, default="fully_loaded_snr10")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-layers", type=int, default=5)
    parser.add_argument("--pgd-steps", type=int, default=3)
    parser.add_argument("--init-step", type=float, default=0.05)
    parser.add_argument("--wmmse-iters", type=int, default=50)
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
        prefix=f"unfolded_{scenario.name}_L{args.num_layers}_K{args.pgd_steps}_seed{args.seed}",
    )
    save_json(
        run_dir / "config.json",
        {
            **vars(args),
            "device": device,
            "scenario": scenario.to_dict(),
            "model_type": "unfolded",
        },
    )
    logger = CSVLogger(run_dir / "training_metrics.csv")

    model = UnfoldedWMMSE(
        num_layers=args.num_layers,
        pgd_steps=args.pgd_steps,
        init_step=args.init_step,
    ).to(device)
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
        v, layer_outputs = model(
            h,
            p_max=scenario.p_max,
            noise_var=scenario.noise_var,
            return_all_layers=True,
        )
        # Auxiliary layer-wise WSR loss follows common unfolding practice.
        loss = sum(
            negative_wsr_loss(h, v_layer, noise_var=scenario.noise_var)
            for v_layer in layer_outputs
        ) / max(len(layer_outputs), 1)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % args.log_every == 0:
            train_wsr = weighted_sum_rate(h, v, scenario.noise_var).mean().item()
            row = {
                "step": step,
                "loss": loss.item(),
                "train_unfolded_mean_wsr": train_wsr,
                "grad_norm": parameter_grad_norm(model.parameters()),
                **_step_size_stats(model),
            }
            for idx, v_layer in enumerate(layer_outputs, start=1):
                row[f"train_layer_{idx}_wsr"] = weighted_sum_rate(h, v_layer, scenario.noise_var).mean().item()

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
                if row["eval_unfolded_mean_wsr"] > best_eval_wsr:
                    best_eval_wsr = row["eval_unfolded_mean_wsr"]
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "args": vars(args),
                            "scenario": scenario.to_dict(),
                            "model_type": "unfolded",
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
                    "model_type": "unfolded",
                },
                run_dir / "checkpoint_last.pt",
            )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "scenario": scenario.to_dict(),
            "model_type": "unfolded",
        },
        run_dir / "checkpoint_last.pt",
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from baselines.linear import mrt, rzf, zf
from baselines.wmmse import wmmse
from core.complex_ops import weighted_sum_rate
from data.channel import benchmark_channel
from experiments.scenarios import BenchmarkScenario, get_scenario_set
from utils.logger import CSVLogger, make_run_dir, save_json
from utils.metrics import compact_metric_row, evaluate_beamformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-set", choices=["literature_core", "sweep", "all"], default="literature_core")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--wmmse-iters", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 10, 30, 50])
    parser.add_argument("--reference-wmmse-iters", type=int, default=50)
    parser.add_argument("--rate-threshold", type=float, default=1.0)
    parser.add_argument("--output-root", type=str, default="outputs/evals")
    return parser.parse_args()


def _runtime_ms_per_sample(fn, device: str, batch_size: int) -> tuple[torch.Tensor, float]:
    start = time.perf_counter()
    v = fn()
    if device == "cuda":
        torch.cuda.synchronize()
    return v, (time.perf_counter() - start) * 1000.0 / batch_size


def evaluate_one_scenario(
    scenario: BenchmarkScenario,
    *,
    seed: int,
    batch_size: int,
    num_batches: int,
    wmmse_iters: list[int],
    reference_wmmse_iters: int,
    rate_threshold: float,
    device: str,
) -> tuple[list[dict], list[dict]]:
    torch.manual_seed(seed)
    raw_rows: list[dict] = []
    aggregate_values: dict[str, list[float]] = defaultdict(list)
    aggregate_runtime: dict[str, list[float]] = defaultdict(list)
    aggregate_metrics_acc: dict[str, list[dict[str, float]]] = defaultdict(list)

    for batch_idx in range(num_batches):
        h = benchmark_channel(
            batch_size=batch_size,
            num_users=scenario.num_users,
            num_tx_antennas=scenario.num_tx_antennas,
            channel=scenario.channel,
            pathloss_db_low=scenario.pathloss_db_low,
            pathloss_db_high=scenario.pathloss_db_high,
            device=device,
        )
        with torch.no_grad():
            v_ref, ref_runtime = _runtime_ms_per_sample(
                lambda: wmmse(
                    h,
                    p_max=scenario.p_max,
                    noise_var=scenario.noise_var,
                    max_iter=reference_wmmse_iters,
                ),
                device,
                batch_size,
            )
            ref_wsr = weighted_sum_rate(h, v_ref, scenario.noise_var)

            methods = {
                "MRT": lambda: mrt(h, p_max=scenario.p_max),
                "ZF": lambda: zf(h, p_max=scenario.p_max),
                "RZF": lambda: rzf(h, p_max=scenario.p_max, reg=scenario.noise_var),
            }
            for iters in wmmse_iters:
                methods[f"WMMSE-{iters}"] = lambda iters=iters: wmmse(
                    h,
                    p_max=scenario.p_max,
                    noise_var=scenario.noise_var,
                    max_iter=iters,
                )

            # Always include the reference as its own method, even when it is
            # already in --wmmse-iters, so summary tables have an explicit anchor.
            methods[f"WMMSE-ref-{reference_wmmse_iters}"] = lambda: v_ref

            for method, fn in methods.items():
                if method.startswith("WMMSE-ref"):
                    v = v_ref
                    runtime_ms = ref_runtime
                else:
                    v, runtime_ms = _runtime_ms_per_sample(fn, device, batch_size)

                metrics = evaluate_beamformer(
                    h,
                    v,
                    scenario.noise_var,
                    p_max=scenario.p_max,
                    reference_wsr=ref_wsr,
                    rate_threshold=rate_threshold,
                )
                metrics["runtime_ms_per_sample"] = runtime_ms
                metrics["batch_idx"] = float(batch_idx)

                wsr = weighted_sum_rate(h, v, scenario.noise_var).detach().cpu()
                aggregate_values[method].extend(wsr.tolist())
                aggregate_runtime[method].append(runtime_ms)
                aggregate_metrics_acc[method].append(metrics)

                raw_rows.append(
                    {
                        "scenario": scenario.name,
                        "seed": seed,
                        "method": method,
                        **scenario.to_dict(),
                        **compact_metric_row(metrics),
                    }
                )

    summary_rows: list[dict] = []
    ref_name = f"WMMSE-ref-{reference_wmmse_iters}"
    ref_mean = torch.tensor(aggregate_values[ref_name], dtype=torch.float64).mean().item()

    for method, values in aggregate_values.items():
        x = torch.tensor(values, dtype=torch.float64)
        merged: dict[str, float] = {
            "mean_wsr": x.mean().item(),
            "std_wsr": x.std(unbiased=False).item(),
            "p10_wsr": torch.quantile(x, 0.10).item(),
            "p50_wsr": torch.quantile(x, 0.50).item(),
            "p90_wsr": torch.quantile(x, 0.90).item(),
            "runtime_ms_per_sample": torch.tensor(aggregate_runtime[method], dtype=torch.float64).mean().item(),
            "reference_mean_wsr": ref_mean,
            "gap_to_reference": (ref_mean - x.mean().item()) / max(abs(ref_mean), 1e-12),
            "relative_wsr_percent": 100.0 * x.mean().item() / max(abs(ref_mean), 1e-12),
        }
        # Average selected batch-level metrics that are not reconstructable from WSR alone.
        for key in [
            "mean_user_rate",
            "mean_min_user_rate",
            "mean_jain_fairness",
            "mean_power",
            "power_utilization",
            "power_violation_ratio",
            "user_rate_outage_ratio",
            "sample_min_rate_outage_ratio",
            "win_rate_vs_reference",
            "mean_abs_wsr_gap",
            "p90_abs_wsr_gap",
        ]:
            vals = [m[key] for m in aggregate_metrics_acc[method] if key in m]
            if vals:
                merged[key] = float(torch.tensor(vals, dtype=torch.float64).mean().item())

        summary_rows.append(
            {
                "scenario": scenario.name,
                "seed": seed,
                "method": method,
                "reference": ref_name,
                **scenario.to_dict(),
                **merged,
            }
        )
    return raw_rows, summary_rows


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenarios = get_scenario_set(args.scenario_set)
    run_dir = make_run_dir(args.output_root, prefix=f"classical_{args.scenario_set}")
    save_json(
        run_dir / "config.json",
        {
            **vars(args),
            "device": device,
            "scenarios": [scenario.to_dict() for scenario in scenarios],
        },
    )
    per_batch_logger = CSVLogger(run_dir / "per_batch_metrics.csv")
    summary_logger = CSVLogger(run_dir / "summary_metrics.csv")

    for scenario in scenarios:
        for seed in args.seeds:
            raw_rows, summary_rows = evaluate_one_scenario(
                scenario,
                seed=seed,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
                wmmse_iters=args.wmmse_iters,
                reference_wmmse_iters=args.reference_wmmse_iters,
                rate_threshold=args.rate_threshold,
                device=device,
            )
            for row in raw_rows:
                per_batch_logger.write(row)
            for row in summary_rows:
                summary_logger.write(row)
                print(row)


if __name__ == "__main__":
    main()

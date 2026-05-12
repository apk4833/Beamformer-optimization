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
from data.channel import iid_rayleigh_channel
from utils.logger import CSVLogger, make_run_dir, save_json
from utils.metrics import beamformer_summary


def summarize(values: list[float]) -> dict[str, float]:
    x = torch.tensor(values, dtype=torch.float64)
    return {
        "mean_wsr": x.mean().item(),
        "std_wsr": x.std(unbiased=False).item(),
        "p10_wsr": torch.quantile(x, 0.10).item(),
        "p50_wsr": torch.quantile(x, 0.50).item(),
        "p90_wsr": torch.quantile(x, 0.90).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-users", type=int, default=4)
    parser.add_argument("--num-tx-antennas", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--p-max", type=float, default=1.0)
    parser.add_argument("--noise-var", type=float, default=1e-3)
    parser.add_argument("--wmmse-iters", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--output-root", type=str, default="outputs/evals")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = make_run_dir(args.output_root, prefix="classical")
    save_json(run_dir / "config.json", vars(args))
    per_seed_logger = CSVLogger(run_dir / "per_seed_metrics.csv")
    summary_logger = CSVLogger(run_dir / "summary_metrics.csv")

    all_values: dict[str, list[float]] = defaultdict(list)

    for seed in args.seeds:
        torch.manual_seed(seed)
        method_values: dict[str, list[float]] = defaultdict(list)
        method_runtime: dict[str, list[float]] = defaultdict(list)
        method_power: dict[str, list[float]] = defaultdict(list)
        method_violation: dict[str, list[float]] = defaultdict(list)

        for _ in range(args.num_batches):
            h = iid_rayleigh_channel(
                args.batch_size,
                args.num_users,
                args.num_tx_antennas,
                device=device,
            )
            methods = {
                "MRT": lambda: mrt(h, p_max=args.p_max),
                "ZF": lambda: zf(h, p_max=args.p_max),
                "RZF": lambda: rzf(h, p_max=args.p_max, reg=args.noise_var),
            }
            for iters in args.wmmse_iters:
                methods[f"WMMSE-{iters}"] = lambda iters=iters: wmmse(
                    h,
                    p_max=args.p_max,
                    noise_var=args.noise_var,
                    max_iter=iters,
                )

            for name, fn in methods.items():
                start = time.perf_counter()
                v = fn()
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                wsr = weighted_sum_rate(h, v, args.noise_var)
                method_values[name].extend(wsr.detach().cpu().tolist())
                method_runtime[name].append(elapsed_ms / args.batch_size)
                ps = beamformer_summary(v, args.p_max)
                method_power[name].append(ps["mean_power"])
                method_violation[name].append(ps["power_violation_ratio"])

        for name, values in method_values.items():
            row = {
                "seed": seed,
                "method": name,
                **summarize(values),
                "runtime_ms_per_sample": float(torch.tensor(method_runtime[name]).mean().item()),
                "mean_power": float(torch.tensor(method_power[name]).mean().item()),
                "power_violation_ratio": float(torch.tensor(method_violation[name]).mean().item()),
            }
            per_seed_logger.write(row)
            all_values[name].extend(values)
            print(row)

    ref_name = f"WMMSE-{max(args.wmmse_iters)}"
    ref_mean = summarize(all_values[ref_name])["mean_wsr"]
    for name, values in all_values.items():
        stats = summarize(values)
        row = {
            "method": name,
            **stats,
            "gap_to_reference": (ref_mean - stats["mean_wsr"]) / max(abs(ref_mean), 1e-12),
            "reference": ref_name,
        }
        summary_logger.write(row)


if __name__ == "__main__":
    main()

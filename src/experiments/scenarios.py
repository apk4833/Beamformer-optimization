from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable


@dataclass(frozen=True)
class BenchmarkScenario:
    """Single-cell MU-MISO benchmark scenario.

    The fields are intentionally close to the notation used in WMMSE and
    deep-unfolding papers: K users, Nt transmit antennas, P/sigma^2 in dB,
    and optional channel-distribution shift parameters.
    """

    name: str
    num_users: int = 4
    num_tx_antennas: int = 4
    snr_db: float = 10.0
    p_max: float = 1.0
    channel: str = "iid_rayleigh"
    pathloss_db_low: float = 0.0
    pathloss_db_high: float = 0.0
    notes: str = ""

    @property
    def noise_var(self) -> float:
        return self.p_max / (10.0 ** (self.snr_db / 10.0))

    @property
    def load_ratio(self) -> float:
        return self.num_users / self.num_tx_antennas

    def to_dict(self) -> dict:
        data = asdict(self)
        data["noise_var"] = self.noise_var
        data["load_ratio"] = self.load_ratio
        return data


def literature_core_scenarios() -> list[BenchmarkScenario]:
    """Scenarios aligned with common WMMSE/deep-unfolding figures."""
    return [
        BenchmarkScenario(
            name="fully_loaded_snr10",
            num_users=4,
            num_tx_antennas=4,
            snr_db=10.0,
            notes="Pellaco-style N=M=4, P/sigma^2=10 dB.",
        ),
        BenchmarkScenario(
            name="fully_loaded_snr20",
            num_users=4,
            num_tx_antennas=4,
            snr_db=20.0,
            notes="Pellaco-style N=M=4, P/sigma^2=20 dB.",
        ),
        BenchmarkScenario(
            name="lightly_loaded_snr10",
            num_users=4,
            num_tx_antennas=8,
            snr_db=10.0,
            notes="Lightly loaded M>N generalization scenario.",
        ),
        BenchmarkScenario(
            name="lightly_loaded_snr20",
            num_users=4,
            num_tx_antennas=8,
            snr_db=20.0,
            notes="Lightly loaded M>N generalization scenario.",
        ),
        BenchmarkScenario(
            name="pathloss_shift_snr10",
            num_users=4,
            num_tx_antennas=4,
            snr_db=10.0,
            channel="pathloss_rayleigh",
            pathloss_db_low=-5.0,
            pathloss_db_high=5.0,
            notes="Domain-shift test with per-user path loss U(-5,5) dB.",
        ),
        BenchmarkScenario(
            name="pathloss_shift_snr20",
            num_users=4,
            num_tx_antennas=4,
            snr_db=20.0,
            channel="pathloss_rayleigh",
            pathloss_db_low=-5.0,
            pathloss_db_high=5.0,
            notes="Domain-shift test with per-user path loss U(-5,5) dB.",
        ),
    ]


def sweep_scenarios(
    *,
    users: Iterable[int] = (2, 4, 6, 8),
    antennas: Iterable[int] = (4, 8, 16),
    snr_db: Iterable[float] = (0.0, 10.0, 20.0, 30.0),
    keep_feasible_zf: bool = True,
) -> list[BenchmarkScenario]:
    scenarios: list[BenchmarkScenario] = []
    for k, nt, snr in product(users, antennas, snr_db):
        if keep_feasible_zf and k > nt:
            continue
        load = "fully" if k == nt else "lightly" if k < nt else "over"
        scenarios.append(
            BenchmarkScenario(
                name=f"sweep_k{k}_nt{nt}_snr{snr:g}_{load}_loaded",
                num_users=k,
                num_tx_antennas=nt,
                snr_db=float(snr),
                notes="Systematic reviewer sweep over K, Nt, and SNR.",
            )
        )
    return scenarios


def get_scenario_set(name: str) -> list[BenchmarkScenario]:
    name = name.lower()
    if name == "literature_core":
        return literature_core_scenarios()
    if name == "sweep":
        return sweep_scenarios()
    if name == "all":
        return literature_core_scenarios() + sweep_scenarios()
    raise ValueError(f"Unknown scenario set: {name}")

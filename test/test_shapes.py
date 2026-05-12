from __future__ import annotations

from mu_miso_bf_lab.baselines.linear import mrt, rzf, zf
from mu_miso_bf_lab.core.complex_ops import total_power, weighted_sum_rate
from mu_miso_bf_lab.data.channel import iid_rayleigh_channel


def test_linear_baseline_shapes():
    h = iid_rayleigh_channel(3, 2, 4)
    for fn in [mrt, zf, rzf]:
        v = fn(h, p_max=1.0)
        assert v.shape == h.shape
        assert float(total_power(v).max()) <= 1.0001
        assert weighted_sum_rate(h, v, 1e-3).shape == (3,)

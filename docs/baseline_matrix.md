# Baseline Matrix

| Family | Method | Role | Implementation entry |
|---|---|---|---|
| Linear | MRT | sanity lower baseline | `baselines.linear.mrt` |
| Linear | ZF | high-SNR interference-canceling baseline | `baselines.linear.zf` |
| Linear | RZF | robust linear baseline | `baselines.linear.rzf` |
| Optimization | WMMSE | strongest classical baseline | `baselines.wmmse.wmmse` |
| Optimization | FP/QT | FP-compatible API, initially mapped to WMMSE-equivalent updates | `baselines.fp.fp_quadratic_transform` |
| Learned optimizer | Unfolded WMMSE | model-driven learning baseline | `models.unfolded_wmmse.UnfoldedWMMSE` |
| Direct learning | MLP beamformer | direct H-to-V neural baseline | `models.mlp_beamformer.DirectMLPBeamformer` |
| RL | DDPG-style environment | direct continuous BF RL stress test | `rl.env.StaticMISOBeamformingEnv` |

The first research milestone should compare MRT/ZF/RZF/WMMSE/unfolded-WMMSE/MLP under the same generated channels, power budget, noise variance, and random seeds.

# MU-MISO Beamforming Baseline Lab

A PyTorch-oriented research scaffold for single-cell MU-MISO downlink beamforming.

This project intentionally focuses on **continuous fully-digital beamforming** under weighted sum-rate (WSR) maximization. It excludes residual policy learning, queue-aware control, and hybrid discrete-continuous action design from the first baseline release.

## Core problem

Given a channel matrix `H` with shape `[B, K, Nt]` and a beamformer matrix `V` with shape `[B, K, Nt]`, solve

```math
\max_{V}\ \sum_{k=1}^{K}\alpha_k\log_2(1 + \mathrm{SINR}_k),\quad
\mathrm{s.t.}\ \sum_k \|v_k\|_2^2\le P_{max}.
```

The repository provides a unified interface for:

- Linear baselines: MRT, ZF, RZF
- Classical optimization: WMMSE
- Fractional-programming-style optimizer scaffold
- Deep unfolded WMMSE with trainable PGD step sizes
- Direct neural beamformer baseline
- Minimal continuous-action RL environment and DDPG skeleton

## Installation

```bash
conda create -n bf_lab python=3.10 -y
conda activate bf_lab
pip install -e .
```

For GPU PyTorch, install the wheel matching your CUDA version before `pip install -e .`.

## Quick smoke test

```bash
python scripts/smoke_test.py
```

## Suggested experiment order

1. Verify metrics and channel generation with MRT/ZF/RZF.
2. Run WMMSE and save labels.
3. Train the direct MLP beamformer supervised by WMMSE labels.
4. Train unfolded WMMSE unsupervised with negative WSR loss.
5. Only after these are stable, test direct RL beamformer baselines.

## Source-code integration notes

The scaffold is designed to absorb public baselines without rewriting the experiment manager:

- `third_party/WMMSE-deep-unfolding/` for Pellaco et al. notebook migration.
- `third_party/gcnwmmse/` for graph-WMMSE style PyTorch modules.
- `third_party/RIS-MISO-Deep-Reinforcement-Learning/` for DDPG-style actor-critic utilities.

Keep third-party code isolated and create adapters under `src/mu_miso_bf_lab/adapters/` if you later import full external repositories.

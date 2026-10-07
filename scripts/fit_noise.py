"""Fit the correlated-noise Cholesky factor from a recorded dataset.

    /home/sid/lerobot_env/bin/python scripts/fit_noise.py [DATASET_ROOT] [CHUNK] [BETA] [OUT]

Builds every length-CHUNK window of every episode, computes their covariance, and saves the
Cholesky factor. Windows do not cross episode boundaries: a chunk spanning the end of one demo and
the start of the next describes a teleport, not a trajectory, and would put nonsense correlations
into the factor.

Actions are normalised to zero mean and unit variance per dimension, then zero-padded from the
task's own width to the policy's max_action_dim. Both steps match what SmolVLA feeds its flow
matching (ACTION uses MEAN_STD normalisation), and both matter. Fitting on raw metres gives a
covariance with variance around 0.01 against an identity term of 1.0, so the shrinkage swamps the
signal and the result is barely distinguishable from independent noise.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "zero_control"))
from zero_control.v2.noise import CorrelatedNoiseSampler  # noqa: E402

MAX_ACTION_DIM = 32       # SmolVLA pads every action to this width


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path.home() / "zero_data/crossv2_base")
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    beta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    out = Path(sys.argv[4]) if len(sys.argv) > 4 else root / "noise_cholesky.pt"

    files = sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"no parquet under {root}/data")
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action"])
                    for f in files])

    windows = []
    for ep in sorted(df.episode_index.unique()):
        a = np.stack(df[df.episode_index == ep].sort_values("frame_index")["action"].to_numpy())
        for i in range(len(a) - chunk + 1):
            windows.append(a[i:i + chunk])
    x = torch.from_numpy(np.asarray(windows, dtype=np.float32))
    act_dim = x.shape[2]

    # Per-dimension mean/std over every frame, which is the statistic MEAN_STD normalisation uses.
    allacts = torch.from_numpy(np.stack(df["action"].to_numpy()).astype(np.float32))
    mu, sd = allacts.mean(0), allacts.std(0).clamp_min(1e-6)
    x = (x - mu) / sd
    if act_dim < MAX_ACTION_DIM:
        x = torch.cat([x, torch.zeros(x.shape[0], chunk, MAX_ACTION_DIM - act_dim)], dim=2)

    print(f"{len(files)} parquet file(s), {df.episode_index.nunique()} episodes")
    print(f"{x.shape[0]} chunks of {chunk} x {act_dim} (padded to {MAX_ACTION_DIM})")

    sampler = CorrelatedNoiseSampler(MAX_ACTION_DIM, chunk, beta)
    stats = sampler.fit(x)
    print("\nfit:")
    for k, v in stats.items():
        print(f"  {k:24} {v}")

    torch.save({"cholesky_L": sampler.cholesky_L, "action_dim": MAX_ACTION_DIM,
                "chunk_size": chunk, "beta": beta, "stats": stats,
                "action_mean": mu, "action_std": sd}, out)
    print(f"\nwrote {out}")

    # Is the result actually different from independent noise? Compare the step-to-step jump of a
    # correlated draw against an independent one; if they match, the factor bought nothing.
    corr = sampler.sample((512, chunk, MAX_ACTION_DIM), "cpu")[:, :, :act_dim]
    ind = torch.randn(512, chunk, act_dim)
    real = x[torch.randperm(x.shape[0])[:512], :, :act_dim]   # already normalised
    for name, v in (("real chunks", real), ("correlated draw", corr), ("independent draw", ind)):
        jump = (v[:, 1:] - v[:, :-1]).abs().mean()
        print(f"  mean |step-to-step change|, {name:18} {jump:.4f}")


if __name__ == "__main__":
    main()

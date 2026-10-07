"""Correlated noise for flow matching.

SmolVLA draws its flow-matching noise as independent samples per timestep and per dimension. That
is fine for the loss, and poor for the trajectory: independent noise has no temporal structure, so
individual samples come out jittery even when their average is right. Real robot chunks are
extremely correlated in time, and the handover in this task is a short coordinated burst, which is
exactly the shape that independent noise smears out.

So sample from N(0, S) where S is the empirical covariance of real action chunks, via its Cholesky
factor. Every sample is then a plausible trajectory rather than a plausible average.

S is shrunk toward the identity, S_reg = beta*S + (1-beta)*I, for two reasons. It keeps the draw
from collapsing onto the training set's exact modes, and it keeps the matrix positive definite:
SmolVLA zero-pads actions from the task's 20 dims to 32, so 12 of every 32 columns are identically
zero and the raw covariance is singular. The identity term is what makes the Cholesky exist.

This is the same construction as PRANA v2, whose fit is post-hoc: it can be applied to a checkpoint
that was trained with independent noise, because it changes the sampling distribution rather than
the weights.
"""

from __future__ import annotations

import torch
from torch import nn


class CorrelatedNoiseSampler(nn.Module):
    """Draws flow-matching noise with the temporal covariance of real action chunks."""

    def __init__(self, action_dim: int, chunk_size: int, beta: float = 0.5) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.beta = float(beta)
        self.total_dim = self.action_dim * self.chunk_size
        # A buffer, not a parameter: it is fitted from data, never gradient-updated, and it has to
        # travel with the checkpoint or inference silently reverts to independent noise.
        self.register_buffer("cholesky_L", torch.eye(self.total_dim, dtype=torch.float32))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))

    @torch.no_grad()
    def fit(self, chunks: torch.Tensor) -> dict:
        """Fit from real action chunks, shaped [N, chunk_size, action_dim].

        Returns a few numbers worth logging: without them there is no way to tell a well-conditioned
        fit from one that silently fell back to near-identity.
        """
        if chunks.ndim != 3 or chunks.shape[1:] != (self.chunk_size, self.action_dim):
            raise ValueError(
                f"expected [N, {self.chunk_size}, {self.action_dim}], got {tuple(chunks.shape)}")
        flat = chunks.reshape(chunks.shape[0], -1).double()
        centered = flat - flat.mean(dim=0, keepdim=True)
        sigma = (centered.T @ centered) / max(chunks.shape[0] - 1, 1)

        eye = torch.eye(self.total_dim, dtype=sigma.dtype)
        sigma_reg = self.beta * sigma + (1.0 - self.beta) * eye
        L = torch.linalg.cholesky(sigma_reg)

        self.cholesky_L.copy_(L.float())
        self.fitted.fill_(True)

        # How far from independent the result actually is. A near-identity correlation means the fit
        # bought nothing, which is worth knowing before spending a training run on it.
        d = torch.sqrt(torch.diag(sigma).clamp_min(1e-12))
        corr = sigma / torch.outer(d, d)
        off = corr - torch.diag(torch.diag(corr))
        return {
            "n_chunks": int(chunks.shape[0]),
            "mean_abs_offdiag_corr": float(off.abs().mean()),
            "max_abs_offdiag_corr": float(off.abs().max()),
            "cond_number": float(torch.linalg.cond(sigma_reg)),
        }

    def sample(self, shape: torch.Size | tuple, device) -> torch.Tensor:
        """Match SmolVLA's sample_noise signature: shape is [B, chunk_size, action_dim]."""
        b, t, d = shape
        if not bool(self.fitted) or (t, d) != (self.chunk_size, self.action_dim):
            # Unfitted, or asked for a shape this factor does not describe. Fall back rather than
            # return a wrongly-shaped or misleading draw.
            return torch.randn(b, t, d, device=device, dtype=torch.float32)
        z = torch.randn(b, self.total_dim, device=device, dtype=torch.float32)
        return (z @ self.cholesky_L.to(device).T).reshape(b, t, d)

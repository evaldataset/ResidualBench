"""Motif-level quality metrics for ResidualBench."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def compute_motif_metrics(
    residual: torch.Tensor | np.ndarray,
    codes: torch.Tensor | np.ndarray,
    reconstruction: torch.Tensor | np.ndarray,
    top_k_regimes: int = 10,
    top_k_examples: int = 10,
) -> dict[str, float]:
    """Compute motif-level quality metrics.

    Parameters
    ----------
    residual : array-like, shape (N, D)
        Original flattened residual windows.
    codes : array-like, shape (N, n_latent)
        Latent codes from any method.
    reconstruction : array-like, shape (N, D)
        Reconstructed residuals.
    top_k_regimes : int
        Number of top-used latent directions to evaluate.
    top_k_examples : int
        Number of top-activated examples per direction.

    Returns
    -------
    dict with keys: recon_mse, mean_cohesion, mean_variance_reduction
    """
    if isinstance(residual, np.ndarray):
        residual = torch.from_numpy(residual).float()
    if isinstance(codes, np.ndarray):
        codes = torch.from_numpy(codes).float()
    if isinstance(reconstruction, np.ndarray):
        reconstruction = torch.from_numpy(reconstruction).float()

    recon_mse = float(nn.functional.mse_loss(reconstruction, residual).item())
    global_var = float(residual.var(dim=0, unbiased=False).mean().item())
    global_norm = torch.nn.functional.normalize(residual.float(), dim=-1)

    usage = (codes.abs() > 1e-6).float().sum(dim=0)
    top_dims = torch.topk(usage, k=min(top_k_regimes, codes.shape[1])).indices.tolist()

    vr_sum, coh_sum = 0.0, 0.0
    for dim_idx in top_dims:
        scores = codes[:, dim_idx].abs()
        k = min(top_k_examples, scores.shape[0])
        _, topk_idx = torch.topk(scores, k=k)
        chosen = residual[topk_idx]
        chosen_norm = global_norm[topk_idx]

        local_var = chosen.var(dim=0, unbiased=False).mean().item()
        vr_sum += 1.0 - (local_var / max(global_var, 1e-8))

        n = chosen_norm.shape[0]
        if n > 1:
            sim = chosen_norm @ chosen_norm.T
            pair_mask = ~torch.eye(n, dtype=torch.bool)
            coh_sum += float(sim[pair_mask].mean().item())
        else:
            coh_sum += 1.0

    n_eval = max(len(top_dims), 1)
    return {
        "recon_mse": recon_mse,
        "mean_cohesion": coh_sum / n_eval,
        "mean_variance_reduction": vr_sum / n_eval,
    }

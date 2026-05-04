"""Analyze whether cross-forecaster residual similarity is trivial or meaningful.

Four analyses:
1. Centered residual cosine: remove mean residual, then compute cosine
2. Permutation test: shuffle time axis for null distribution
3. Forecast vs residual cosine: compare cosine(ŷ_A, ŷ_B) vs cosine(r_A, r_B)
4. Residual/target energy ratio: ||r|| / ||y||
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def pairwise_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean per-window cosine similarity between two (N, D) tensors."""
    na = F.normalize(a, dim=-1)
    nb = F.normalize(b, dim=-1)
    return float((na * nb).sum(-1).mean().item())


def analyze_dataset(ds_name: str, base_dir: Path) -> dict | None:
    """Run all 4 analyses for one dataset."""
    fc_data = {}
    for fc_dir in sorted((base_dir / ds_name).iterdir()):
        res_path = fc_dir / "residuals.pt"
        if not res_path.exists():
            continue
        data = torch.load(res_path, map_location="cpu", weights_only=False)
        if "test_residual" not in data:
            continue
        fc_data[fc_dir.name] = {
            "residual": data["test_residual"].flatten(1).float(),
            "target": data["test_target"].flatten(1).float(),
            "forecast": data["test_forecast"].flatten(1).float(),
        }

    fc_names = sorted(fc_data.keys())
    if len(fc_names) < 2:
        return None

    min_n = min(fc_data[fc]["residual"].shape[0] for fc in fc_names)
    for fc in fc_names:
        for k in fc_data[fc]:
            fc_data[fc][k] = fc_data[fc][k][:min_n]

    results: dict = {"n_windows": min_n, "n_forecasters": len(fc_names)}

    # === Analysis 1: Raw vs Centered cosine ===
    raw_pairs = []
    centered_pairs = []
    for i in range(len(fc_names)):
        for j in range(i + 1, len(fc_names)):
            r_a = fc_data[fc_names[i]]["residual"]
            r_b = fc_data[fc_names[j]]["residual"]

            raw_cos = pairwise_cosine(r_a, r_b)

            # Center: subtract global mean residual
            mean_r = torch.stack([
                fc_data[fc]["residual"] for fc in fc_names
            ]).mean(0)
            r_a_c = r_a - mean_r
            r_b_c = r_b - mean_r
            centered_cos = pairwise_cosine(r_a_c, r_b_c)

            raw_pairs.append(raw_cos)
            centered_pairs.append(centered_cos)

    results["raw_cosine_mean"] = float(np.mean(raw_pairs))
    results["centered_cosine_mean"] = float(np.mean(centered_pairs))
    results["centering_drop"] = float(
        (np.mean(raw_pairs) - np.mean(centered_pairs))
        / max(np.mean(raw_pairs), 1e-8)
        * 100
    )

    # === Analysis 2: Permutation test ===
    n_perms = 200
    perm_cosines = []
    r_a = fc_data[fc_names[0]]["residual"]
    r_b = fc_data[fc_names[1]]["residual"]
    observed = pairwise_cosine(r_a, r_b)

    for _ in range(n_perms):
        perm_idx = torch.randperm(min_n)
        r_b_perm = r_b[perm_idx]
        perm_cosines.append(pairwise_cosine(r_a, r_b_perm))

    perm_mean = float(np.mean(perm_cosines))
    perm_std = float(np.std(perm_cosines))
    z_score = (observed - perm_mean) / max(perm_std, 1e-8)
    p_value = float(np.mean([p >= observed for p in perm_cosines]))

    results["permutation_test"] = {
        "observed": observed,
        "null_mean": perm_mean,
        "null_std": perm_std,
        "z_score": z_score,
        "p_value": p_value,
    }

    # === Analysis 3: Forecast cosine vs Residual cosine ===
    forecast_pairs = []
    residual_pairs = []
    for i in range(len(fc_names)):
        for j in range(i + 1, len(fc_names)):
            f_a = fc_data[fc_names[i]]["forecast"]
            f_b = fc_data[fc_names[j]]["forecast"]
            r_a = fc_data[fc_names[i]]["residual"]
            r_b = fc_data[fc_names[j]]["residual"]

            forecast_pairs.append(pairwise_cosine(f_a, f_b))
            residual_pairs.append(pairwise_cosine(r_a, r_b))

    results["forecast_cosine_mean"] = float(np.mean(forecast_pairs))
    results["residual_cosine_mean"] = float(np.mean(residual_pairs))
    results["residual_minus_forecast"] = float(
        np.mean(residual_pairs) - np.mean(forecast_pairs)
    )

    # === Analysis 4: Residual/target energy ratio ===
    energy_ratios = []
    for fc in fc_names:
        r_norm = fc_data[fc]["residual"].norm(dim=-1).mean().item()
        y_norm = fc_data[fc]["target"].norm(dim=-1).mean().item()
        energy_ratios.append(r_norm / max(y_norm, 1e-8))

    results["residual_energy_ratio"] = {
        "mean": float(np.mean(energy_ratios)),
        "min": float(np.min(energy_ratios)),
        "max": float(np.max(energy_ratios)),
    }

    return results


def main() -> None:
    base_dir = Path("results/benchmark")
    if not base_dir.exists():
        base_dir = Path("results/optionb")

    all_results = {}
    datasets = []
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        for fc in d.iterdir():
            if fc.is_dir() and (fc / "residuals.pt").exists():
                datasets.append(d.name)
                break

    for ds_name in datasets:
        print(f"\n=== {ds_name} ===", flush=True)
        r = analyze_dataset(ds_name, base_dir)
        if r is None:
            continue
        all_results[ds_name] = r

        print(f"  Raw cosine:      {r['raw_cosine_mean']:.4f}")
        print(f"  Centered cosine: {r['centered_cosine_mean']:.4f} "
              f"(drop: {r['centering_drop']:.1f}%)")
        pt = r["permutation_test"]
        print(f"  Permutation:     obs={pt['observed']:.4f} "
              f"null={pt['null_mean']:.4f}±{pt['null_std']:.4f} "
              f"z={pt['z_score']:.1f} p={pt['p_value']:.4f}")
        print(f"  Forecast cosine: {r['forecast_cosine_mean']:.4f} "
              f"vs Residual: {r['residual_cosine_mean']:.4f} "
              f"(diff: {r['residual_minus_forecast']:+.4f})")
        er = r["residual_energy_ratio"]
        print(f"  Energy ratio:    ||r||/||y|| = {er['mean']:.3f} "
              f"(range: {er['min']:.3f}-{er['max']:.3f})")

    out_path = base_dir / "trivial_similarity_analysis.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

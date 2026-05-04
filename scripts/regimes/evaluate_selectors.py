"""Evaluate all model selection strategies including learned regime selector.

Compares: oracle, best-fixed, uniform, lag-1, learned selector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.regimes.regime_selector import evaluate_selectors


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--base-dir", default="results/benchmark")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        base_dir = Path("results/optionb")

    device = args.device if torch.cuda.is_available() else "cpu"
    results = {}

    for ds_dir in base_dir.iterdir():
        if not ds_dir.is_dir() or ds_dir.name.startswith("."):
            continue
        ds_name = ds_dir.name

        # Load all forecasters' data
        fc_data = {}
        for fc_dir in sorted(ds_dir.iterdir()):
            res_path = fc_dir / "residuals.pt"
            if not res_path.exists():
                continue
            data = torch.load(
                res_path, map_location="cpu", weights_only=False,
            )
            if "test_residual" not in data:
                continue
            fc_data[fc_dir.name] = data

        if len(fc_data) < 2:
            continue

        fc_names = sorted(fc_data.keys())

        # Compute per-window MSE for each forecaster
        min_n = min(
            fc_data[fc]["test_residual"].shape[0] for fc in fc_names
        )
        mse_matrix = np.stack([
            fc_data[fc]["test_residual"][:min_n]
            .pow(2).mean(dim=(1, 2)).numpy()
            for fc in fc_names
        ], axis=1)  # (N, n_fc)

        # Use forecasts as selector input (inference-time accessible).
        # Stack forecasts from all forecasters as feature vector.
        fc_forecasts = []
        for fc in fc_names:
            fc_forecasts.append(
                fc_data[fc]["test_forecast"][:min_n].flatten(1)
            )
        inputs = torch.cat(fc_forecasts, dim=1).numpy()

        print(f"\n{ds_name}: {len(fc_names)} forecasters, {min_n} windows")
        sel = evaluate_selectors(
            inputs, mse_matrix, fc_names, device=device,
        )
        results[ds_name] = sel

        print(f"  best_fixed={sel['best_fixed']:.4f} "
              f"({sel['best_forecaster']})")
        print(f"  lag1={sel['lag1']:.4f} "
              f"(vs bf: {sel['lag1_vs_bf_pct']:+.1f}%)")
        print(f"  learned={sel['learned']:.4f} "
              f"(vs bf: {sel['learned_vs_bf_pct']:+.1f}%)")
        print(f"  oracle={sel['oracle']:.4f} "
              f"(vs bf: {sel['oracle_vs_bf_pct']:+.1f}%)")

    # Save
    out_path = base_dir / "selector_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

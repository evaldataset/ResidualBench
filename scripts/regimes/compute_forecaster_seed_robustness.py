"""Forecaster-seed robustness analysis from multi-seed checkpoints.

For each (dataset, forecaster) pair where we have seed_{42, 11, 22}:
 - Test MSE mean/std and CV across seeds
 - Hungarian alignment of TopK prototypes across forecaster seeds (fixed decomp seed=42)

Inputs:
  results/benchmark/{ds}/{fc}/residuals.pt          (seed 42 from main benchmark)
  results/benchmark_multiseed/{ds}/{fc}/seed_{N}/residuals.pt   (seeds 11, 22)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.regimes.regime_autoencoder import SparseResidualRegimeAutoencoder
from src.utils.reproducibility import seed_everything

DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "Weather", "Electricity", "Traffic", "Exchange", "ILI",
]
FORECASTERS = ["dlinear", "patchtst", "itransformer", "nbeats", "timesnet"]
FORECASTER_SEEDS = [42, 11, 22]


def load_residuals(ds: str, fc: str, seed: int) -> dict | None:
    if seed == 42:
        p = Path(f"results/benchmark/{ds}/{fc}/residuals.pt")
    else:
        p = Path(f"results/benchmark_multiseed/{ds}/{fc}/seed_{seed}/residuals.pt")
    if not p.exists():
        return None
    return torch.load(p, map_location="cpu", weights_only=False)


def extract_prototypes(
    train_r: torch.Tensor,
    test_r: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    D = train_r.shape[1]
    seed_everything(42)
    model = SparseResidualRegimeAutoencoder(D, 64, 256, "topk", topk_k=16).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    ld = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_r), batch_size=128, shuffle=True,
    )
    for _ in range(20):
        model.train()
        for (b,) in ld:
            b = b.to(device)
            out = model(b)
            F.mse_loss(out["reconstruction"], b).backward()
            opt.step()
            opt.zero_grad()
    model.eval()
    with torch.no_grad():
        out = model(test_r.to(device))
    codes = out["regimes"].cpu()
    usage = (codes.abs() > 1e-6).float().sum(0)
    top10 = torch.topk(usage, k=min(10, codes.shape[1])).indices
    prototypes = []
    for idx in top10:
        scores = codes[:, idx].abs()
        k = min(10, scores.shape[0])
        top_s = torch.topk(scores, k=k).indices
        prototypes.append(test_r[top_s].mean(0))
    return torch.stack(prototypes)


def pair_hungarian(pa: torch.Tensor, pb: torch.Tensor) -> float:
    na = nn.functional.normalize(pa.float(), dim=-1)
    nb = nn.functional.normalize(pb.float(), dim=-1)
    sim = (na @ nb.T).numpy()
    row, col = linear_sum_assignment(-sim)
    return float(np.mean([sim[r, c] for r, c in zip(row, col, strict=True)]))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    results: dict[str, dict] = {}

    for ds in DATASETS:
        ds_res: dict = {}
        for fc in FORECASTERS:
            # MSE across seeds
            mses = []
            protos_for_seeds = {}
            for seed in FORECASTER_SEEDS:
                data = load_residuals(ds, fc, seed)
                if data is None:
                    continue
                # Test MSE = mean over (H, C) and samples of residual^2
                test_mse = float(data["test_residual"].pow(2).mean())
                mses.append(test_mse)

                # Compute prototypes only for datasets with reasonable test size
                test_r = data["test_residual"].flatten(1).float()
                train_r = data["train_residual"].flatten(1).float()
                if test_r.shape[0] >= 50 and train_r.shape[1] <= 10000:
                    protos_for_seeds[seed] = extract_prototypes(train_r, test_r, device)

            if not mses:
                continue
            fc_result: dict = {
                "n_seeds": len(mses),
                "test_mse_mean": float(np.mean(mses)),
                "test_mse_std": float(np.std(mses)),
                "test_mse_cv": float(np.std(mses) / np.mean(mses)) if np.mean(mses) > 0 else 0.0,
                "test_mse_per_seed": [float(v) for v in mses],
            }

            # Prototype alignment across forecaster seeds
            if len(protos_for_seeds) >= 2:
                seeds_p = sorted(protos_for_seeds.keys())
                pairs = []
                for i in range(len(seeds_p)):
                    for j in range(i + 1, len(seeds_p)):
                        h = pair_hungarian(
                            protos_for_seeds[seeds_p[i]],
                            protos_for_seeds[seeds_p[j]],
                        )
                        pairs.append({
                            "seed_a": seeds_p[i],
                            "seed_b": seeds_p[j],
                            "hungarian": h,
                        })
                fc_result["forecaster_seed_hungarian"] = {
                    "pairs": pairs,
                    "avg": float(np.mean([p["hungarian"] for p in pairs])),
                }

            ds_res[fc] = fc_result
            print(
                f"  {ds}/{fc}: n={len(mses)} "
                f"mse={fc_result['test_mse_mean']:.4f}"
                f"±{fc_result['test_mse_std']:.4f} "
                f"cv={fc_result['test_mse_cv']:.4f}",
                flush=True,
            )

        if ds_res:
            results[ds] = ds_res

    Path("results/benchmark/forecaster_seed_robustness.json").write_text(
        json.dumps(results, indent=2)
    )

    print("\n=== Summary: Test MSE CV by (dataset, forecaster) ===")
    for ds in DATASETS:
        if ds not in results:
            continue
        print(f"\n{ds}:")
        for fc in FORECASTERS:
            if fc in results[ds]:
                r = results[ds][fc]
                tag = ""
                if "forecaster_seed_hungarian" in r:
                    tag = f"  Hungarian={r['forecaster_seed_hungarian']['avg']:.3f}"
                print(f"  {fc:15s} n={r['n_seeds']} CV={r['test_mse_cv']*100:.2f}%{tag}")


if __name__ == "__main__":
    main()

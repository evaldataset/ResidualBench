"""Compute cross-seed (decomposition) Hungarian alignment across all 9 datasets.

For each dataset, trains TopK with 3 different decomposition seeds on the same
DLinear residuals, then computes pairwise Hungarian alignment between prototypes.
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

SEEDS = [42, 11, 22]
DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "Weather", "Electricity", "Traffic", "Exchange", "ILI",
]
FORECASTER = "dlinear"


def extract_prototypes(
    train_r: torch.Tensor,
    test_r: torch.Tensor,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Train TopK with given seed, return top-10 prototypes."""
    D = train_r.shape[1]
    seed_everything(seed)
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
        top_s = torch.topk(scores, k=min(10, scores.shape[0])).indices
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
        res_path = Path(f"results/benchmark/{ds}/{FORECASTER}/residuals.pt")
        if not res_path.exists():
            print(f"SKIP {ds}: no residuals", flush=True)
            continue
        print(f"\n=== {ds} ===", flush=True)
        data = torch.load(res_path, map_location="cpu", weights_only=False)
        train_r = data["train_residual"].flatten(1).float()
        test_r = data["test_residual"].flatten(1).float()

        # Skip tiny datasets
        if test_r.shape[0] < 10:
            print(f"  SKIP {ds}: only {test_r.shape[0]} test windows", flush=True)
            continue

        protos: dict[int, torch.Tensor] = {}
        for seed in SEEDS:
            protos[seed] = extract_prototypes(train_r, test_r, seed, device)

        pairwise = []
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                h = pair_hungarian(protos[SEEDS[i]], protos[SEEDS[j]])
                pairwise.append({
                    "seed_a": SEEDS[i],
                    "seed_b": SEEDS[j],
                    "hungarian": h,
                })
        avg = float(np.mean([p["hungarian"] for p in pairwise]))
        results[ds] = {"pairwise": pairwise, "avg_hungarian": avg}
        print(f"  avg Hungarian = {avg:.4f}", flush=True)

    out_path = Path("results/benchmark/cross_seed_stability_all.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")

    print("\n=== Summary ===")
    for ds in DATASETS:
        if ds in results:
            print(f"  {ds:12s} {results[ds]['avg_hungarian']:.3f}")


if __name__ == "__main__":
    main()

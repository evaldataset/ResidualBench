"""Evaluate core residual regime metrics from a hybrid checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.regimes.residual_pipeline import collect_residual_batches, load_residual_hybrid
from src.utils.metrics import mae, mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/hybrid/best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _cfg, loaders = load_residual_hybrid(args.checkpoint, device)
    data = collect_residual_batches(model, loaders[args.split], device)

    base = data["base_forecast"]
    hybrid = data["hybrid_forecast"]
    target = data["target"]
    concept = data["concept_forecast"]
    concepts = data["concepts"]
    residual_target = data["residual_target"]

    residual_energy = residual_target.pow(2).mean().item()
    residual_error = (residual_target - concept).pow(2).mean().item()
    explained = max(residual_energy - residual_error, 0.0)

    metrics = {
        "split": args.split,
        "base_mse": mse(base, target).item(),
        "hybrid_mse": mse(hybrid, target).item(),
        "base_mae": mae(base, target).item(),
        "hybrid_mae": mae(hybrid, target).item(),
        "mse_delta": mse(hybrid, target).item() - mse(base, target).item(),
        "mean_active_concepts": (concepts.abs() > 1e-6).float().sum(dim=-1).mean().item(),
        "residual_energy": residual_energy,
        "explained_residual_energy": explained,
        "explained_ratio": explained / max(residual_energy, 1e-8),
        "gate": float(model.gate.item()),
    }

    out_path = Path(args.checkpoint).parent / f"regime_metrics_{args.split}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()

"""Export residual regime tensors from a trained hybrid checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.regimes.residual_pipeline import collect_residual_batches, load_residual_hybrid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="results/hybrid/best.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, _cfg, loaders = load_residual_hybrid(args.checkpoint, device)
    payload = collect_residual_batches(model, loaders[args.split], device)

    output_path = (
        Path(args.output)
        if args.output is not None
        else Path(args.checkpoint).parent / f"residual_dataset_{args.split}.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved residual dataset to {output_path}", flush=True)


if __name__ == "__main__":
    main()

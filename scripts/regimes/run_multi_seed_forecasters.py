"""Train base forecasters with additional seeds (11, 22) on all datasets.

For each (dataset, forecaster) pair that already has a seed=42 checkpoint,
train two more seeds and collect residuals. Output goes to:
  results/benchmark_multiseed/{dataset}/{forecaster}/seed_{N}/

Usage (split across 2 GPUs):
    run_multi_seed_forecasters.py --gpu 0 --datasets ETTh1 ETTh2 ETTm1 ETTm2 Weather
    run_multi_seed_forecasters.py --gpu 1 --datasets Electricity Exchange Traffic ILI
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.regimes.run_full_benchmark import (
    DATASETS,
    FORECASTERS,
    HORIZON,
    SEQ_LEN,
)
from src.data.datasets import DATASET_META, build_dataloaders, load_csv_dataset
from src.models.factory import build_model
from src.utils.reproducibility import seed_everything


def train_and_collect(
    fc_name: str,
    ds_name: str,
    seed: int,
    device: torch.device,
    out_dir: Path,
) -> bool:
    save_dir = out_dir / ds_name / fc_name / f"seed_{seed}"
    ckpt_path = save_dir / "best.pt"
    res_path = save_dir / "residuals.pt"
    if res_path.exists():
        return True

    ds_info = DATASETS[ds_name]
    n_channels = int(DATASET_META[ds_name]["n_channels"])

    batch_size = 64
    n_epochs = 30
    if ds_name in ("Electricity", "Traffic"):
        batch_size = 32
        n_epochs = 15
    if ds_name == "ILI":
        batch_size = 16
        n_epochs = 50

    fc_cfg = dict(FORECASTERS[fc_name])
    if ds_name in ("Electricity", "Traffic") and fc_name == "itransformer":
        fc_cfg["d_model"] = 64
        fc_cfg["n_heads"] = 4
    if ds_name in ("Electricity", "Traffic") and fc_name == "nbeats":
        fc_cfg["hidden_size"] = 128

    cfg = {
        "model": fc_cfg,
        "data": {
            "data_path": ds_info["path"],
            "dataset_name": ds_name,
            "seq_len": SEQ_LEN,
            "horizon": HORIZON,
            "train_ratio": 0.7,
            "val_ratio": 0.1,
        },
    }

    seed_everything(seed)

    try:
        datasets = load_csv_dataset(
            path=ds_info["path"],
            dataset_name=ds_name,
            seq_len=SEQ_LEN,
            horizon=HORIZON,
        )
        loaders = build_dataloaders(
            datasets, batch_size=batch_size, num_workers=0,
        )

        model = build_model(cfg, n_channels).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs,
        )

        best_val = float("inf")
        patience = 0
        max_patience = 10

        for _epoch in range(n_epochs):
            model.train()
            for batch in loaders["train"]:
                x = batch["input"].to(device)
                y = batch["target"].to(device)
                out = model(x)
                forecast = out["forecast"] if isinstance(out, dict) else out
                loss = F.mse_loss(forecast, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in loaders["val"]:
                    x = batch["input"].to(device)
                    y = batch["target"].to(device)
                    out = model(x)
                    forecast = out["forecast"] if isinstance(out, dict) else out
                    val_loss += F.mse_loss(forecast, y).item()
                    n_val += 1

            avg_val = val_loss / max(n_val, 1)
            if avg_val < best_val:
                best_val = avg_val
                patience = 0
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                        "val_mse": best_val,
                        "seed": seed,
                    },
                    ckpt_path,
                )
            else:
                patience += 1
                if patience >= max_patience:
                    break

        # Collect residuals
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        result: dict[str, torch.Tensor] = {}
        for split in ["train", "test"]:
            targets, forecasts = [], []
            with torch.no_grad():
                for batch in loaders[split]:
                    x = batch["input"].to(device)
                    y = batch["target"].to(device)
                    out = model(x)
                    pred = out["forecast"] if isinstance(out, dict) else out
                    targets.append(y.cpu())
                    forecasts.append(pred.cpu())
            y_all = torch.cat(targets, 0)
            f_all = torch.cat(forecasts, 0)
            result[f"{split}_residual"] = y_all - f_all
            result[f"{split}_target"] = y_all
            result[f"{split}_forecast"] = f_all

        torch.save(result, res_path)
        print(
            f"    {fc_name}/{ds_name}/seed{seed}: "
            f"val={best_val:.4f}",
            flush=True,
        )
        return True
    except Exception as e:
        print(
            f"    ERROR {fc_name}/{ds_name}/seed{seed}: {e}",
            flush=True,
        )
        return False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--datasets", nargs="+", required=True,
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[11, 22],
    )
    parser.add_argument(
        "--forecasters", nargs="+", default=None,
    )
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    out_dir = Path("results/benchmark_multiseed")

    fc_list = args.forecasters or list(FORECASTERS.keys())

    for seed in args.seeds:
        for ds_name in args.datasets:
            print(f"\n=== seed={seed} dataset={ds_name} ===", flush=True)
            for fc_name in fc_list:
                train_and_collect(fc_name, ds_name, seed, device, out_dir)

    print(f"\nGPU {args.gpu} done!", flush=True)


if __name__ == "__main__":
    main()

"""Multi-horizon analysis: train forecasters + run benchmark at H=48,192,336.

Tests whether findings (cohesion rankings, cross-forecaster alignment,
temporal persistence) are stable across horizons.

Usage:
    python scripts/regimes/run_multi_horizon.py --device cuda
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.datasets import DATASET_META, build_dataloaders, load_csv_dataset
from src.models.factory import build_model
from src.regimes.regime_autoencoder import SparseResidualRegimeAutoencoder
from src.utils.reproducibility import seed_everything

HORIZONS = [48, 192, 336]
DATASETS = ["ETTh1", "Weather"]
FORECASTERS = {
    "dlinear": {"type": "dlinear", "individual": True},
    "patchtst": {
        "type": "patchtst", "d_model": 16, "n_heads": 4,
        "n_layers": 3, "d_ff": 128, "patch_len": 16, "stride": 8,
        "dropout": 0.3,
    },
    "itransformer": {
        "type": "itransformer", "d_model": 128, "n_heads": 4,
        "n_layers": 2, "dropout": 0.1,
    },
}
SEQ_LEN = 96


def train_and_collect(
    fc_name: str, ds_name: str, horizon: int,
    device: torch.device, out_dir: Path,
) -> bool:
    """Train forecaster at given horizon, collect residuals."""
    save_dir = out_dir / f"H{horizon}" / ds_name / fc_name
    if (save_dir / "residuals.pt").exists():
        return True

    ds_info = {"path": f"data/{ds_name}.csv"}
    n_channels = int(DATASET_META[ds_name]["n_channels"])
    fc_cfg = dict(FORECASTERS[fc_name])
    cfg = {
        "model": fc_cfg,
        "data": {
            "data_path": ds_info["path"],
            "dataset_name": ds_name,
            "seq_len": SEQ_LEN,
            "horizon": horizon,
        },
    }

    seed_everything(42)
    try:
        datasets = load_csv_dataset(
            path=ds_info["path"], dataset_name=ds_name,
            seq_len=SEQ_LEN, horizon=horizon,
        )
        loaders = build_dataloaders(datasets, batch_size=64, num_workers=0)
        model = build_model(cfg, n_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        best_val = float("inf")
        for _epoch in range(30):
            model.train()
            for batch in loaders["train"]:
                x, y = batch["input"].to(device), batch["target"].to(device)
                out = model(x)
                pred = out["forecast"] if isinstance(out, dict) else out
                loss = F.mse_loss(pred, y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            val_loss = 0.0
            n_val = 0
            with torch.no_grad():
                for batch in loaders["val"]:
                    x, y = batch["input"].to(device), batch["target"].to(device)
                    out = model(x)
                    pred = out["forecast"] if isinstance(out, dict) else out
                    val_loss += F.mse_loss(pred, y).item()
                    n_val += 1
            avg_val = val_loss / max(n_val, 1)
            if avg_val < best_val:
                best_val = avg_val
                save_dir.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": cfg, "val_mse": best_val,
                }, save_dir / "best.pt")

        # Collect residuals
        ckpt = torch.load(save_dir / "best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        result = {}
        for split in ["train", "test"]:
            targets, forecasts = [], []
            with torch.no_grad():
                for batch in loaders[split]:
                    x, y = batch["input"].to(device), batch["target"].to(device)
                    out = model(x)
                    pred = out["forecast"] if isinstance(out, dict) else out
                    targets.append(y.cpu())
                    forecasts.append(pred.cpu())
            y_all = torch.cat(targets, 0)
            f_all = torch.cat(forecasts, 0)
            result[f"{split}_residual"] = y_all - f_all
            result[f"{split}_target"] = y_all
            result[f"{split}_forecast"] = f_all
        torch.save(result, save_dir / "residuals.pt")
        print(f"    {fc_name}/{ds_name}/H{horizon}: val={best_val:.4f}", flush=True)
        return True

    except Exception as e:
        print(f"    ERROR {fc_name}/{ds_name}/H{horizon}: {e}", flush=True)
        return False


def compute_metrics_at_horizon(
    horizon: int, out_dir: Path, device: torch.device,
) -> dict:
    """Compute cohesion + alignment + persistence at one horizon."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    from scripts.regimes.compare_regime_baselines import compute_motif_metrics

    results = {}
    for ds_name in DATASETS:
        ds_results: dict = {}

        # Collect residuals from all forecasters
        fc_residuals = {}
        for fc_name in FORECASTERS:
            res_path = out_dir / f"H{horizon}" / ds_name / fc_name / "residuals.pt"
            if not res_path.exists():
                continue
            data = torch.load(res_path, map_location="cpu", weights_only=False)
            fc_residuals[fc_name] = {
                "train": data["train_residual"].flatten(1).float(),
                "test": data["test_residual"].flatten(1).float(),
            }

        if len(fc_residuals) < 2:
            continue

        # Cohesion: TopK on first forecaster
        fc0 = list(fc_residuals.keys())[0]
        train_r = fc_residuals[fc0]["train"]
        test_r = fc_residuals[fc0]["test"]
        D = train_r.shape[1]

        seed_everything(42)
        # TopK
        topk = SparseResidualRegimeAutoencoder(
            D, 64, 256, "topk", topk_k=16,
        ).to(device)
        opt = torch.optim.AdamW(topk.parameters(), lr=1e-3, weight_decay=1e-5)
        ld = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_r), batch_size=128, shuffle=True,
        )
        for _ in range(20):
            topk.train()
            for (b,) in ld:
                b = b.to(device)
                out = topk(b)
                F.mse_loss(out["reconstruction"], b).backward()
                opt.step()
                opt.zero_grad()
        topk.eval()
        with torch.no_grad():
            out = topk(test_r.to(device))
        codes = out["regimes"].cpu()
        recon = out["reconstruction"].cpu()
        m = compute_motif_metrics(test_r, codes, recon)
        ds_results["topk_cohesion"] = m["mean_cohesion"]

        # PCA cohesion
        pca = PCA(n_components=16, random_state=42).fit(train_r.numpy())
        pc = torch.from_numpy(pca.transform(test_r.numpy())).float()
        pr = torch.from_numpy(
            pc.numpy() @ pca.components_ + pca.mean_
        ).float()
        m_pca = compute_motif_metrics(test_r, pc, pr)
        ds_results["pca_cohesion"] = m_pca["mean_cohesion"]

        # k-means cohesion
        km = KMeans(n_clusters=16, random_state=42, n_init=10).fit(train_r.numpy())
        kl = km.predict(test_r.numpy())
        kc = np.zeros((len(test_r), 16), dtype=np.float32)
        kc[np.arange(len(test_r)), kl] = 1.0
        m_km = compute_motif_metrics(
            test_r, torch.from_numpy(kc),
            torch.from_numpy(km.cluster_centers_[kl]).float(),
        )
        ds_results["kmeans_cohesion"] = m_km["mean_cohesion"]

        # Cross-forecaster raw cosine
        fc_list = list(fc_residuals.keys())
        min_n = min(fc_residuals[fc]["test"].shape[0] for fc in fc_list)
        cosines = []
        for i in range(len(fc_list)):
            for j in range(i + 1, len(fc_list)):
                ra = F.normalize(fc_residuals[fc_list[i]]["test"][:min_n], dim=-1)
                rb = F.normalize(fc_residuals[fc_list[j]]["test"][:min_n], dim=-1)
                cosines.append(float((ra * rb).sum(-1).mean()))
        ds_results["raw_cosine"] = float(np.mean(cosines))

        # Temporal persistence
        dominant = codes.abs().argmax(dim=1).numpy()
        ds_results["persistence"] = float(np.mean(dominant[1:] == dominant[:-1]))

        results[ds_name] = ds_results
        print(
            f"  H{horizon}/{ds_name}: topk_coh={ds_results['topk_cohesion']:.3f} "
            f"raw_cos={ds_results['raw_cosine']:.3f} "
            f"persist={ds_results['persistence']:.3f}",
            flush=True,
        )

    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path("results/multi_horizon")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train at each horizon
    for horizon in HORIZONS:
        print(f"\n=== Horizon {horizon} ===", flush=True)
        for ds_name in DATASETS:
            for fc_name in FORECASTERS:
                train_and_collect(fc_name, ds_name, horizon, device, out_dir)

    # Compute metrics
    all_results = {}
    for horizon in HORIZONS:
        print(f"\n=== Metrics H{horizon} ===", flush=True)
        all_results[f"H{horizon}"] = compute_metrics_at_horizon(
            horizon, out_dir, device,
        )

    # Also include H96 from main benchmark
    print("\n=== Metrics H96 (from main benchmark) ===", flush=True)
    h96 = {}
    for ds_name in DATASETS:
        with open("results/benchmark/analysis.json") as f:
            a = json.load(f)
        raw = a.get("raw_residual_similarity", {}).get(ds_name, {})
        pers = a.get("temporal_persistence", {}).get(ds_name, {})
        # Get cohesion from all_results
        with open("results/benchmark/all_results.json") as f:
            data = json.load(f)
        topk_rows = [
            r for r in data
            if r["dataset"] == ds_name and r["method"] == "TopK"
        ]
        pca_rows = [
            r for r in data
            if r["dataset"] == ds_name and r["method"] == "PCA"
        ]
        km_rows = [
            r for r in data
            if r["dataset"] == ds_name and r["method"] == "k-means"
        ]
        h96[ds_name] = {
            "topk_cohesion": float(np.mean([r["cohesion"] for r in topk_rows])),
            "pca_cohesion": float(np.mean([r["cohesion"] for r in pca_rows])),
            "kmeans_cohesion": float(np.mean([r["cohesion"] for r in km_rows])),
            "raw_cosine": raw.get("avg_cosine", 0),
            "persistence": pers.get("same_consecutive", 0),
        }
    all_results["H96"] = h96

    # Save
    out_path = out_dir / "multi_horizon_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n=== Cross-Horizon Stability ===")
    for ds_name in DATASETS:
        print(f"\n{ds_name}:")
        print(f"  {'Horizon':>8s} {'TopK_coh':>9s} {'PCA_coh':>8s} "
              f"{'KM_coh':>7s} {'Raw_cos':>8s} {'Persist':>8s}")
        for h in [48, 96, 192, 336]:
            key = f"H{h}"
            if key in all_results and ds_name in all_results[key]:
                r = all_results[key][ds_name]
                print(
                    f"  {h:>8d} {r['topk_cohesion']:>9.3f} "
                    f"{r['pca_cohesion']:>8.3f} "
                    f"{r['kmeans_cohesion']:>7.3f} "
                    f"{r['raw_cosine']:>8.3f} "
                    f"{r['persistence']:>8.3f}"
                )

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

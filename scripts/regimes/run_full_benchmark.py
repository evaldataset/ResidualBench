"""Full ResidualBench: train 5 forecasters on 9 datasets, collect residuals,
run 7 methods × 3 seeds, compute cross-forecaster alignment + model selection.

Usage:
    python scripts/regimes/run_full_benchmark.py --phase all --device cuda
    python scripts/regimes/run_full_benchmark.py --phase train --device cuda
    python scripts/regimes/run_full_benchmark.py --phase benchmark --device cuda
    python scripts/regimes/run_full_benchmark.py --phase analysis --device cuda
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.datasets import DATASET_META, build_dataloaders, load_csv_dataset
from src.models.factory import build_model
from src.utils.reproducibility import seed_everything

# ============================================================
# Configuration
# ============================================================

DATASETS = {
    "ETTh1": {"path": "data/ETTh1.csv", "freq": "h"},
    "ETTh2": {"path": "data/ETTh2.csv", "freq": "h"},
    "ETTm1": {"path": "data/ETTm1.csv", "freq": "15min"},
    "ETTm2": {"path": "data/ETTm2.csv", "freq": "15min"},
    "Weather": {"path": "data/Weather.csv", "freq": "10min"},
    "Electricity": {"path": "data/Electricity.csv", "freq": "h"},
    "Traffic": {"path": "data/Traffic.csv", "freq": "h"},
    "Exchange": {"path": "data/Exchange.csv", "freq": "d"},
    "ILI": {"path": "data/ILI.csv", "freq": "w"},
}

FORECASTERS = {
    "dlinear": {
        "type": "dlinear",
        "individual": True,
    },
    "patchtst": {
        "type": "patchtst",
        "d_model": 16,
        "n_heads": 4,
        "n_layers": 3,
        "d_ff": 128,
        "patch_len": 16,
        "stride": 8,
        "dropout": 0.3,
    },
    "itransformer": {
        "type": "itransformer",
        "d_model": 128,
        "n_heads": 4,
        "n_layers": 2,
        "dropout": 0.1,
    },
    "nbeats": {
        "type": "nbeats",
        "n_stacks": 2,
        "n_blocks": 3,
        "hidden_size": 256,
        "theta_size": 32,
    },
    "timesnet": {
        "type": "timesnet",
        "d_model": 64,
        "d_ff": 64,
        "n_layers": 2,
        "top_k": 3,
        "dropout": 0.1,
    },
}

SEEDS = [42, 11, 22]
SEQ_LEN = 96
HORIZON = 96


# ============================================================
# Phase 1: Train forecasters
# ============================================================

def train_forecaster(
    fc_name: str,
    ds_name: str,
    device: torch.device,
    out_dir: Path,
    n_epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
) -> Path | None:
    """Train a single forecaster on a dataset, return checkpoint path."""
    save_dir = out_dir / ds_name / fc_name
    ckpt_path = save_dir / "best.pt"
    if ckpt_path.exists():
        print(f"    SKIP {fc_name}/{ds_name} — checkpoint exists", flush=True)
        return ckpt_path

    ds_info = DATASETS[ds_name]
    n_channels = int(DATASET_META[ds_name]["n_channels"])

    # Adjust for large datasets
    if ds_name in ("Electricity", "Traffic"):
        batch_size = 32
        n_epochs = 15
    if ds_name == "ILI":
        batch_size = 16
        n_epochs = 50

    # Adjust model for high-channel datasets
    fc_cfg = dict(FORECASTERS[fc_name])
    if ds_name in ("Electricity", "Traffic") and fc_name == "itransformer":
        fc_cfg["d_model"] = 64  # reduce for memory
        fc_cfg["n_heads"] = 4
    if ds_name in ("Electricity", "Traffic") and fc_name == "nbeats":
        fc_cfg["hidden_size"] = 128  # reduce for memory

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

    seed_everything(42)

    try:
        datasets = load_csv_dataset(
            path=ds_info["path"],
            dataset_name=ds_name,
            seq_len=SEQ_LEN,
            horizon=HORIZON,
        )
        loaders = build_dataloaders(datasets, batch_size=batch_size, num_workers=0)

        model = build_model(cfg, n_channels).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs,
        )

        best_val = float("inf")
        patience = 0
        max_patience = 10

        for epoch in range(n_epochs):
            # Train
            model.train()
            train_loss = 0.0
            n_batches = 0
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
                train_loss += loss.item()
                n_batches += 1
            scheduler.step()

            # Validate
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
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "val_mse": best_val,
                    "epoch": epoch,
                }, ckpt_path)
            else:
                patience += 1
                if patience >= max_patience:
                    break

        avg_train = train_loss / max(n_batches, 1)
        print(
            f"    {fc_name}/{ds_name}: "
            f"train={avg_train:.4f} val={best_val:.4f} "
            f"(epoch {epoch + 1})",
            flush=True,
        )
        return ckpt_path

    except Exception as e:
        print(f"    ERROR {fc_name}/{ds_name}: {e}", flush=True)
        return None


# ============================================================
# Phase 2: Collect residuals
# ============================================================

def collect_residuals(
    fc_name: str,
    ds_name: str,
    ckpt_path: Path,
    device: torch.device,
    out_dir: Path,
) -> bool:
    """Collect train/test residuals from a trained forecaster."""
    save_path = out_dir / ds_name / fc_name / "residuals.pt"
    if save_path.exists():
        return True

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = ckpt["config"]
        n_channels = int(DATASET_META[ds_name]["n_channels"])
        model = build_model(cfg, n_channels).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        datasets = load_csv_dataset(
            path=cfg["data"]["data_path"],
            dataset_name=ds_name,
            seq_len=SEQ_LEN,
            horizon=HORIZON,
        )
        loaders = build_dataloaders(datasets, batch_size=64, num_workers=0)

        result = {}
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

        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, save_path)
        n_train = result["train_residual"].shape[0]
        n_test = result["test_residual"].shape[0]
        print(
            f"    Residuals {fc_name}/{ds_name}: "
            f"train={n_train} test={n_test}",
            flush=True,
        )
        return True

    except Exception as e:
        print(f"    ERROR residuals {fc_name}/{ds_name}: {e}", flush=True)
        return False


# ============================================================
# Phase 3: Run benchmark methods
# ============================================================

def run_benchmark_methods(
    train_residual: torch.Tensor,
    test_residual: torch.Tensor,
    seed: int,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    """Run all 7 methods (train-fit, test-eval) and return metrics."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA, FastICA

    from scripts.regimes.compare_regime_baselines import (
        compute_motif_metrics,
        train_dense_ae,
    )
    from src.regimes.regime_autoencoder import SparseResidualRegimeAutoencoder

    seed_everything(seed)
    train_r = train_residual.flatten(1).float()
    test_r = test_residual.flatten(1).float()
    D = test_r.shape[1]
    train_np, test_np = train_r.numpy(), test_r.numpy()

    results = {}

    def eval_method(name: str, codes: torch.Tensor, recon: torch.Tensor) -> None:
        m = compute_motif_metrics(test_r, codes, recon)
        results[name] = {
            "recon_mse": m["recon_mse"],
            "cohesion": m["mean_cohesion"],
            "vr": m["mean_variance_reduction"],
        }

    # PCA
    pca = PCA(
        n_components=16, svd_solver="randomized", random_state=seed,
    ).fit(train_np)
    pc = torch.from_numpy(pca.transform(test_np)).float()
    pr = torch.from_numpy(pc.numpy() @ pca.components_ + pca.mean_).float()
    eval_method("PCA", pc, pr)

    # Dense AE 16
    dae = train_dense_ae(train_r, n_latent=16, device=device)
    with torch.no_grad():
        do = dae(test_r.to(device))
    eval_method("Dense_AE_16", do["codes"].cpu(), do["reconstruction"].cpu())

    # Dense AE 64
    dae64 = train_dense_ae(train_r, n_latent=64, device=device)
    with torch.no_grad():
        do64 = dae64(test_r.to(device))
    eval_method("Dense_AE_64", do64["codes"].cpu(), do64["reconstruction"].cpu())

    # k-means
    km = KMeans(n_clusters=16, random_state=seed, n_init=10).fit(train_np)
    kl = km.predict(test_np)
    kc = np.zeros((len(test_np), 16), dtype=np.float32)
    kc[np.arange(len(test_np)), kl] = 1.0
    eval_method(
        "k-means",
        torch.from_numpy(kc),
        torch.from_numpy(km.cluster_centers_[kl]).float(),
    )

    # TopK
    model = SparseResidualRegimeAutoencoder(
        D, 64, 256, "topk", topk_k=16,
    ).to(device)
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
    eval_method("TopK", out["regimes"].cpu(), out["reconstruction"].cpu())

    # ICA
    try:
        ica = FastICA(n_components=16, random_state=seed, max_iter=500)
        ica.fit(train_np)
        ic = torch.from_numpy(ica.transform(test_np)).float()
        ir = torch.from_numpy(ica.inverse_transform(ic.numpy())).float()
        eval_method("ICA", ic, ir)
    except Exception:
        pass

    # Spectral (skip for large datasets). Fit only on train data, then assign
    # test windows via train centroids to preserve the benchmark split.
    if D <= 5000 and 16 <= len(test_np) <= 5000 and len(train_np) >= 16:
        try:
            from sklearn.cluster import SpectralClustering
            from sklearn.neighbors import NearestCentroid

            n_sub = min(500, len(train_np))
            train_sub = train_np[:n_sub]
            n_neighbors = min(10, max(1, n_sub - 1))
            sc_labels = SpectralClustering(
                n_clusters=16, random_state=seed,
                affinity="nearest_neighbors", n_neighbors=n_neighbors,
                assign_labels="kmeans",
            ).fit_predict(train_sub)
            nc = NearestCentroid().fit(train_sub, sc_labels)
            full_labels = nc.predict(test_np)
            sc_codes = np.zeros((len(test_np), 16), dtype=np.float32)
            sc_codes[np.arange(len(test_np)), full_labels] = 1.0
            class_to_centroid = {
                int(cls): nc.centroids_[i] for i, cls in enumerate(nc.classes_)
            }
            sc_recon = np.stack(
                [class_to_centroid[int(label)] for label in full_labels],
            )
            eval_method(
                "Spectral",
                torch.from_numpy(sc_codes),
                torch.from_numpy(sc_recon).float(),
            )
        except Exception:
            pass

    return results


# ============================================================
# Phase 4: Cross-forecaster alignment + model selection
# ============================================================

def cross_forecaster_alignment(
    out_dir: Path, device: torch.device,
) -> dict:
    """Compare regime prototypes across forecasters for same dataset."""
    from scipy.optimize import linear_sum_assignment

    from src.regimes.regime_autoencoder import SparseResidualRegimeAutoencoder

    results = {}
    for ds_name in DATASETS:
        fc_prototypes = {}
        for fc_name in FORECASTERS:
            res_path = out_dir / ds_name / fc_name / "residuals.pt"
            if not res_path.exists():
                continue

            data = torch.load(res_path, map_location="cpu", weights_only=False)
            train_r = data["train_residual"].flatten(1).float()
            test_r = data["test_residual"].flatten(1).float()
            D = train_r.shape[1]

            seed_everything(42)
            model = SparseResidualRegimeAutoencoder(
                D, 64, 256, "topk", topk_k=16,
            ).to(device)
            opt = torch.optim.AdamW(
                model.parameters(), lr=1e-3, weight_decay=1e-5,
            )
            ld = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_r),
                batch_size=128, shuffle=True,
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
                top_s = torch.topk(
                    scores, k=min(10, scores.shape[0]),
                ).indices
                prototypes.append(test_r[top_s].mean(0))
            fc_prototypes[fc_name] = torch.stack(prototypes)

        # Pairwise Hungarian
        fc_list = list(fc_prototypes.keys())
        if len(fc_list) < 2:
            continue
        pairwise = []
        for i in range(len(fc_list)):
            for j in range(i + 1, len(fc_list)):
                pa = fc_prototypes[fc_list[i]]
                pb = fc_prototypes[fc_list[j]]
                na = nn.functional.normalize(pa.float(), dim=-1)
                nb = nn.functional.normalize(pb.float(), dim=-1)
                sim = (na @ nb.T).numpy()
                row_ind, col_ind = linear_sum_assignment(-sim)
                hung = float(np.mean(
                    [sim[r, c] for r, c in zip(row_ind, col_ind, strict=True)]
                ))
                pairwise.append({
                    "fc_a": fc_list[i], "fc_b": fc_list[j],
                    "hungarian": hung,
                })

        avg_hung = float(np.mean([p["hungarian"] for p in pairwise]))
        results[ds_name] = {
            "pairwise": pairwise, "avg_hungarian": avg_hung,
        }
        print(f"  {ds_name}: avg Hungarian = {avg_hung:.4f}", flush=True)

    return results


def raw_residual_similarity(out_dir: Path) -> dict:
    """Compute raw per-window cosine similarity between forecasters."""
    results = {}
    for ds_name in DATASETS:
        fc_residuals = {}
        for fc_name in FORECASTERS:
            res_path = out_dir / ds_name / fc_name / "residuals.pt"
            if not res_path.exists():
                continue
            data = torch.load(res_path, map_location="cpu", weights_only=False)
            fc_residuals[fc_name] = data["test_residual"].flatten(1).float()

        fc_list = list(fc_residuals.keys())
        if len(fc_list) < 2:
            continue

        # Ensure same number of windows
        min_n = min(r.shape[0] for r in fc_residuals.values())
        pairwise = []
        for i in range(len(fc_list)):
            for j in range(i + 1, len(fc_list)):
                ra = fc_residuals[fc_list[i]][:min_n]
                rb = fc_residuals[fc_list[j]][:min_n]
                na = nn.functional.normalize(ra, dim=-1)
                nb = nn.functional.normalize(rb, dim=-1)
                cos = (na * nb).sum(-1).mean().item()
                pairwise.append({
                    "a": fc_list[i], "b": fc_list[j],
                    "mean_cosine": cos,
                })

        avg_cos = float(np.mean([p["mean_cosine"] for p in pairwise]))
        results[ds_name] = {"pairwise": pairwise, "avg_cosine": avg_cos}
        print(f"  {ds_name}: avg raw cosine = {avg_cos:.4f}", flush=True)

    return results


def model_selection(out_dir: Path) -> dict:
    """Test lag-1 model selection across all datasets."""
    results = {}
    for ds_name in DATASETS:
        fc_mses = {}
        for fc_name in FORECASTERS:
            res_path = out_dir / ds_name / fc_name / "residuals.pt"
            if not res_path.exists():
                continue
            data = torch.load(res_path, map_location="cpu", weights_only=False)
            test_r = data["test_residual"]
            fc_mses[fc_name] = test_r.pow(2).mean(dim=(1, 2)).numpy()

        if len(fc_mses) < 2:
            continue

        fc_names = list(fc_mses.keys())
        min_n = min(len(v) for v in fc_mses.values())
        mse_mat = np.stack(
            [fc_mses[fc][:min_n] for fc in fc_names], axis=1,
        )
        N = mse_mat.shape[0]

        oracle_mse = float(mse_mat.min(axis=1).mean())
        avg_mses = mse_mat.mean(axis=0)
        best_idx = int(np.argmin(avg_mses))
        best_fixed_mse = float(mse_mat[:, best_idx].mean())

        # Lag-1
        lag1_choices = np.argmin(mse_mat[:-1], axis=1)
        lag1_mse = float(np.mean(
            [mse_mat[i + 1, lag1_choices[i]] for i in range(N - 1)]
        ))

        # Ensemble: average forecasts
        fc_forecasts = {}
        for fc_name in fc_names:
            data = torch.load(
                out_dir / ds_name / fc_name / "residuals.pt",
                map_location="cpu", weights_only=False,
            )
            fc_forecasts[fc_name] = data["test_forecast"][:min_n]
        targets = torch.load(
            out_dir / ds_name / fc_names[0] / "residuals.pt",
            map_location="cpu", weights_only=False,
        )["test_target"][:min_n]
        ens_pred = torch.stack(
            [fc_forecasts[fc] for fc in fc_names],
        ).mean(0)
        ens_mse = float(
            (targets - ens_pred).pow(2).mean(dim=(1, 2)).mean()
        )

        # Individual forecaster MSEs
        individual = {
            fc: float(mse_mat[:, i].mean()) for i, fc in enumerate(fc_names)
        }

        vs_best_pct = (best_fixed_mse - lag1_mse) / best_fixed_mse * 100
        vs_ens_pct = (ens_mse - lag1_mse) / ens_mse * 100

        results[ds_name] = {
            "individual": individual,
            "best_fixed": best_fixed_mse,
            "best_forecaster": fc_names[best_idx],
            "ensemble": ens_mse,
            "lag1": lag1_mse,
            "oracle": oracle_mse,
            "lag1_vs_best_fixed_pct": vs_best_pct,
            "lag1_vs_ensemble_pct": vs_ens_pct,
            "n_forecasters": len(fc_names),
            "n_windows": N,
        }
        print(
            f"  {ds_name}: best_fixed={best_fixed_mse:.4f} "
            f"ens={ens_mse:.4f} lag1={lag1_mse:.4f} "
            f"oracle={oracle_mse:.4f} (lag1 vs best: {vs_best_pct:+.1f}%)",
            flush=True,
        )

    return results


def temporal_persistence(out_dir: Path) -> dict:
    """Measure temporal persistence of failure modes."""
    from src.regimes.regime_autoencoder import SparseResidualRegimeAutoencoder

    results = {}
    device = torch.device("cpu")

    for ds_name in DATASETS:
        # Use first available forecaster's residuals
        for fc_name in FORECASTERS:
            res_path = out_dir / ds_name / fc_name / "residuals.pt"
            if res_path.exists():
                break
        else:
            continue

        data = torch.load(res_path, map_location="cpu", weights_only=False)
        train_r = data["train_residual"].flatten(1).float()
        test_r = data["test_residual"].flatten(1).float()
        D = train_r.shape[1]

        seed_everything(42)
        model = SparseResidualRegimeAutoencoder(
            D, 64, 256, "topk", topk_k=16,
        ).to(device)
        opt = torch.optim.AdamW(
            model.parameters(), lr=1e-3, weight_decay=1e-5,
        )
        ld = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_r),
            batch_size=128, shuffle=True,
        )
        for _ in range(20):
            model.train()
            for (b,) in ld:
                out = model(b)
                F.mse_loss(out["reconstruction"], b).backward()
                opt.step()
                opt.zero_grad()
        model.eval()
        with torch.no_grad():
            out = model(test_r)
        codes = out["regimes"]
        dominant = codes.abs().argmax(dim=1).numpy()

        same = float(np.mean(dominant[1:] == dominant[:-1]))
        # Autocorrelation
        d_float = dominant.astype(float)
        d_mean = d_float - d_float.mean()
        ac = float(np.corrcoef(d_mean[:-1], d_mean[1:])[0, 1])

        results[ds_name] = {
            "same_consecutive": same,
            "autocorrelation": ac,
        }
        print(
            f"  {ds_name}: same={same:.3f} ac={ac:.3f}", flush=True,
        )

    return results


# ============================================================
# Main
# ============================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Full ResidualBench")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--phase", default="all",
        choices=["train", "residuals", "benchmark", "analysis", "all"],
    )
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Subset of datasets (default: all)",
    )
    parser.add_argument(
        "--forecasters", nargs="*", default=None,
        help="Subset of forecasters (default: all)",
    )
    args = parser.parse_args()

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu",
    )
    out_dir = Path("results/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds_list = args.datasets or list(DATASETS.keys())
    fc_list = args.forecasters or list(FORECASTERS.keys())

    # Phase 1: Train
    if args.phase in ("train", "all"):
        print("\n" + "=" * 60, flush=True)
        print("  PHASE 1: Training forecasters", flush=True)
        print("=" * 60, flush=True)

        for ds_name in ds_list:
            for fc_name in fc_list:
                train_forecaster(fc_name, ds_name, device, out_dir)

    # Phase 2: Collect residuals
    if args.phase in ("residuals", "train", "all"):
        print("\n" + "=" * 60, flush=True)
        print("  PHASE 2: Collecting residuals", flush=True)
        print("=" * 60, flush=True)

        for ds_name in ds_list:
            for fc_name in fc_list:
                ckpt = out_dir / ds_name / fc_name / "best.pt"
                if not ckpt.exists():
                    continue
                collect_residuals(fc_name, ds_name, ckpt, device, out_dir)

    # Phase 3: Benchmark
    if args.phase in ("benchmark", "all"):
        print("\n" + "=" * 60, flush=True)
        print("  PHASE 3: Running benchmark methods", flush=True)
        print("=" * 60, flush=True)

        all_results = []
        for ds_name in ds_list:
            for fc_name in fc_list:
                res_path = out_dir / ds_name / fc_name / "residuals.pt"
                if not res_path.exists():
                    continue
                data = torch.load(
                    res_path, map_location="cpu", weights_only=False,
                )
                train_r = data["train_residual"]
                test_r = data["test_residual"]

                for seed in SEEDS:
                    print(
                        f"  {ds_name}/{fc_name}/seed={seed}...",
                        flush=True,
                    )
                    metrics = run_benchmark_methods(
                        train_r, test_r, seed, device,
                    )
                    for method, m in metrics.items():
                        all_results.append({
                            "dataset": ds_name,
                            "forecaster": fc_name,
                            "seed": seed,
                            "method": method,
                            **m,
                        })

        results_path = out_dir / "all_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(
            f"\nSaved {len(all_results)} entries to {results_path}",
            flush=True,
        )

    # Phase 4: Analysis
    if args.phase in ("analysis", "all"):
        print("\n" + "=" * 60, flush=True)
        print("  PHASE 4: Cross-forecaster analysis", flush=True)
        print("=" * 60, flush=True)

        print("\n--- Cross-Forecaster Alignment ---", flush=True)
        alignment = cross_forecaster_alignment(out_dir, device)

        print("\n--- Raw Residual Similarity ---", flush=True)
        raw_sim = raw_residual_similarity(out_dir)

        print("\n--- Model Selection ---", flush=True)
        selection = model_selection(out_dir)

        print("\n--- Temporal Persistence ---", flush=True)
        persistence = temporal_persistence(out_dir)

        analysis = {
            "cross_forecaster_alignment": alignment,
            "raw_residual_similarity": raw_sim,
            "model_selection": selection,
            "temporal_persistence": persistence,
        }
        analysis_path = out_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\nSaved analysis to {analysis_path}", flush=True)

        # Print summary
        print("\n" + "=" * 60, flush=True)
        print("  SUMMARY", flush=True)
        print("=" * 60, flush=True)
        for ds in selection:
            s = selection[ds]
            print(
                f"  {ds}: {s['n_forecasters']} fc, "
                f"lag1 vs best: {s['lag1_vs_best_fixed_pct']:+.1f}%",
            )


if __name__ == "__main__":
    main()

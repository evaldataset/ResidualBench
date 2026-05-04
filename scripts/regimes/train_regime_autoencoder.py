"""Train a sparse residual regime autoencoder on exported residual datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.regimes.regime_autoencoder import SparseResidualRegimeAutoencoder, regime_cohesion_loss
from src.utils.reproducibility import seed_everything


def load_residual_tensor(path: str, key: str = "residual_target") -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tensor = payload[key]
    return tensor.flatten(start_dim=1).float()


def make_loader(
    tensor: torch.Tensor, batch_size: int, shuffle: bool
) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(tensor)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate(
    model: SparseResidualRegimeAutoencoder,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_active = 0.0
    n_batches = 0
    with torch.no_grad():
        for (residual,) in loader:
            residual = residual.to(device)
            out = model(residual)
            recon_loss = nn.functional.mse_loss(out["reconstruction"], residual)
            active = (out["regimes"].abs() > 1e-6).float().sum(dim=-1).mean()
            total_loss += recon_loss.item()
            total_active += active.item()
            n_batches += 1
    return {
        "recon_mse": total_loss / max(n_batches, 1),
        "mean_active_regimes": total_active / max(n_batches, 1),
    }


def regime_diversity_loss(regimes: torch.Tensor) -> torch.Tensor:
    """Penalize correlated regime activations within a batch."""
    if regimes.shape[0] <= 1:
        return regimes.new_tensor(0.0)
    centered = regimes - regimes.mean(dim=0, keepdim=True)
    cov = centered.transpose(0, 1) @ centered / max(regimes.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return off_diag.pow(2).mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Config seed takes precedence over CLI default
    seed = cfg.get("training", {}).get("seed", args.seed)
    seed_everything(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_tensor = load_residual_tensor(cfg["data"]["train_dataset"])
    val_tensor = load_residual_tensor(cfg["data"]["val_dataset"])
    test_tensor = load_residual_tensor(cfg["data"]["test_dataset"])

    train_loader = make_loader(train_tensor, cfg["data"]["batch_size"], shuffle=True)
    val_loader = make_loader(val_tensor, cfg["data"]["batch_size"], shuffle=False)
    test_loader = make_loader(test_tensor, cfg["data"]["batch_size"], shuffle=False)

    model = SparseResidualRegimeAutoencoder(
        input_dim=train_tensor.shape[1],
        n_regimes=cfg["model"].get("n_regimes", 64),
        d_hidden=cfg["model"].get("d_hidden", 256),
        activation=cfg["model"].get("activation", "jumprelu"),
        init_threshold=cfg["model"].get("init_threshold", 0.1),
        topk_k=cfg["model"].get("topk_k", 10),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    best_val = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_loss_sum = 0.0
        train_active_sum = 0.0
        train_batches = 0
        for (residual,) in train_loader:
            residual = residual.to(device)
            out = model(residual)
            recon_loss = nn.functional.mse_loss(out["reconstruction"], residual)
            active = (out["regimes"].abs() > 1e-6).float().sum(dim=-1).mean()
            diversity = regime_diversity_loss(out["regimes"])
            cohesion = regime_cohesion_loss(
                out["regimes"], residual,
                top_k=cfg["loss"].get("cohesion_top_k", 3),
            )
            loss = (
                recon_loss
                + cfg["loss"].get("sparsity_weight", 1e-3) * active
                + cfg["loss"].get("diversity_weight", 0.0) * diversity
                + cfg["loss"].get("cohesion_weight", 0.0) * cohesion
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            optimizer.step()
            train_loss_sum += recon_loss.item()
            train_active_sum += active.item()
            train_batches += 1

        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d} | train_recon={train_loss_sum / max(train_batches, 1):.4f} "
            f"train_active={train_active_sum / max(train_batches, 1):.2f} "
            f"val_recon={val_metrics['recon_mse']:.4f} "
            f"val_active={val_metrics['mean_active_regimes']:.2f}",
            flush=True,
        )

        if val_metrics["recon_mse"] < best_val:
            best_val = val_metrics["recon_mse"]
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= cfg["training"]["patience"]:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    save_dir = Path(cfg["logging"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "config": cfg}, save_dir / "best.pt")

    test_metrics = evaluate(model, test_loader, device)
    (save_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print(
        f"\nTest: recon_mse={test_metrics['recon_mse']:.4f} "
        f"active={test_metrics['mean_active_regimes']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()

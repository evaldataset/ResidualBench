"""ResidualBench: main benchmark class with standardized evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from residualbench.metrics import compute_motif_metrics


class ResidualMethod(Protocol):
    """Protocol for methods to be evaluated by ResidualBench.

    Any method must implement fit/encode/reconstruct.

    Example
    -------
    >>> class MyMethod:
    ...     def fit(self, train_data: np.ndarray) -> None:
    ...         self.model = train_on(train_data)
    ...     def encode(self, data: np.ndarray) -> np.ndarray:
    ...         return self.model.transform(data)
    ...     def reconstruct(self, data: np.ndarray) -> np.ndarray:
    ...         return self.model.inverse_transform(data)
    """

    def fit(self, train_data: np.ndarray) -> None: ...
    def encode(self, data: np.ndarray) -> np.ndarray: ...
    def reconstruct(self, data: np.ndarray) -> np.ndarray: ...


class ResidualBench:
    """Standardized benchmark for forecast failure mode discovery.

    Supports two modes:
    1. Load pre-computed residuals from .pt files
    2. Pass numpy arrays directly

    Example
    -------
    >>> bench = ResidualBench()
    >>> bench.load_residuals("ETTh1/dlinear", "results/benchmark/ETTh1/dlinear/residuals.pt")
    >>> results = bench.evaluate(my_method, name="MyMethod", seeds=[42, 11, 22])
    >>> bench.summary()
    """

    def __init__(self) -> None:
        self.datasets: dict[str, dict[str, torch.Tensor]] = {}
        self.results: list[dict] = []

    def load_residuals(
        self, name: str, path: str | Path,
    ) -> None:
        """Load pre-computed residuals from a .pt file.

        Expected keys: 'train_residual', 'test_residual' with shape (N, H, C).
        """
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        if "train_residual" not in data or "test_residual" not in data:
            raise KeyError(
                f"Residual file must contain 'train_residual' and "
                f"'test_residual' keys. Found: {list(data.keys())}"
            )
        self.datasets[name] = {
            "train": data["train_residual"].flatten(start_dim=1).float(),
            "test": data["test_residual"].flatten(start_dim=1).float(),
        }

    def add_dataset(
        self,
        name: str,
        train_residuals: np.ndarray,
        test_residuals: np.ndarray,
    ) -> None:
        """Add dataset from numpy arrays.

        Parameters
        ----------
        name : str
            Dataset identifier.
        train_residuals : np.ndarray, shape (N_train, D)
            Flattened training residuals.
        test_residuals : np.ndarray, shape (N_test, D)
            Flattened test residuals.
        """
        self.datasets[name] = {
            "train": torch.from_numpy(train_residuals).float(),
            "test": torch.from_numpy(test_residuals).float(),
        }

    def load_benchmark_dir(self, base_dir: str | Path) -> None:
        """Load all residuals from a benchmark results directory.

        Expects structure: base_dir/{dataset}/{forecaster}/residuals.pt
        """
        base = Path(base_dir)
        for ds_dir in sorted(base.iterdir()):
            if not ds_dir.is_dir():
                continue
            for fc_dir in sorted(ds_dir.iterdir()):
                res_path = fc_dir / "residuals.pt"
                if res_path.exists():
                    name = f"{ds_dir.name}/{fc_dir.name}"
                    self.load_residuals(name, res_path)

    def evaluate(
        self,
        method: ResidualMethod,
        name: str,
        seeds: list[int] | None = None,
        top_k_regimes: int = 10,
        top_k_examples: int = 10,
    ) -> list[dict]:
        """Evaluate a method on all loaded datasets with proper train/test split.

        Parameters
        ----------
        method : ResidualMethod
            Method with fit/encode/reconstruct interface.
        name : str
            Method name for result tracking.
        seeds : list[int]
            Random seeds for repeated evaluation.
        top_k_regimes : int
            Number of top latent directions to evaluate.
        top_k_examples : int
            Number of top-activated examples per direction.

        Returns
        -------
        list[dict]
            Per-seed, per-dataset evaluation results.
        """
        if seeds is None:
            seeds = [42]

        run_results = []
        for ds_name, data in self.datasets.items():
            train_np = data["train"].numpy()
            test_r = data["test"]
            test_np = test_r.numpy()

            for seed in seeds:
                np.random.seed(seed)
                method.fit(train_np)
                codes = method.encode(test_np)
                recon = method.reconstruct(test_np)

                metrics = compute_motif_metrics(
                    test_r,
                    torch.from_numpy(codes).float(),
                    torch.from_numpy(recon).float(),
                    top_k_regimes=top_k_regimes,
                    top_k_examples=top_k_examples,
                )
                entry = {
                    "dataset": ds_name,
                    "method": name,
                    "seed": seed,
                    **metrics,
                }
                run_results.append(entry)
                self.results.append(entry)

        return run_results

    def summary(self) -> dict[str, dict[str, dict[str, float]]]:
        """Return and print summary of all results.

        Returns
        -------
        dict mapping method -> dataset -> {mean_cohesion, std_cohesion, ...}
        """
        if not self.results:
            print("No results yet. Run evaluate() first.")
            return {}

        datasets = sorted(set(r["dataset"] for r in self.results))
        methods = sorted(set(r["method"] for r in self.results))

        summary: dict[str, dict[str, dict[str, float]]] = {}

        # Print header
        print(f"{'Method':15s}", end="")
        for ds in datasets:
            ds_short = ds.split("/")[0] if "/" in ds else ds
            print(f"  {ds_short:>12s}", end="")
        print()
        print("-" * (15 + 14 * len(datasets)))

        for method in methods:
            summary[method] = {}
            print(f"{method:15s}", end="")
            for ds in datasets:
                rows = [
                    r for r in self.results
                    if r["method"] == method and r["dataset"] == ds
                ]
                if rows:
                    cohs = [r["mean_cohesion"] for r in rows]
                    mean_c = float(np.mean(cohs))
                    std_c = float(np.std(cohs))
                    summary[method][ds] = {
                        "mean_cohesion": mean_c,
                        "std_cohesion": std_c,
                        "mean_recon": float(
                            np.mean([r["recon_mse"] for r in rows])
                        ),
                        "std_recon": float(
                            np.std([r["recon_mse"] for r in rows])
                        ),
                        "mean_vr": float(
                            np.mean(
                                [r["mean_variance_reduction"] for r in rows]
                            )
                        ),
                    }
                    print(f"  {mean_c:.3f}±{std_c:.3f}", end="")
                else:
                    print(f"  {'---':>12s}", end="")
            print()

        return summary

    def to_dataframe(self) -> pd.DataFrame:  # noqa: F821
        """Convert results to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.results)

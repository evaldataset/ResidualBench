"""Generate all figures for the ResidualBench paper.

Figures:
1. Cohesion-Reconstruction Pareto plot (per dataset)
2. Cross-forecaster alignment heatmap
3. Leaky vs Proper evaluation comparison
4. Sparsity scaling curve
5. Temporal persistence visualization
6. Model selection bar chart
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 8,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})

FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST: dict[str, dict[str, object]] = {}

METHOD_COLORS = {
    "PCA": "#1f77b4",
    "Dense_AE_16": "#ff7f0e",
    "Dense_AE_64": "#d62728",
    "k-means": "#2ca02c",
    "TopK": "#9467bd",
    "ICA": "#8c564b",
    "Spectral": "#e377c2",
}

METHOD_MARKERS = {
    "PCA": "o",
    "Dense_AE_16": "s",
    "Dense_AE_64": "D",
    "k-means": "^",
    "TopK": "*",
    "ICA": "v",
    "Spectral": "p",
}


def load_benchmark_results() -> list[dict]:
    """Load benchmark results used by paper figures."""
    path = Path("results/benchmark/all_results.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required benchmark artifact: {path}. "
            "Run scripts/regimes/run_full_benchmark.py first."
        )
    MANIFEST["benchmark_results"] = {"path": str(path)}
    return json.loads(path.read_text())


def load_analysis() -> dict:
    """Load analysis results used by paper figures."""
    path = Path("results/benchmark/analysis.json")
    if not path.exists():
        raise FileNotFoundError(
            f"Missing required analysis artifact: {path}. "
            "Run scripts/regimes/run_full_benchmark.py --phase analysis first."
        )
    MANIFEST["analysis"] = {"path": str(path)}
    return json.loads(path.read_text())


def load_improvements() -> dict:
    """Load improvements data."""
    p = Path("results/improvements/all_improvements.json")
    if not p.exists():
        raise FileNotFoundError(
            f"Missing required improvements artifact: {p}. "
            "Run the improvement analysis scripts before generating paper figures."
        )
    MANIFEST["improvements"] = {"path": str(p)}
    return json.loads(p.read_text())


def require_keys(data: dict, keys: list[str], artifact: str) -> None:
    """Fail fast when a figure input artifact is incomplete."""
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{artifact} missing required keys: {missing}")


def record_figure(name: str, outputs: list[str], inputs: list[str]) -> None:
    """Record figure provenance for reproducibility."""
    MANIFEST[name] = {"outputs": outputs, "inputs": inputs}


# ============================================================
# Figure 1: Pareto plot (Cohesion vs Reconstruction)
# ============================================================

def fig_pareto(results: list[dict]) -> None:
    """Cohesion vs Recon MSE Pareto plot, one panel per dataset."""
    datasets = sorted(set(r["dataset"] for r in results))
    methods = sorted(set(r["method"] for r in results))
    n_ds = len(datasets)

    fig, axes = plt.subplots(1, min(n_ds, 4), figsize=(3.5 * min(n_ds, 4), 3))
    if n_ds == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets[:4], strict=False):
        for method in methods:
            rows = [
                r for r in results
                if r["dataset"] == ds and r["method"] == method
            ]
            if not rows:
                continue
            coh = np.mean([r["cohesion"] for r in rows])
            recon = np.mean([r["recon_mse"] for r in rows])
            coh_std = np.std([r["cohesion"] for r in rows])
            recon_std = np.std([r["recon_mse"] for r in rows])
            ax.errorbar(
                recon, coh,
                xerr=recon_std, yerr=coh_std,
                fmt=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, "gray"),
                markersize=8,
                capsize=2,
                label=method,
            )
        ax.set_xlabel("Reconstruction MSE")
        ax.set_ylabel("Motif Cohesion")
        ax.set_title(ds)
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best", ncol=1)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pareto.pdf")
    fig.savefig(FIG_DIR / "pareto.png")
    plt.close(fig)
    record_figure(
        "pareto",
        ["paper/figures/pareto.pdf", "paper/figures/pareto.png"],
        ["results/benchmark/all_results.json"],
    )
    print("  Saved pareto.pdf", flush=True)


# ============================================================
# Figure 2: Cross-forecaster alignment heatmap
# ============================================================

def fig_cross_forecaster(analysis: dict) -> None:
    """Heatmap of cross-forecaster Hungarian alignment."""
    align_data = analysis.get("cross_forecaster_alignment") or analysis.get(
        "rq1_cross_forecaster", {},
    )
    if not align_data:
        raise KeyError("analysis.json missing cross_forecaster_alignment data")

    datasets = sorted(align_data.keys())
    # Collect all forecaster pairs
    all_fcs = set()
    for ds in datasets:
        for p in align_data[ds].get("pairwise", []):
            all_fcs.add(p["fc_a"])
            all_fcs.add(p["fc_b"])
    fc_list = sorted(all_fcs)
    n_fc = len(fc_list)

    n_ds = len(datasets)
    ncols = min(3, n_ds)
    nrows = (n_ds + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 3 * nrows),
    )
    if n_ds == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, ds in zip(axes[:n_ds], datasets, strict=False):
        mat = np.eye(n_fc)
        for p in align_data[ds].get("pairwise", []):
            i = fc_list.index(p["fc_a"])
            j = fc_list.index(p["fc_b"])
            mat[i, j] = p["hungarian"]
            mat[j, i] = p["hungarian"]

        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(n_fc))
        ax.set_yticks(range(n_fc))
        ax.set_xticklabels(fc_list, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(fc_list, fontsize=7)
        ax.set_title(ds, fontsize=10)
        for i in range(n_fc):
            for j in range(n_fc):
                ax.text(
                    j, i, f"{mat[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                )

    # Hide unused axes
    for ax in axes[n_ds:]:
        ax.set_visible(False)
    fig.colorbar(im, ax=axes[:n_ds].tolist(), fraction=0.02, pad=0.04)
    fig.suptitle("Cross-Forecaster Regime Alignment (Hungarian)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cross_forecaster.pdf")
    fig.savefig(FIG_DIR / "cross_forecaster.png")
    plt.close(fig)
    record_figure(
        "cross_forecaster",
        ["paper/figures/cross_forecaster.pdf", "paper/figures/cross_forecaster.png"],
        ["results/benchmark/analysis.json"],
    )
    print("  Saved cross_forecaster.pdf", flush=True)


# ============================================================
# Figure 3: Leaky vs Proper evaluation
# ============================================================

def fig_leaky_vs_proper() -> None:
    """Bar chart comparing leaky vs proper evaluation scores."""
    leaky_path = Path("results/regimes/etth1_full_baseline_comparison.json")
    proper_path = Path("results/regimes/etth1_proper_baseline_comparison.json")
    missing = [str(p) for p in [leaky_path, proper_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing leaky/proper figure artifacts: {missing}")

    leaky_data = json.loads(leaky_path.read_text())
    proper_data = json.loads(proper_path.read_text())
    method_map = {
        "TopK": ("sparse_topk", "sparse_topk"),
        "PCA": ("pca", "pca"),
        "k-means": ("kmeans", "kmeans"),
        "K-SVD": ("ksvd", "ksvd"),
        "Dense_AE_16": ("dense_ae", "dense_ae"),
    }
    require_keys(leaky_data, [v[0] for v in method_map.values()], str(leaky_path))
    require_keys(proper_data, [v[1] for v in method_map.values()], str(proper_path))
    leaky = {
        display: float(leaky_data[lkey]["mean_cohesion"])
        for display, (lkey, _pkey) in method_map.items()
    }
    proper = {
        display: float(proper_data[pkey]["mean_cohesion"])
        for display, (_lkey, pkey) in method_map.items()
    }

    methods = list(leaky.keys())
    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x - width / 2, [leaky[m] for m in methods], width,
           label="Leaky (fit on test)", color="#d62728", alpha=0.8)
    ax.bar(x + width / 2, [proper[m] for m in methods], width,
           label="Proper (fit on train)", color="#1f77b4", alpha=0.8)

    # Annotate inflation %
    for i, m in enumerate(methods):
        inflation = (leaky[m] - proper[m]) / proper[m] * 100
        if inflation > 0:
            ax.annotate(
                f"+{inflation:.0f}%",
                (x[i] - width / 2, leaky[m]),
                textcoords="offset points", xytext=(0, 5),
                ha="center", fontsize=7, color="#d62728",
            )

    ax.set_ylabel("Motif Cohesion")
    ax.set_title("ETTh1: Leaky vs Proper Evaluation")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "leaky_vs_proper.pdf")
    fig.savefig(FIG_DIR / "leaky_vs_proper.png")
    plt.close(fig)
    record_figure(
        "leaky_vs_proper",
        ["paper/figures/leaky_vs_proper.pdf", "paper/figures/leaky_vs_proper.png"],
        [str(leaky_path), str(proper_path)],
    )
    print("  Saved leaky_vs_proper.pdf", flush=True)


# ============================================================
# Figure 4: Sparsity scaling curve
# ============================================================

def fig_sparsity_scaling(improvements: dict) -> None:
    """Plot cohesion and recon vs sparsity k."""
    scaling = improvements.get("7_sparsity_scaling", [])
    if not scaling:
        raise KeyError("all_improvements.json missing 7_sparsity_scaling data")

    ks = [s["k"] for s in scaling]
    cohs = [s["cohesion"] for s in scaling]
    recons = [s["recon"] for s in scaling]

    fig, ax1 = plt.subplots(figsize=(4, 3))
    color1 = "#1f77b4"
    color2 = "#d62728"

    ax1.plot(ks, cohs, "o-", color=color1, label="Cohesion", markersize=6)
    ax1.set_xlabel("Sparsity k (active units)")
    ax1.set_ylabel("Motif Cohesion", color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(ks, recons, "s--", color=color2, label="Recon MSE", markersize=6)
    ax2.set_ylabel("Reconstruction MSE", color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    ax1.set_title("Sparsity-Reconstruction Tradeoff (ETTh1)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sparsity_scaling.pdf")
    fig.savefig(FIG_DIR / "sparsity_scaling.png")
    plt.close(fig)
    record_figure(
        "sparsity_scaling",
        ["paper/figures/sparsity_scaling.pdf", "paper/figures/sparsity_scaling.png"],
        ["results/improvements/all_improvements.json"],
    )
    print("  Saved sparsity_scaling.pdf", flush=True)


# ============================================================
# Figure 5: Model selection comparison
# ============================================================

def fig_model_selection(_analysis: dict) -> None:
    """Bar chart of model selection strategies.

    Uses selector_comparison.json (temporal 70/30 split) so the figure matches
    Table 5 in the paper. Only datasets with >=50 test windows are shown.
    """
    sel_path = Path("results/benchmark/selector_comparison.json")
    if not sel_path.exists():
        raise FileNotFoundError(
            f"Missing {sel_path}. Run scripts/regimes/evaluate_selectors.py first."
        )
    sel = json.loads(sel_path.read_text())

    # Filter to datasets with enough test windows (exclude ILI which has lag1=nan)
    datasets = sorted(
        ds for ds, v in sel.items()
        if isinstance(v.get("lag1"), int | float) and v.get("lag1") == v.get("lag1")
    )

    strategies = ["best_fixed", "lag1", "oracle"]
    labels = ["Best Fixed", "Lag-1", "Oracle"]
    colors = ["#ff7f0e", "#9467bd", "#1f77b4"]

    x = np.arange(len(datasets))
    width = 0.27

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for i, (strat, label, color) in enumerate(
        zip(strategies, labels, colors, strict=True)
    ):
        vals = [sel[ds][strat] for ds in datasets]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_ylabel("Test MSE")
    ax.set_title("Model Selection (temporal 70/30 split, 5 forecasters)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_selection.pdf")
    fig.savefig(FIG_DIR / "model_selection.png")
    plt.close(fig)
    record_figure(
        "model_selection",
        ["paper/figures/model_selection.pdf", "paper/figures/model_selection.png"],
        ["results/benchmark/selector_comparison.json"],
    )
    print("  Saved model_selection.pdf", flush=True)


# ============================================================
# Figure 6: Temporal persistence
# ============================================================

def fig_temporal_persistence(analysis: dict) -> None:
    """Bar chart of temporal persistence metrics."""
    pers = analysis.get("temporal_persistence", {})
    if not pers:
        raise KeyError("analysis.json missing temporal_persistence data")

    datasets = sorted(pers.keys())
    same = [pers[ds]["same_consecutive"] for ds in datasets]
    ac = [pers[ds]["autocorrelation"] for ds in datasets]

    x = np.arange(len(datasets))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(x - width / 2, same, width, label="Same-regime ratio", color="#1f77b4")
    ax.bar(x + width / 2, ac, width, label="Autocorrelation", color="#ff7f0e")
    ax.set_ylabel("Score")
    ax.set_title("Temporal Persistence of Failure Modes")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "temporal_persistence.pdf")
    fig.savefig(FIG_DIR / "temporal_persistence.png")
    plt.close(fig)
    record_figure(
        "temporal_persistence",
        ["paper/figures/temporal_persistence.pdf", "paper/figures/temporal_persistence.png"],
        ["results/benchmark/analysis.json"],
    )
    print("  Saved temporal_persistence.pdf", flush=True)


def main() -> None:
    print("Generating figures...", flush=True)

    results = load_benchmark_results()
    analysis = load_analysis()
    improvements = load_improvements()

    if not results:
        raise ValueError("Benchmark results artifact is empty")
    fig_pareto(results)

    fig_cross_forecaster(analysis)
    fig_leaky_vs_proper()
    fig_sparsity_scaling(improvements)
    fig_model_selection(analysis)
    fig_temporal_persistence(analysis)

    manifest_path = FIG_DIR / "manifest.json"
    MANIFEST["generated_at"] = datetime.now(timezone.utc).isoformat()
    MANIFEST["command"] = "python scripts/regimes/generate_figures.py"
    manifest_path.write_text(json.dumps(MANIFEST, indent=2) + "\n")
    print(f"  Saved {manifest_path}", flush=True)

    print(f"\nAll figures saved to {FIG_DIR}/", flush=True)


if __name__ == "__main__":
    main()

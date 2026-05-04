"""Plot cross-forecaster vs cross-seed vs cross-forecaster-seed Hungarian alignment.

Compelling visualization of the three variation sources, showing they are
comparable in magnitude — a key finding that supports the "partly data-intrinsic"
interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path("paper/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "serif",
})


def main() -> None:
    analysis = json.loads(Path("results/benchmark/analysis.json").read_text())
    cross_seed = json.loads(
        Path("results/benchmark/cross_seed_stability_all.json").read_text(),
    )
    fc_seed = json.loads(
        Path("results/benchmark/forecaster_seed_robustness.json").read_text(),
    )

    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Traffic", "Exchange"]

    cross_fc = []
    cross_seed_vals = []
    cross_fc_seed = []

    for ds in datasets:
        # Cross-forecaster
        cross_fc_v = analysis.get(
            "cross_forecaster_alignment", {},
        ).get(ds, {}).get("avg_hungarian")
        cross_fc.append(cross_fc_v)

        # Cross-decomposition-seed
        cross_seed_v = cross_seed.get(ds, {}).get("avg_hungarian")
        cross_seed_vals.append(cross_seed_v)

        # Cross-forecaster-seed (average across forecasters that have H)
        fcs = fc_seed.get(ds, {})
        hs = []
        for v in fcs.values():
            h = v.get("forecaster_seed_hungarian", {}).get("avg")
            if h is not None:
                hs.append(h)
        cross_fc_seed.append(float(np.mean(hs)) if hs else None)

    # Filter to datasets where all three are available
    plot_ds = []
    plot_fc = []
    plot_seed = []
    plot_fc_seed = []
    for i, ds in enumerate(datasets):
        if all(v is not None for v in [cross_fc[i], cross_seed_vals[i], cross_fc_seed[i]]):
            plot_ds.append(ds)
            plot_fc.append(cross_fc[i])
            plot_seed.append(cross_seed_vals[i])
            plot_fc_seed.append(cross_fc_seed[i])

    x = np.arange(len(plot_ds))
    width = 0.27

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    ax.bar(
        x - width, plot_fc, width,
        label="Cross-forecaster (fixed seed)",
        color="#1f77b4", alpha=0.85,
    )
    ax.bar(
        x, plot_seed, width,
        label="Cross-decomposition-seed (fixed forecaster)",
        color="#ff7f0e", alpha=0.85,
    )
    ax.bar(
        x + width, plot_fc_seed, width,
        label="Cross-forecaster-seed (fixed decomposition)",
        color="#2ca02c", alpha=0.85,
    )

    # Null baseline
    ax.axhline(0.05, linestyle="--", color="gray", alpha=0.6, label="Null baseline (0.05)")

    ax.set_ylabel("Hungarian alignment")
    ax.set_title(
        "Three sources of prototype variation are comparable in magnitude",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(plot_ds, rotation=20, ha="right")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 0.85)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "alignment_sources.pdf")
    fig.savefig(FIG_DIR / "alignment_sources.png")
    plt.close(fig)
    print(f"Saved alignment_sources.pdf/png (datasets: {plot_ds})")


if __name__ == "__main__":
    main()

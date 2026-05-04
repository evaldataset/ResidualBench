"""Compute bootstrap CI for cohesion and full Wilcoxon+effect-size table.

Outputs:
  results/benchmark/bootstrap_ci.json        — percentile CIs per (ds, method)
  results/benchmark/wilcoxon_full.json       — full pairwise Wilcoxon + effect size
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


def bootstrap_ci(values: list[float], n_boot: int = 10000, alpha: float = 0.05) -> dict:
    arr = np.asarray(values, dtype=float)
    rng = np.random.RandomState(42)
    n = len(arr)
    if n < 2:
        return {"mean": float(arr.mean()), "lo": None, "hi": None, "n": n}
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = arr[idx].mean()
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return {"mean": float(arr.mean()), "std": float(arr.std()), "lo": lo, "hi": hi, "n": n}


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Rank-biserial correlation for paired samples (effect size for Wilcoxon)."""
    diffs = a - b
    diffs = diffs[diffs != 0]
    if len(diffs) == 0:
        return 0.0
    ranks = rankdata(np.abs(diffs))
    r_plus = ranks[diffs > 0].sum()
    r_minus = ranks[diffs < 0].sum()
    total = r_plus + r_minus
    if total == 0:
        return 0.0
    return float((r_plus - r_minus) / total)


def main() -> None:
    with open("results/benchmark/all_results.json") as f:
        data = json.load(f)

    datasets = sorted(set(r["dataset"] for r in data))
    methods = ["PCA", "Dense_AE_16", "Dense_AE_64", "k-means", "TopK", "ICA", "Spectral"]

    # ========== Bootstrap CIs ==========
    boot: dict[str, dict[str, dict]] = {}
    for ds in datasets:
        boot[ds] = {}
        for m in methods:
            rows = [r for r in data if r["dataset"] == ds and r["method"] == m]
            if not rows:
                continue
            cohs = [r["cohesion"] for r in rows]
            boot[ds][m] = bootstrap_ci(cohs)

    # ========== Wilcoxon with effect size ==========
    wil: dict[str, dict[str, dict]] = {}
    for ds in datasets:
        wil[ds] = {}
        topk = sorted([r["cohesion"] for r in data if r["dataset"] == ds and r["method"] == "TopK"])
        if not topk:
            continue
        for m in methods:
            if m == "TopK":
                continue
            other = sorted([r["cohesion"] for r in data if r["dataset"] == ds and r["method"] == m])
            if not other:
                continue
            n = min(len(topk), len(other))
            a = np.array(topk[:n])
            b = np.array(other[:n])
            if np.all(a == b):
                wil[ds][m] = {"p": 1.0, "effect": 0.0, "n": n, "diff": 0.0}
                continue
            try:
                _, p = wilcoxon(a, b)
            except ValueError:
                p = float("nan")
            wil[ds][m] = {
                "p": float(p),
                "effect": rank_biserial(a, b),
                "n": int(n),
                "mean_topk": float(a.mean()),
                "mean_other": float(b.mean()),
                "diff": float(a.mean() - b.mean()),
            }

    # Holm-Bonferroni correction within each comparison (across datasets)
    # Use dataset as multiple comparison dimension, test = TopK vs each method separately
    for m in methods:
        if m == "TopK":
            continue
        pvals = []
        for ds in datasets:
            if ds in wil and m in wil[ds]:
                pvals.append((ds, wil[ds][m]["p"]))
        if not pvals:
            continue
        pvals_sorted = sorted(pvals, key=lambda x: x[1])
        k = len(pvals_sorted)
        for i, (ds, p) in enumerate(pvals_sorted):
            threshold = 0.05 / (k - i)
            wil[ds][m]["holm_threshold"] = float(threshold)
            wil[ds][m]["holm_sig"] = bool(p < threshold)

    Path("results/benchmark/bootstrap_ci.json").write_text(json.dumps(boot, indent=2))
    Path("results/benchmark/wilcoxon_full.json").write_text(json.dumps(wil, indent=2))

    # Print summary
    print("=== Bootstrap 95% CI for Cohesion ===")
    for ds in ["ETTh1", "ETTh2", "ETTm1", "ETTm2"]:
        print(f"\n{ds}:")
        for m in methods:
            if m in boot[ds]:
                c = boot[ds][m]
                if c["lo"] is not None:
                    print(f"  {m:15s} {c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]")

    print("\n=== Wilcoxon (TopK vs each, with Holm-Bonferroni) ===")
    for ds in datasets:
        print(f"\n{ds}:")
        for m in methods:
            if m == "TopK":
                continue
            if ds in wil and m in wil[ds]:
                w = wil[ds][m]
                sig = "*" if w.get("holm_sig") else " "
                print(
                    f"  {m:15s} diff={w['diff']:+.3f} p={w['p']:.4g} "
                    f"r_rb={w['effect']:+.3f} holm<{w.get('holm_threshold', 0):.4g} {sig}",
                )

    print("\nSaved bootstrap_ci.json and wilcoxon_full.json")


if __name__ == "__main__":
    main()

# Reproducing ResidualBench Results

This guide reproduces the four headline findings reported in the paper
"ResidualBench: A Benchmark and Evaluation Protocol for Cross-Model Forecast
Failure Mode Discovery" (NeurIPS 2026 Evaluations & Datasets Track).

## Hardware

- 1 GPU with >=12 GB VRAM (we used NVIDIA A100/3090; CPU works but is slow on
  N-BEATS / TimesNet / TopK-SAE).
- ~50 GB free disk for residual artifacts at `H=96`.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

This installs the `residualbench` package (`pyproject.toml`) and dev
dependencies (`pytest`, `ruff`, `mypy`).

Sanity check:

```bash
pytest tests/test_residualbench.py -q
```

## Step 1 — Datasets (~5 min)

```bash
python scripts/download_data.py --include-large
```

Downloads ETTh1/h2, ETTm1/m2, Weather, Electricity, Traffic, Exchange, ILI from
their public hosts to `data/`. Total ~3 GB.

## Step 2 — Train forecasters and collect residuals (~6-10 GPU-hours)

Trains all 5 forecasters (DLinear, PatchTST, iTransformer, N-BEATS, TimesNet)
with seed 42 on all 9 datasets at `H=96`, dumping residuals to
`results/benchmark/<dataset>/<forecaster>/residuals.pt`.

```bash
python scripts/regimes/run_full_benchmark.py --phase forecasters --device cuda
```

For the seed-robustness study (Appendix K, 90 additional residuals):

```bash
python scripts/regimes/run_multi_seed_forecasters.py --gpu 0
python scripts/regimes/run_multi_seed_forecasters.py --gpu 1   # if available
```

## Step 3 — Run the decomposition harness (~30 min CPU + ~30 min GPU)

```bash
python scripts/regimes/run_full_benchmark.py --phase methods --device cuda
```

Runs the 7 decomposition methods (PCA, Dense AE-16, Dense AE-64, k-means,
TopK SAE, ICA, Spectral) under the proper train/test protocol on each
(dataset, forecaster) pair, totalling 855 of 945 configurations (Spectral is
omitted on 6 high-dim datasets; see paper Section 3.4).

## Step 4 — Reproduce headline findings

### Finding 1 (no single method dominates) and Finding 3 (proper protocol)

```bash
python scripts/regimes/evaluate_regime_metrics.py
python scripts/regimes/generate_figures.py        # paper Tables 3, 4 + Figs 2, 3
```

### Finding 2 (cross-forecaster sharing)

```bash
python scripts/regimes/evaluate_regime_stability.py --alignment hungarian
python scripts/regimes/analyze_trivial_similarity.py            # Appendix F
python scripts/regimes/compute_cross_seed_all.py                # Appendix K
python scripts/regimes/plot_alignment_sources.py                # Figure 4
```

### Finding 4 (lag-1 selector + learned selector)

```bash
python scripts/regimes/evaluate_selectors.py --base-dir results/benchmark
```

### Statistical tables (Appendix G, J)

```bash
python scripts/regimes/compute_bootstrap_wilcoxon.py
```

### Seed-robustness study (Appendix K)

```bash
python scripts/regimes/compute_forecaster_seed_robustness.py
```

### Multi-horizon stability (Appendix I)

```bash
python scripts/regimes/run_multi_horizon.py --datasets ETTh1 Weather \
    --horizons 48 96 192 336
```

## Expected outputs

After Step 4 you should have:

- `results/benchmark/analysis.json` — main numbers cited in Sections 4.1-4.2
- `results/benchmark/selector_comparison.json` — selector / lag-1 numbers
- `results/benchmark/bootstrap_ci.json`, `wilcoxon_full.json` — Appendix J/G
- `results/benchmark/forecaster_seed_robustness.json` — Appendix K
- `paper/figures/*.pdf` — all paper figures regenerated

## End-to-end smoke test (~20 min on a single GPU)

For reviewers who want to verify the pipeline end-to-end on a single small
dataset:

```bash
python scripts/regimes/run_full_benchmark.py \
    --datasets ETTh1 --forecasters dlinear patchtst --device cuda
python scripts/regimes/evaluate_regime_metrics.py --datasets ETTh1
```

This runs only ETTh1 with DLinear and PatchTST (the two cheapest forecasters)
and produces a partial `analysis.json` that should match the ETTh1 row of
paper Table 3 within seed noise.

## Troubleshooting

- **CUDA OOM on Electricity/Traffic.** Reduce N-BEATS/TimesNet model sizes via
  `--model-scale 0.5` (paper Section 3.1 documents this).
- **Spectral clustering hangs on high-dim datasets.** Expected; we omit
  Spectral on Electricity/Traffic/ILI/Exchange/Weather/ETTm1 and report 855
  rather than 945 configurations.
- **ILI Hungarian alignment is heterogeneous.** Expected stress case (3 test
  windows at `H=96`); paper Section 4.2 reports 4-11x null on the other 8
  datasets and discusses the ILI exception.

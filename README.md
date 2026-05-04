# ResidualBench

**ResidualBench: A Benchmark and Evaluation Protocol for Cross-Model Forecast Failure Mode Discovery**

ResidualBench is a benchmark for systematically comparing methods that discover forecast failure modes from residual windows. It includes pre-computed residuals from 5 forecasters across 9 datasets and 7 decomposition methods.

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from residualbench import ResidualBench, PCAMethod, TopKSAEMethod

bench = ResidualBench()
bench.load_residuals("ETTh1/dlinear", "results/benchmark/ETTh1/dlinear/residuals.pt")

results = bench.evaluate(PCAMethod(16), name="PCA", seeds=[42, 11, 22])
results = bench.evaluate(TopKSAEMethod(64, k=16), name="TopK", seeds=[42, 11, 22])
bench.summary()
```

## Datasets

| Dataset | Domain | Channels | Frequency |
|---------|--------|----------|-----------|
| ETTh1/h2 | Energy | 7 | 1h |
| ETTm1/m2 | Energy | 7 | 15min |
| Weather | Climate | 21 | 10min |
| Electricity | Power | 321 | 1h |
| Traffic | Transport | 862 | 1h |
| Exchange | Finance | 8 | 1d |
| ILI | Medical | 7 | 1w |

## Forecasters

DLinear, PatchTST, iTransformer, N-BEATS, TimesNet

## Methods

PCA, Dense AE-16, Dense AE-64, k-means, TopK SAE, ICA, Spectral

## Running the Full Benchmark

```bash
# Download datasets
python scripts/download_data.py --include-large

# Train all forecasters and run benchmark
python scripts/regimes/run_full_benchmark.py --phase all --device cuda

# Generate figures
python scripts/regimes/generate_figures.py

# Evaluate model selection
python scripts/regimes/evaluate_selectors.py --base-dir results/benchmark
```

## Custom Method Evaluation

Implement the `fit/encode/reconstruct` protocol:

```python
class MyMethod:
    def fit(self, train_data):
        # train_data: np.ndarray (N_train, D)
        ...

    def encode(self, data):
        # data: np.ndarray (N, D) -> codes: np.ndarray (N, n_latent)
        ...

    def reconstruct(self, data):
        # data: np.ndarray (N, D) -> recon: np.ndarray (N, D)
        ...

bench.evaluate(MyMethod(), name="MyMethod", seeds=[42, 11, 22])
```

## Citation

```bibtex
@inproceedings{residualbench2026,
  title={ResidualBench: A Benchmark and Evaluation Protocol for Cross-Model Forecast Failure Mode Discovery},
  author={Anonymous},
  booktitle={NeurIPS 2026 Evaluations and Datasets Track},
  year={2026}
}
```

## License

MIT

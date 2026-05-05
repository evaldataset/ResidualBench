# Contributing to ResidualBench

Thanks for considering a contribution! ResidualBench grows by community PRs that
add new forecasters, new decomposition methods, or new robustness analyses.

## Adding a new decomposition method

Implement the `Method` protocol used by `residualbench/methods.py`:

```python
class MyMethod:
    def fit(self, train_data: np.ndarray) -> None: ...
    def encode(self, data: np.ndarray) -> np.ndarray: ...
    def reconstruct(self, data: np.ndarray) -> np.ndarray: ...
```

Then register it via:

```python
from residualbench import ResidualBench
bench = ResidualBench()
bench.evaluate(MyMethod(), name="MyMethod", seeds=[42, 11, 22])
```

A PR adding a new method should include:
- An implementation file under `residualbench/`.
- A unit test in `tests/test_residualbench.py` covering at least the synthetic
  motif templates from Appendix B of the paper.
- Documentation of any non-default hyperparameters in the method's docstring.
- A short benchmark run report on at least ETTh1 (the smoke pipeline takes
  about 22 minutes on a single 24 GB GPU).

## Adding a new forecaster

Train your forecaster on the canonical 9 datasets (or a subset) and dump
residual tensors to:

```
results/benchmark/<dataset>/<forecaster>/seed_<n>/residuals.pt
```

Each `residuals.pt` must contain at minimum the keys `train_input`,
`train_target`, `train_pred`, `train_residual` and the corresponding `test_*`
splits. See `scripts/regimes/run_full_benchmark.py` for the reference dumping
pipeline.

A PR adding a new forecaster should include:
- A configuration file under `configs/regimes/`.
- A short README section describing the architecture and any modifications
  required to fit our 24 GB GPU budget (capacity reductions on Electricity /
  Traffic should be flagged in line with paper Limitation 5).
- The MD5 of each `residuals.pt` so reviewers can verify integrity.

## Maintenance commitments

The authors commit to:

1. Quarterly release cadence for at least the first year following the
   proceedings publication, accepting community PRs that pass CI.
2. A public issue tracker for benchmark-result regressions.
3. Freezing the v0.1 residual artifact (Croissant'd, MD5-pinned) so historical
   numbers remain reproducible after later versions add forecasters or
   methods.

## Versioning

Every benchmark release tags both the residual archive (Hugging Face dataset
version) and the harness (pip-installable `residualbench`) with matching
semver tags. Numbers in the paper correspond to `residualbench==0.1.0` and the
v0.1 residual archive on Hugging Face. Croissant's `prov:wasGeneratedBy`
entry pins these versions explicitly.

## Code style and tests

- Run `ruff check .` and `pytest tests/test_residualbench.py -q` locally before
  opening a PR; CI runs the same checks.
- Prefer minimal, well-documented changes; large refactors should open an
  issue for discussion first.
- Anonymous contributions during the NeurIPS 2026 review window are welcome
  and will be attributed in the camera-ready acknowledgements upon request.

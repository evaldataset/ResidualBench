"""ResidualBench: Evaluation protocol for forecast failure mode discovery.

Quick start:
    >>> from residualbench import ResidualBench, PCAMethod
    >>> bench = ResidualBench()
    >>> bench.add_dataset("my_data", train_residuals, test_residuals)
    >>> results = bench.evaluate(PCAMethod(16), name="PCA", seeds=[42, 11, 22])
    >>> bench.summary()
"""

from residualbench.bench import ResidualBench, ResidualMethod
from residualbench.methods import (
    DenseAEMethod,
    ICAMethod,
    KMeansMethod,
    PCAMethod,
    TopKSAEMethod,
)
from residualbench.metrics import compute_motif_metrics

__version__ = "0.2.0"
__all__ = [
    "ResidualBench",
    "ResidualMethod",
    "compute_motif_metrics",
    "PCAMethod",
    "KMeansMethod",
    "ICAMethod",
    "DenseAEMethod",
    "TopKSAEMethod",
]

"""Tests for the residualbench package."""

from __future__ import annotations

import numpy as np
import pytest

from residualbench import (
    DenseAEMethod,
    ICAMethod,
    KMeansMethod,
    PCAMethod,
    ResidualBench,
    compute_motif_metrics,
)


@pytest.fixture()
def synthetic_data() -> tuple[np.ndarray, np.ndarray]:
    """Create synthetic train/test residual data."""
    rng = np.random.RandomState(42)
    train = rng.randn(200, 96).astype(np.float32)
    test = rng.randn(50, 96).astype(np.float32)
    return train, test


class TestMetrics:
    def test_compute_motif_metrics_returns_keys(self, synthetic_data: tuple) -> None:
        train, test = synthetic_data
        import torch
        codes = torch.randn(50, 16)
        recon = torch.from_numpy(test)
        m = compute_motif_metrics(test, codes, recon)
        assert "recon_mse" in m
        assert "mean_cohesion" in m
        assert "mean_variance_reduction" in m

    def test_perfect_reconstruction(self, synthetic_data: tuple) -> None:
        _, test = synthetic_data
        import torch
        codes = torch.randn(50, 16)
        recon = torch.from_numpy(test)
        m = compute_motif_metrics(test, codes, recon)
        assert m["recon_mse"] < 1e-6


class TestMethods:
    def test_pca_fit_encode_reconstruct(self, synthetic_data: tuple) -> None:
        train, test = synthetic_data
        m = PCAMethod(8)
        m.fit(train)
        codes = m.encode(test)
        recon = m.reconstruct(test)
        assert codes.shape == (50, 8)
        assert recon.shape == test.shape

    def test_kmeans_fit_encode_reconstruct(self, synthetic_data: tuple) -> None:
        train, test = synthetic_data
        m = KMeansMethod(8)
        m.fit(train)
        codes = m.encode(test)
        recon = m.reconstruct(test)
        assert codes.shape == (50, 8)
        assert recon.shape == test.shape

    def test_ica_fit_encode_reconstruct(self, synthetic_data: tuple) -> None:
        train, test = synthetic_data
        m = ICAMethod(8)
        m.fit(train)
        codes = m.encode(test)
        recon = m.reconstruct(test)
        assert codes.shape == (50, 8)
        assert recon.shape == test.shape

    def test_dense_ae_fit_encode_reconstruct(self, synthetic_data: tuple) -> None:
        train, test = synthetic_data
        m = DenseAEMethod(8, n_epochs=2)
        m.fit(train)
        codes = m.encode(test)
        recon = m.reconstruct(test)
        assert codes.shape == (50, 8)
        assert recon.shape == test.shape


class TestBench:
    def test_add_dataset_and_evaluate(self, synthetic_data: tuple) -> None:
        train, test = synthetic_data
        bench = ResidualBench()
        bench.add_dataset("test_ds", train, test)
        results = bench.evaluate(PCAMethod(8), name="PCA", seeds=[42])
        assert len(results) == 1
        assert results[0]["dataset"] == "test_ds"
        assert results[0]["method"] == "PCA"

    def test_summary_returns_dict(self, synthetic_data: tuple) -> None:
        train, test = synthetic_data
        bench = ResidualBench()
        bench.add_dataset("ds1", train, test)
        bench.evaluate(PCAMethod(8), name="PCA", seeds=[42])
        summary = bench.summary()
        assert "PCA" in summary
        assert "ds1" in summary["PCA"]

    def test_load_residuals_rejects_bad_keys(self, tmp_path: str) -> None:
        import torch
        bad_path = tmp_path / "bad.pt"
        torch.save({"wrong_key": torch.zeros(10)}, str(bad_path))
        bench = ResidualBench()
        with pytest.raises(KeyError, match="train_residual"):
            bench.load_residuals("bad", str(bad_path))

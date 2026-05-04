"""Built-in method wrappers implementing the ResidualMethod protocol."""

from __future__ import annotations

import numpy as np


class PCAMethod:
    """PCA baseline for residual decomposition."""

    def __init__(self, n_components: int = 16) -> None:
        self.n_components = n_components
        self._pca = None

    def fit(self, train_data: np.ndarray) -> None:
        from sklearn.decomposition import PCA
        self._pca = PCA(
            n_components=self.n_components, svd_solver="randomized",
        )
        self._pca.fit(train_data)

    def encode(self, data: np.ndarray) -> np.ndarray:
        return self._pca.transform(data)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        codes = self._pca.transform(data)
        return codes @ self._pca.components_ + self._pca.mean_


class KMeansMethod:
    """k-means clustering baseline."""

    def __init__(self, n_clusters: int = 16) -> None:
        self.n_clusters = n_clusters
        self._km = None

    def fit(self, train_data: np.ndarray) -> None:
        from sklearn.cluster import KMeans
        self._km = KMeans(n_clusters=self.n_clusters, n_init=10)
        self._km.fit(train_data)

    def encode(self, data: np.ndarray) -> np.ndarray:
        labels = self._km.predict(data)
        codes = np.zeros((len(data), self.n_clusters), dtype=np.float32)
        codes[np.arange(len(data)), labels] = 1.0
        return codes

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        labels = self._km.predict(data)
        return self._km.cluster_centers_[labels].astype(np.float32)


class ICAMethod:
    """Independent Component Analysis baseline."""

    def __init__(self, n_components: int = 16) -> None:
        self.n_components = n_components
        self._ica = None

    def fit(self, train_data: np.ndarray) -> None:
        from sklearn.decomposition import FastICA
        self._ica = FastICA(n_components=self.n_components, max_iter=500)
        self._ica.fit(train_data)

    def encode(self, data: np.ndarray) -> np.ndarray:
        return self._ica.transform(data)

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        codes = self._ica.transform(data)
        return self._ica.inverse_transform(codes).astype(np.float32)


class DenseAEMethod:
    """Dense autoencoder baseline (no sparsity)."""

    def __init__(self, n_latent: int = 16, n_epochs: int = 20) -> None:
        self.n_latent = n_latent
        self.n_epochs = n_epochs
        self._model = None

    def fit(self, train_data: np.ndarray) -> None:
        import torch
        import torch.nn as nn

        D = train_data.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _AE(nn.Module):
            def __init__(self, d_in: int, d_latent: int) -> None:
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Linear(d_in, 256), nn.ReLU(),
                    nn.Linear(256, d_latent),
                )
                self.dec = nn.Sequential(
                    nn.Linear(d_latent, 256), nn.ReLU(),
                    nn.Linear(256, d_in),
                )

            def forward(self, x: torch.Tensor) -> dict:
                z = self.enc(x)
                return {"codes": z, "reconstruction": self.dec(z)}

        self._model = _AE(D, self.n_latent).to(device)
        self._device = device
        opt = torch.optim.Adam(self._model.parameters(), lr=1e-3)
        t = torch.from_numpy(train_data).float()
        ld = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(t), batch_size=128, shuffle=True,
        )
        for _ in range(self.n_epochs):
            self._model.train()
            for (b,) in ld:
                b = b.to(device)
                out = self._model(b)
                nn.functional.mse_loss(out["reconstruction"], b).backward()
                opt.step()
                opt.zero_grad()
        self._model.eval()

    def encode(self, data: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(data).float().to(self._device)
            return self._model(t)["codes"].cpu().numpy()

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(data).float().to(self._device)
            return self._model(t)["reconstruction"].cpu().numpy()


class TopKSAEMethod:
    """TopK Sparse Autoencoder method."""

    def __init__(
        self,
        n_units: int = 64,
        k: int = 16,
        hidden_dim: int = 256,
        n_epochs: int = 20,
    ) -> None:
        self.n_units = n_units
        self.k = k
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self._model = None

    def fit(self, train_data: np.ndarray) -> None:
        import torch
        import torch.nn as nn

        from src.regimes.regime_autoencoder import (
            SparseResidualRegimeAutoencoder,
        )

        D = train_data.shape[1]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        self._model = SparseResidualRegimeAutoencoder(
            D, self.n_units, self.hidden_dim, "topk", topk_k=self.k,
        ).to(device)
        opt = torch.optim.AdamW(
            self._model.parameters(), lr=1e-3, weight_decay=1e-5,
        )
        t = torch.from_numpy(train_data).float()
        ld = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(t), batch_size=128, shuffle=True,
        )
        for _ in range(self.n_epochs):
            self._model.train()
            for (b,) in ld:
                b = b.to(device)
                out = self._model(b)
                nn.functional.mse_loss(out["reconstruction"], b).backward()
                opt.step()
                opt.zero_grad()
        self._model.eval()

    def encode(self, data: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(data).float().to(self._device)
            return self._model(t)["regimes"].cpu().numpy()

    def reconstruct(self, data: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            t = torch.from_numpy(data).float().to(self._device)
            return self._model(t)["reconstruction"].cpu().numpy()

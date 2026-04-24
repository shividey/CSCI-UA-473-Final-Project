from __future__ import annotations

import numpy as np


class ArrayStandardScaler:
    def __init__(self, mean_: np.ndarray, scale_: np.ndarray):
        self.mean_ = np.asarray(mean_, dtype=float)
        self.scale_ = np.asarray(scale_, dtype=float)

    @classmethod
    def fit(cls, x: np.ndarray) -> "ArrayStandardScaler":
        mean_ = x.mean(axis=0)
        scale_ = x.std(axis=0, ddof=0)
        scale_ = np.where(scale_ == 0, 1.0, scale_)
        return cls(mean_=mean_, scale_=scale_)

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return (x - self.mean_) / self.scale_


class ArrayPCA:
    def __init__(
        self,
        components_: np.ndarray,
        explained_variance_ratio_: np.ndarray,
        mean_: np.ndarray,
    ):
        self.components_ = np.asarray(components_, dtype=float)
        self.explained_variance_ratio_ = np.asarray(explained_variance_ratio_, dtype=float)
        self.mean_ = np.asarray(mean_, dtype=float)
        self.n_components_ = self.components_.shape[0]

    @classmethod
    def fit(cls, x: np.ndarray, variance_threshold: float = 0.90) -> "ArrayPCA":
        x = np.asarray(x, dtype=float)
        mean_ = x.mean(axis=0)
        centered = x - mean_
        _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
        explained_variance = (singular_values**2) / max(x.shape[0] - 1, 1)
        explained_ratio = explained_variance / explained_variance.sum()
        cumulative = np.cumsum(explained_ratio)
        n_components = int(np.searchsorted(cumulative, variance_threshold) + 1)
        return cls(
            components_=vt[:n_components],
            explained_variance_ratio_=explained_ratio[:n_components],
            mean_=mean_,
        )

    def transform(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        centered = x - self.mean_
        return centered @ self.components_.T


class ArrayKMeans:
    def __init__(self, cluster_centers_: np.ndarray, labels_: np.ndarray):
        self.cluster_centers_ = np.asarray(cluster_centers_, dtype=float)
        self.labels_ = np.asarray(labels_, dtype=int)
        self.n_clusters = self.cluster_centers_.shape[0]

    @staticmethod
    def _dist_sq(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
        return ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        n_clusters: int,
        n_init: int = 100,
        max_iter: int = 300,
        random_state: int = 42,
    ) -> "ArrayKMeans":
        x = np.asarray(x, dtype=float)
        best_inertia = None
        best_labels = None
        best_centers = None

        for init_idx in range(n_init):
            rng = np.random.default_rng(random_state + init_idx)
            center_ids = [rng.integers(len(x))]
            while len(center_ids) < n_clusters:
                centers = x[center_ids]
                dist_sq = cls._dist_sq(x, centers).min(axis=1)
                probs = dist_sq / dist_sq.sum()
                center_ids.append(rng.choice(len(x), p=probs))

            centers = x[center_ids].copy()
            labels = None

            for _ in range(max_iter):
                dist_sq = cls._dist_sq(x, centers)
                new_labels = dist_sq.argmin(axis=1)
                new_centers = centers.copy()

                for cluster_id in range(n_clusters):
                    members = x[new_labels == cluster_id]
                    if len(members) == 0:
                        new_centers[cluster_id] = x[rng.integers(len(x))]
                    else:
                        new_centers[cluster_id] = members.mean(axis=0)

                if labels is not None and np.array_equal(new_labels, labels):
                    centers = new_centers
                    labels = new_labels
                    break

                centers = new_centers
                labels = new_labels

            inertia = float(((x - centers[labels]) ** 2).sum())
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centers = centers.copy()

        return cls(cluster_centers_=best_centers, labels_=best_labels)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self._dist_sq(x, self.cluster_centers_).argmin(axis=1)

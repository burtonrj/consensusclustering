from __future__ import annotations

from itertools import combinations
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def resample(x: np.ndarray, frac: float) -> tuple[np.ndarray, np.ndarray]:
    resampled_indices = np.random.choice(
        range(x.shape[0]), size=int(x.shape[0] * frac), replace=False
    )
    return resampled_indices, x[resampled_indices, :]


def compute_connectivity_matrix(labels: np.ndarray) -> np.ndarray:
    out_of_bag_idx = np.where(labels == -1)[0]
    connectivity_matrix = np.equal.outer(labels, labels).astype("int8")
    rows, cols = zip(*list(combinations(out_of_bag_idx, 2)))
    connectivity_matrix[rows, cols] = 0
    connectivity_matrix[cols, rows] = 0
    connectivity_matrix[out_of_bag_idx, out_of_bag_idx] = 0
    return connectivity_matrix


def compute_identity_matrix(x: np.ndarray, resampled_indices: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    identity_matrix = np.zeros((n, n), dtype="int8")
    rows, cols = zip(*list(combinations(resampled_indices, 2)))
    identity_matrix[rows, cols] = 1
    identity_matrix[cols, rows] = 1
    identity_matrix[resampled_indices, resampled_indices] = 1
    return identity_matrix


def compute_consensus_matrix(
    connectivity_matrices: list[np.ndarray], identity_matrices: list[np.ndarray]
) -> np.ndarray:
    return np.nan_to_num(
        np.sum(connectivity_matrices, axis=0) / np.sum(identity_matrices, axis=0), nan=0
    )


def valid_clustering_obj(clustering_obj) -> bool:
    if not hasattr(clustering_obj, "fit_predict"):
        return False
    if not hasattr(clustering_obj, "set_params"):
        return False
    return True


def cluster(
    x: np.ndarray, resample_frac: float, k: int, clustering_obj
) -> tuple[Type, np.ndarray, np.ndarray]:
    if not valid_clustering_obj(clustering_obj):
        raise ValueError("clustering_obj must have fit_predict and set_params methods")
    clustering_obj.set_params(n_clusters=k)
    resampled_indices, resampled_x = resample(x, resample_frac)
    resampled_labels = clustering_obj.fit_predict(resampled_x)
    labels = np.full(x.shape[0], -1)
    labels[resampled_indices] = resampled_labels
    return clustering_obj, resampled_indices, labels


class ConsensusClustering:
    def __init__(
        self,
        clustering_obj,
        min_clusters: int,
        max_clusters: int,
        n_resamples: int,
        resample_frac: float = 0.5,
    ):
        self.clustering_obj = clustering_obj
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.n_resamples = n_resamples
        self.resample_frac = resample_frac
        self.consensus_matrices_: list[np.ndarray] = []

    def consensus_k(self, k: int) -> np.ndarray:
        if len(self.consensus_matrices_) == 0:
            raise ValueError("Consensus matrices not computed yet.")
        return self.consensus_matrices_[k - self.min_clusters]

    def fit(self, x: np.ndarray):
        for k in range(self.min_clusters, self.max_clusters + 1):
            connectivity_matrices = []
            identity_matrices = []
            for _ in range(self.n_resamples):
                clustering_obj, resampled_indices, labels = cluster(
                    x, self.resample_frac, k, self.clustering_obj
                )
                connectivity_matrices.append(compute_connectivity_matrix(labels))
                identity_matrices.append(compute_identity_matrix(x, resampled_indices))
            self.consensus_matrices_.append(
                compute_consensus_matrix(connectivity_matrices, identity_matrices)
            )

    def hist(self, k: int):
        hist, bins = np.histogram(self.consensus_k(k).ravel(), density=True)
        return hist, bins

    def cdf(self, k: int):
        hist, bins = self.hist(k)
        ecdf = np.cumsum(hist)
        bin_widths = np.diff(bins)
        ecdf = ecdf * bin_widths
        return ecdf, hist, bins

    def area_under_cdf(self, k: int):
        ecdf, hist, bins = self.cdf(k)
        return np.sum(ecdf * (bins[1:] - bins[:-1]))

    def change_in_area_under_cdf(self):
        auc = [
            self.area_under_cdf(k)
            for k in range(self.min_clusters, self.max_clusters + 1)
        ]
        return np.diff(auc)

    def best_k(self):
        change = self.change_in_area_under_cdf()
        largest_change_in_auc = np.argmax(change) + 1
        return list(range(self.min_clusters, self.max_clusters + 1))[
            largest_change_in_auc
        ]

    def plot_clustermap(self, k: int, **kwargs):
        return sns.clustermap(self.consensus_k(k), **kwargs)

    def plot_hist(self, k: int, ax: plt.Axes | None = None):
        ax = ax if ax is not None else plt.subplots(figsize=(6.5, 6.5))[1]
        hist, bins = self.hist(k)
        ax.bar(bins[:-1], hist, width=np.diff(bins))
        ax.set_xlabel("Consensus index value")
        ax.set_ylabel("Density")
        return ax

    def plot_cdf(self, ax: plt.Axes | None = None):
        ax = ax if ax is not None else plt.subplots(figsize=(6.5, 6.5))[1]
        for k in range(self.min_clusters, self.max_clusters + 1):
            ecdf, hist, bins = self.cdf(k)
            ax.plot(bins[:-1], ecdf, label=f"{k} clusters")
        ax.set_xlabel("Consensus index value")
        ax.set_ylabel("CDF")
        ax.legend()
        return ax

    def plot_change_area_under_cdf(self, ax: plt.Axes | None = None):
        ax = ax if ax is not None else plt.subplots(figsize=(6.5, 6.5))[1]
        k = [k for k in range(self.min_clusters + 1, self.max_clusters + 1)]
        change = self.change_in_area_under_cdf()
        ax.plot(
            k,
            change,
            "--",
            marker="o",
            markerfacecolor="white",
            markeredgecolor="black",
        )
        ax.set_xlabel("K")
        ax.set_ylabel("Change in area under CDF")
        return ax

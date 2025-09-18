import numpy as np
import pytest
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_classification

from consensusclustering.consensus import (
    ConsensusClustering,
    compute_connectivity_matrix,
    compute_consensus_matrix,
    compute_identity_matrix,
    resample,
)


@pytest.fixture(scope="session")
def uniform_data():
    """Generate uniform random data with fixed seed for consistent testing."""
    np.random.seed(42)
    return np.random.uniform(low=0.0, high=1.0, size=(60, 600))


@pytest.fixture(scope="session")
def gaussian_data():
    """Generate Gaussian data with fixed seed for consistent testing."""
    return make_classification(
        n_samples=60,
        n_features=600,
        n_informative=600,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=42,
        class_sep=10.0,
    )[0]


@pytest.fixture(scope="session")
def labels_data():
    """Generate test labels for connectivity matrix testing."""
    return np.array([0, 1, 1, 2, 0, -1, -1, 2, 2, 1, 3, 3, -1, -1])


@pytest.mark.parametrize("frac", [0.1, 0.5, 0.9])
def test_resample(frac, uniform_data):
    idx, sample = resample(uniform_data, frac=frac)
    assert idx.shape[0] == sample.shape[0]
    assert idx.shape[0] == int(uniform_data.shape[0] * frac)


def test_connectivity_matrix(labels_data):
    connectivity_matrix = compute_connectivity_matrix(labels_data)
    assert connectivity_matrix.sum() == 26
    for pairs in [(0, 4), (1, 2), (1, 9), (2, 9), (3, 7), (3, 8), (7, 8), (10, 11)]:
        if connectivity_matrix[pairs[0], pairs[1]] == 0:
            print(pairs)
        assert connectivity_matrix[pairs[0], pairs[1]] == 1
        assert connectivity_matrix[pairs[1], pairs[0]] == 1
    for pairs in [(5, 6), (5, 12), (6, 12), (11, 12)]:
        assert connectivity_matrix[pairs[0], pairs[1]] == 0
        assert connectivity_matrix[pairs[1], pairs[0]] == 0


def test_compute_identity_matrix(uniform_data):
    identity_matrix = compute_identity_matrix(uniform_data, np.array([0, 1, 2, 3, 4]))
    assert identity_matrix.shape == (60, 60)
    assert identity_matrix.sum() == 25
    for pairs in [
        (0, 1),
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]:
        assert identity_matrix[pairs[0], pairs[1]] == 1
        assert identity_matrix[pairs[1], pairs[0]] == 1


def test_computer_consensus_matrix():
    connectivity_matrices = [np.array([[1, 1], [1, 0]]), np.array([[0, 1], [0, 0]])]
    identity_matrices = [
        np.array([[1, 1], [1, 1]]),
        np.array([[1, 1], [0, 0]]),
    ]
    consensus_matrix = compute_consensus_matrix(
        connectivity_matrices, identity_matrices
    )
    assert consensus_matrix.shape == (2, 2)
    assert np.array_equal(consensus_matrix, np.array([[0.5, 1.0], [1.0, 0.0]]))


@pytest.mark.parametrize("n_jobs", [0, -1])
def test_consensus_cluster_uniform(n_jobs, uniform_data):
    # Set a consistent seed for the clustering process
    np.random.seed(42)
    clustering = ConsensusClustering(
        clustering_obj=AgglomerativeClustering(metric="euclidean", linkage="average"),
        min_clusters=2,
        max_clusters=6,
        n_resamples=100,
        resample_frac=0.8,
    )
    clustering.fit(uniform_data, n_jobs=n_jobs)
    assert len(clustering.consensus_matrices_) == 5
    hist, bins = clustering.hist(3)
    assert np.mean(hist) > 0.7
    assert clustering.best_k() == 2


@pytest.mark.parametrize("n_jobs", [-1, 0])
def test_consensus_cluster_gaussian(n_jobs, gaussian_data):
    # Set a consistent seed for the clustering process
    np.random.seed(42)
    clustering = ConsensusClustering(
        clustering_obj=AgglomerativeClustering(metric="euclidean", linkage="average"),
        min_clusters=2,
        max_clusters=6,
        n_resamples=100,
        resample_frac=0.5,
    )
    clustering.fit(gaussian_data, n_jobs=n_jobs)
    assert len(clustering.consensus_matrices_) == 5
    hist, bins = clustering.hist(3)
    assert hist[0] > 0.0
    assert hist[9] > 0.0
    assert all(hist[1:9] == 0.0)
    assert clustering.best_k() == 3

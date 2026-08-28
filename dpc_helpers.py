"""Shared implementation helpers for density-peak clustering algorithms."""

import numpy as np


def validate_common_inputs(
    X,
    den_thres,
    center_quantile,
    scaling,
    cutoff_quantile,
    center_z,
):
    """Validate inputs shared by DPC and GGDPC."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if X.shape[0] < 2:
        raise ValueError("X must contain at least two observations.")
    if X.shape[1] < 1:
        raise ValueError("X must contain at least one feature.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")
    if not isinstance(scaling, (bool, np.bool_)):
        raise TypeError("scaling must be a boolean.")

    den_thres = float(den_thres)
    cutoff_quantile = float(cutoff_quantile)
    center_z = float(center_z)
    if not np.isfinite(den_thres) or not 0 <= den_thres <= 1:
        raise ValueError("den_thres must be in [0, 1].")
    if not np.isfinite(cutoff_quantile) or not 0 <= cutoff_quantile <= 1:
        raise ValueError("cutoff_quantile must be in [0, 1].")
    if not np.isfinite(center_z) or center_z < 0:
        raise ValueError("center_z must be a finite non-negative number.")

    if center_quantile is not None:
        center_quantile = float(center_quantile)
        if not np.isfinite(center_quantile) or not 0 <= center_quantile <= 1:
            raise ValueError("center_quantile must be in [0, 1].")

    return X, den_thres, center_quantile, cutoff_quantile, center_z


def prepare_coordinates(X, scaling):
    """Create the coordinate system used by all internal distance calculations."""
    if not scaling:
        return X, None

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def validate_density(den_est, n_samples):
    """Return a finite one-dimensional density array."""
    den_est = np.asarray(den_est, dtype=float)
    if den_est.shape != (n_samples,):
        raise ValueError("den_est must have shape (n_samples,).")
    if not np.all(np.isfinite(den_est)):
        raise ValueError("den_est must contain only finite values.")
    return den_est


def validate_update_locations(grad_new, X, scaler):
    """Transform externally supplied update locations into working coordinates."""
    grad_new = np.asarray(grad_new, dtype=float)
    if grad_new.shape != X.shape:
        raise ValueError("grad_new must have shape (n_samples, n_features).")
    if not np.all(np.isfinite(grad_new)):
        raise ValueError("grad_new must contain only finite values.")
    return scaler.transform(grad_new) if scaler is not None else grad_new


def validate_distance_matrix(dist_mat, n_samples):
    """Validate a precomputed Euclidean-style distance matrix."""
    dist_mat = np.asarray(dist_mat, dtype=float)
    if dist_mat.shape != (n_samples, n_samples):
        raise ValueError("dist_mat must have shape (n_samples, n_samples).")
    if not np.all(np.isfinite(dist_mat)):
        raise ValueError("dist_mat must contain only finite values.")
    if np.any(dist_mat < 0):
        raise ValueError("dist_mat cannot contain negative distances.")
    if not np.allclose(dist_mat, dist_mat.T):
        raise ValueError("dist_mat must be symmetric.")
    if not np.allclose(np.diag(dist_mat), 0):
        raise ValueError("dist_mat must have a zero diagonal.")
    return dist_mat

def resolve_bandwidth(X, bandwidth=None):
    if bandwidth is not None:
        bandwidth = float(bandwidth)
        if not np.isfinite(bandwidth) or bandwidth <= 0:
            raise ValueError("bandwidth must be positive and finite.")
        return bandwidth

    n_samples, n_features = X.shape
    silverman_factor = (n_samples * (n_features + 2) / 4) ** (-1 / (n_features + 4))

    return silverman_factor * np.mean(np.std(X, axis=0))


def cutoff_density(dist_mat, cutoff_quantile):
    """Estimate density by counting neighbors inside a distance cutoff."""
    n_samples = dist_mat.shape[0]
    upper = dist_mat[np.triu_indices(n_samples, k=1)]
    cutoff_distance = np.quantile(upper, cutoff_quantile)

    within_cutoff = dist_mat < cutoff_distance
    np.fill_diagonal(within_cutoff, False)
    den_est = within_cutoff.sum(axis=1).astype(float)
    return den_est, cutoff_distance


def nearest_higher_density(X, den_est, dist_mat=None):
    """Find each point's nearest predecessor in decreasing-density order."""
    n_samples = X.shape[0]
    # Sort points by density in descending order
    sorted_indices = np.argsort(-den_est, kind="stable")
    delta = np.empty(n_samples, dtype=float)
    nearest_higher = np.full(n_samples, -1, dtype=int)

    highest = sorted_indices[0]
    if dist_mat is not None:
        # Assign the maximum distance index to the point with the highest density value
        delta[highest] = np.nextafter(np.max(dist_mat[highest]), np.inf)
        for rank in range(1, n_samples):
            idx_i = sorted_indices[rank]
            higher_ind = sorted_indices[:rank]
            # Compute distances to points with higher density
            distances = dist_mat[idx_i, higher_ind]
            parent = higher_ind[np.argmin(distances)]
            delta[idx_i] = dist_mat[idx_i, parent]
            nearest_higher[idx_i] = parent
    else:
        delta[highest] = np.nextafter(np.max(np.linalg.norm(X - X[highest], axis=1)), np.inf)
        for rank in range(1, n_samples):
            idx_i = sorted_indices[rank]
            higher_ind = sorted_indices[:rank]
            # Compute distances to points with higher density
            distances = np.linalg.norm(X[idx_i] - X[higher_ind], axis=1)
            parent = higher_ind[np.argmin(distances)]
            delta[idx_i] = np.linalg.norm(X[idx_i] - X[parent])
            nearest_higher[idx_i] = parent

    return delta, nearest_higher, sorted_indices


def gradient_guided_nearest_higher(X, shifted_X, den_est, dist_mat=None):
    """Find higher-density parents using one-step update locations as guides."""
    n_samples = X.shape[0]
    # Sort points by density in descending order
    sorted_indices = np.argsort(-den_est, kind="stable")
    delta = np.empty(n_samples, dtype=float)
    nearest_higher = np.full(n_samples, -1, dtype=int)

    highest = sorted_indices[0]
    if dist_mat is None:
        max_distance = np.max(np.linalg.norm(X - X[highest], axis=1))
    else:
        max_distance = np.max(dist_mat[highest])
    # Assign the maximum distance to the point with the highest density value
    delta[highest] = np.nextafter(max_distance, np.inf)

    for rank in range(1, n_samples):
        idx_i = sorted_indices[rank]
        higher_ind = sorted_indices[:rank]
        # Compute distances to points with higher density
        guide_distances = np.linalg.norm(shifted_X[idx_i] - X[higher_ind], axis=1)
        parent = higher_ind[np.argmin(guide_distances)]

        if dist_mat is None:
            delta[idx_i] = np.linalg.norm(X[idx_i] - X[parent])
        else:
            delta[idx_i] = dist_mat[idx_i, parent]
        nearest_higher[idx_i] = parent

    return delta, nearest_higher, sorted_indices

def ggdpc_graph_distances(X, nearest_higher, sorted_indices):
    """Compute the graph distances from each point to the root node (i.e. the point with the highest density) of 
       the GGDPC tree."""
    n_samples = X.shape[0]
    graph_dist = np.full(n_samples, np.inf, dtype=float)
    graph_dist[sorted_indices[0]] = 0  # The root node has a graph distance of 0
    for rank in range(1, n_samples):
        idx_i = sorted_indices[rank]
        parent = nearest_higher[idx_i]
        if parent >= 0:
            graph_dist[idx_i] = graph_dist[parent] + np.linalg.norm(X[idx_i] - X[parent])

    return graph_dist


def select_centers(score, valid_mask, center_quantile, center_z):
    """Select centers from points that pass the density threshold."""
    if not np.all(np.isfinite(score)):
        raise ValueError("The center-selection score contains NaN or infinity.")

    eligible = np.flatnonzero(valid_mask)
    if eligible.size == 0:
        raise ValueError("No points satisfy the density threshold.")
    values = score[eligible]

    if center_quantile is None:
        threshold = values.mean() + center_z * values.std()
    else:
        threshold = np.quantile(values, center_quantile)

    cluster_centers = eligible[values > threshold]
    if cluster_centers.size == 0:
        cluster_centers = eligible[[np.argmax(values)]]
    return cluster_centers


def propagate_labels(sorted_indices, nearest_higher, cluster_centers, valid_mask):
    """Propagate center labels through nearest-higher-density links."""
    # Initialize all cluster labels as noise (-1)
    labels = np.full(sorted_indices.size, -1, dtype=int)
    labels[cluster_centers] = np.arange(cluster_centers.size)

    for idx_i in sorted_indices:
        if valid_mask[idx_i] and labels[idx_i] == -1:
            parent = nearest_higher[idx_i]
            if parent >= 0:
                labels[idx_i] = labels[parent]
    return labels


def ggdpc_dendrogram(X, parents):
    """
    Construct a single-linkage dendrogram from a GGDPC graph.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Original observations.

    parents : array-like, shape (n_samples,)
        GGDPC parent array. parents[i] is the endpoint of the directed
        edge i -> parents[i]. The graph root must have parent -1.

    Returns
    -------
    Z : ndarray, shape (n_samples - 1, 4)
        SciPy linkage matrix.
    """
    X = np.asarray(X, dtype=float)
    parents = np.asarray(parents, dtype=int)

    if X.ndim != 2:
        raise ValueError("X must have shape (n_samples, n_features).")

    n = X.shape[0]

    if parents.shape != (n,):
        raise ValueError("parents must have shape (n_samples,).")

    if np.any((parents < -1) | (parents >= n)):
        raise ValueError("Each parent must be -1 or a valid observation index.")

    if np.count_nonzero(parents == -1) != 1:
        raise ValueError("A GGDPC tree must have exactly one root.")

    children = np.flatnonzero(parents >= 0)

    if np.any(children == parents[children]):
        raise ValueError("A point cannot be its own parent.")

    from sklearn.metrics import pairwise_distances
    euclidean_distances = pairwise_distances(X)

    # Make nonedges strictly larger than every possible graph-edge distance.
    # The strict inequality avoids ambiguous ties between nonedges and the
    # longest graph edge.
    d_nonedge = np.nextafter(euclidean_distances.max(), np.inf)

    # D_ij = edge length when i and j are adjacent in the GGDPC graph;
    # otherwise D_ij = d_nonedge.
    D = np.full((n, n), d_nonedge, dtype=float)
    np.fill_diagonal(D, 0.0)

    edge_lengths = euclidean_distances[children, parents[children]]
    D[children, parents[children]] = edge_lengths
    D[parents[children], children] = edge_lengths

    # scipy.linkage expects the upper triangle in condensed form.
    from scipy.spatial.distance import squareform
    condensed_D = squareform(D, checks=False)

    from scipy.cluster.hierarchy import linkage
    Z = linkage(condensed_D, method="single", optimal_ordering=True)

    return Z
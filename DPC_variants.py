"""
@author: Yikun Zhang
Last Editing: Aug 9, 2026

Description: This script contains Python implementations of several Density Peak
Clustering (DPC) variants used as comparison methods for the GGDPC algorithm.
"""

from collections import deque
import warnings

import numpy as np


#=======================================================================================#

def _validate_input(X, n_clusters):
    """Validate the common inputs used by the DPC variants."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    if X.shape[0] < 2:
        raise ValueError("X must contain at least two observations.")
    if X.shape[1] < 1:
        raise ValueError("X must contain at least one feature.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values.")

    n_clusters = int(n_clusters)
    if not 1 <= n_clusters <= X.shape[0]:
        raise ValueError("n_clusters must be between 1 and n_samples.")
    return X, n_clusters


def _minmax_scale(X):
    """Scale each feature to [0, 1], leaving constant features at zero."""
    data_min = np.min(X, axis=0)
    data_range = np.max(X, axis=0) - data_min
    data_range[data_range == 0] = 1.0
    return (X - data_min) / data_range


def _nearest_higher_density(dist_mat, den_est):
    """Find the nearest earlier point after stable decreasing-density sorting."""
    n_samples = den_est.size
    sorted_indices = np.argsort(-den_est, kind="stable")
    delta = np.full(n_samples, np.inf)
    nearest_higher = np.full(n_samples, -1, dtype=int)

    highest = sorted_indices[0]
    delta[highest] = np.max(dist_mat[highest])
    for rank in range(1, n_samples):
        idx_i = sorted_indices[rank]
        higher_ind = sorted_indices[:rank]
        j = np.argmin(dist_mat[idx_i, higher_ind])
        delta[idx_i] = dist_mat[idx_i, higher_ind[j]]
        nearest_higher[idx_i] = higher_ind[j]
    return delta, nearest_higher, sorted_indices


def _assign_by_nearest_higher(dist_mat, sorted_indices, nearest_higher, cluster_centers):
    """Assign noncenters to their nearest higher-density ancestor."""
    n_samples = dist_mat.shape[0]
    labels = np.full(n_samples, -1, dtype=int)
    labels[cluster_centers] = np.arange(cluster_centers.size)

    for idx_i in sorted_indices:
        if labels[idx_i] == -1:
            parent = nearest_higher[idx_i]
            if parent != -1:
                labels[idx_i] = labels[parent]

    # This fallback is only active if the highest-density point was not among the
    # requested top decision values.
    unassigned = np.where(labels == -1)[0]
    if unassigned.size:
        labels[unassigned] = np.argmin(
            dist_mat[np.ix_(unassigned, cluster_centers)], axis=1)
    return labels


def _matlab_distance_percentile(dist_mat, percent):
    """Reproduce the 1-based rounded distance percentile used by the MATLAB codes."""
    upper = np.sort(dist_mat[np.triu_indices(dist_mat.shape[0], k=1)])
    position = int(np.floor(upper.size * percent / 100.0 + 0.5))
    position = min(max(position, 1), upper.size)
    return upper[position - 1]


#=======================================================================================#

def DPC_KNN_PCA(X, n_clusters, k=8, n_components=None, return_details=False):
    """
    Density Peak Clustering based on k-nearest neighbors and PCA (DPC-KNN-PCA).

    Parameters
    ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.
        n_clusters : int
            Number of clusters.
        k : int, optional
            Number of nearest neighbors used for local-density estimation.
            Default is 8, as used for the Aggregation dataset in Wang et al. (2025).
        n_components : int, optional
            Number of PCA components. If None, all available components are retained.
        return_details : bool, optional
            If True, also return a dictionary with intermediate quantities.

    Returns
    -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        details : dict, optional
            Contains the cluster centers, nearest-higher links, density, delta,
            decision values, PCA scores, and fitted PCA model.

    References
    ----------
        Du, M., Ding, S. and Jia, H. (2016). Study on density peaks clustering based
        on k-nearest neighbors and principal component analysis. Knowledge-Based
        Systems, 99, 135-145.
    """
    from sklearn.decomposition import PCA
    from sklearn.metrics import pairwise_distances

    X, n_clusters = _validate_input(X, n_clusters)
    n_samples, n_features = X.shape
    k = int(k)
    if not 1 <= k < n_samples:
        raise ValueError("k must be between 1 and n_samples - 1.")

    max_components = min(n_samples, n_features)
    if n_components is None:
        n_components = max_components
    n_components = int(n_components)
    if not 1 <= n_components <= max_components:
        raise ValueError("n_components must be between 1 and min(X.shape).")

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    dist_mat = pairwise_distances(X_pca)

    neighbor_indices = np.argsort(dist_mat, axis=1, kind="stable")[:, 1:k + 1]
    kth_distance = np.take_along_axis(dist_mat, neighbor_indices[:, -1:], axis=1).ravel()

    # This is the density definition in the authors' released MATLAB implementation.
    den_est = np.exp(-(kth_distance / k) ** 2)
    delta, nearest_higher, sorted_indices = _nearest_higher_density(dist_mat, den_est)
    gamma = den_est * delta
    cluster_centers = np.sort(np.argsort(gamma, kind="stable")[-n_clusters:])
    labels = _assign_by_nearest_higher(
        dist_mat, sorted_indices, nearest_higher, cluster_centers)

    if return_details:
        details = {
            "cluster_centers": cluster_centers,
            "1nn_higher": nearest_higher,
            "den_est": den_est,
            "delta": delta,
            "gamma": gamma,
            "neighbor_indices": neighbor_indices,
            "X_pca": X_pca,
            "pca": pca}
        return labels, details
    else:
        return labels


def SNN_DPC(X, n_clusters, k=4, scale=True, return_details=False):
    """
    Shared-nearest-neighbor-based Density Peak Clustering (SNN-DPC).

    Parameters
    ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.
        n_clusters : int
            Number of clusters.
        k : int, optional
            Neighborhood size. The released algorithm includes each point itself in
            this count. Default is 4 for the Aggregation dataset.
        scale : bool, optional
            Whether to apply feature-wise min-max scaling before clustering.
        return_details : bool, optional
            If True, also return a dictionary with intermediate quantities.

    Returns
    -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        details : dict, optional
            Contains the cluster centers, neighborhoods, shared-neighbor similarity,
            density, delta, and decision values.

    References
    ----------
        Liu, R., Wang, H. and Yu, X. (2018). Shared-nearest-neighbor-based clustering
        by fast search and find of density peaks. Information Sciences, 450, 200-226.
    """
    from sklearn.metrics import pairwise_distances

    X, n_clusters = _validate_input(X, n_clusters)
    n_samples = X.shape[0]
    k = int(k)
    if not 2 <= k <= n_samples:
        raise ValueError("k must be between 2 and n_samples.")

    X_work = _minmax_scale(X) if scale else X.copy()
    dist_mat = pairwise_distances(X_work)
    distance_order = np.argsort(dist_mat, axis=1, kind="stable")
    neighbor_indices = distance_order[:, :k]

    neighbor_mask = np.zeros((n_samples, n_samples), dtype=bool)
    neighbor_mask[np.arange(n_samples)[:, None], neighbor_indices] = True
    shared_count = neighbor_mask.astype(int) @ neighbor_mask.astype(int).T
    mutual_mask = neighbor_mask & neighbor_mask.T

    similarity = np.zeros((n_samples, n_samples), dtype=float)
    for idx_i in range(n_samples):
        for idx_j in np.flatnonzero(mutual_mask[idx_i, :idx_i]):
            shared = np.flatnonzero(neighbor_mask[idx_i] & neighbor_mask[idx_j])
            distance_sum = np.sum(dist_mat[idx_i, shared] + dist_mat[idx_j, shared])
            if distance_sum > 0:
                value = shared.size**2 / distance_sum
                similarity[idx_i, idx_j] = value
                similarity[idx_j, idx_i] = value

    den_est = np.sum(np.sort(similarity, axis=1)[:, -k:], axis=1)
    neighbor_distance_sum = np.sum(
        np.take_along_axis(dist_mat, neighbor_indices, axis=1), axis=1)
    sorted_indices = np.argsort(-den_est, kind="stable")

    delta = np.full(n_samples, np.inf)
    for rank in range(1, n_samples):
        idx_i = sorted_indices[rank]
        higher_ind = sorted_indices[:rank]
        modified_distance = dist_mat[idx_i, higher_ind] * (
            neighbor_distance_sum[idx_i] + neighbor_distance_sum[higher_ind])
        delta[idx_i] = np.min(modified_distance)
    delta[sorted_indices[0]] = np.max(delta[np.isfinite(delta)])

    gamma = den_est * delta
    cluster_centers = np.sort(np.argsort(gamma, kind="stable")[-n_clusters:])

    labels = np.full(n_samples, -1, dtype=int)
    labels[cluster_centers] = np.arange(n_clusters)

    # First assignment: breadth-first expansion through sufficiently shared
    # neighborhoods.
    queue = deque(cluster_centers.tolist())
    while queue:
        idx_i = queue.popleft()
        for idx_j in neighbor_indices[idx_i]:
            if labels[idx_j] == -1 and shared_count[idx_i, idx_j] >= k / 2:
                labels[idx_j] = labels[idx_i]
                queue.append(idx_j)

    # Second assignment: iteratively use the strongest labeled-neighbor vote.
    unassigned = np.where(labels == -1)[0]
    current_k = k
    while unassigned.size:
        vote_count = np.zeros((unassigned.size, n_clusters), dtype=int)
        for row, idx_i in enumerate(unassigned):
            current_neighbors = distance_order[idx_i, :current_k]
            assigned_neighbors = current_neighbors[labels[current_neighbors] != -1]
            if assigned_neighbors.size:
                vote_count[row] = np.bincount(
                    labels[assigned_neighbors], minlength=n_clusters)

        most_votes = np.max(vote_count)
        if most_votes > 0:
            selected = np.max(vote_count, axis=1) == most_votes
            # The reference MATLAB behavior chooses the last label when tied.
            reversed_choice = np.argmax(vote_count[selected, ::-1], axis=1)
            labels[unassigned[selected]] = n_clusters - 1 - reversed_choice
            unassigned = np.where(labels == -1)[0]
        elif current_k < n_samples:
            current_k += 1
        else:
            labels[unassigned] = np.argmin(
                dist_mat[np.ix_(unassigned, cluster_centers)], axis=1)
            break

    if return_details:
        details = {
            "cluster_centers": cluster_centers,
            "neighbor_indices": neighbor_indices,
            "shared_neighbor_count": shared_count,
            "similarity": similarity,
            "den_est": den_est,
            "delta": delta,
            "gamma": gamma,
            "X_transformed": X_work}
        return labels, details
    else:
        return labels


#=======================================================================================#

def _connectivity_search(idx_i, idx_j, dist_mat, dc, distance_threshold):
    """Connectivity search used by DPC-CE."""
    visited = np.zeros(dist_mat.shape[0], dtype=bool)
    visited[idx_i] = True
    frontier = np.array([idx_i], dtype=int)
    mean_neighbor_distance = 0.0
    find_time = 0

    while True:
        find_time += 1
        state = 1 if dist_mat[idx_i, idx_j] < 2 * dc else 0
        next_mask = np.zeros(dist_mat.shape[0], dtype=bool)

        for idx_k in frontier:
            neighbors = np.where(dist_mat[idx_k] < distance_threshold)[0]
            if neighbors.size:
                mean_neighbor_distance = max(
                    mean_neighbor_distance, np.mean(dist_mat[idx_k, neighbors]))
                next_mask[neighbors] = True

        next_mask[visited] = False
        next_frontier = np.where(next_mask)[0]

        if visited[idx_j]:
            state = 2
        elif next_frontier.size == 0:
            state = 3

        if state != 0:
            return state, np.where(visited)[0], find_time, mean_neighbor_distance

        visited[next_frontier] = True
        frontier = next_frontier


def DPC_CE(X, n_clusters, dc_percent=2.0, path_distance_ratio=0.25,
           distance_penalty_ratio=0.3, n_candidates=20, scale=True,
           return_details=False):
    """
    Density Peak Clustering with Connectivity Estimation (DPC-CE).

    Parameters
    ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.
        n_clusters : int
            Number of clusters. This replaces interactive rectangle selection in the
            released MATLAB code.
        dc_percent : float, optional
            Percentage of pairwise distances used to choose the cutoff distance.
        path_distance_ratio : float, optional
            Ratio used to construct the path-search distance threshold.
        distance_penalty_ratio : float, optional
            Penalty applied to long connecting paths.
        n_candidates : int, optional
            Number of large-delta candidate points examined for connectivity.
        scale : bool, optional
            Whether to apply the centering and global scaling used in the reference code.
        return_details : bool, optional
            If True, also return a dictionary with intermediate quantities.

    Returns
    -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        details : dict, optional
            Contains centers, nearest-higher links, density, corrected delta,
            decision values, cutoff distance, and connectivity-search records.

    References
    ----------
        Guo, W. et al. (2022). Density peak clustering with connectivity estimation.
        Knowledge-Based Systems, 243, 108501.
    """
    from sklearn.metrics import pairwise_distances

    X, n_clusters = _validate_input(X, n_clusters)
    n_samples = X.shape[0]
    if not 0 < dc_percent <= 100:
        raise ValueError("dc_percent must be in (0, 100].")
    if path_distance_ratio <= 0:
        raise ValueError("path_distance_ratio must be positive.")
    if distance_penalty_ratio < 0:
        raise ValueError("distance_penalty_ratio must be non-negative.")
    n_candidates = int(n_candidates)
    if n_candidates < 2:
        raise ValueError("n_candidates must be at least 2.")

    if scale:
        X_work = X - np.mean(X, axis=0)
        scale_value = np.max(np.abs(X_work))
        if scale_value > 0:
            X_work = X_work / scale_value
    else:
        X_work = X.copy()

    dist_mat = pairwise_distances(X_work)
    dc = _matlab_distance_percentile(dist_mat, dc_percent)
    if dc <= 0:
        raise ValueError("The cutoff distance is zero; remove duplicate points or increase dc_percent.")

    within_cutoff = (dist_mat < dc) & (~np.eye(n_samples, dtype=bool))
    den_est = np.sum(np.exp(-(dist_mat / dc)**2) * within_cutoff, axis=1)
    delta, nearest_higher, sorted_indices = _nearest_higher_density(dist_mat, den_est)

    n_candidates = min(n_candidates, n_samples)
    candidate_indices = np.argsort(-delta, kind="stable")[:n_candidates]
    candidate_indices = candidate_indices[
        np.argsort(-den_est[candidate_indices], kind="stable")]

    connectivity_records = []
    for rank in range(1, candidate_indices.size):
        idx_i = candidate_indices[rank]
        pair_targets = []
        if nearest_higher[idx_i] != -1:
            pair_targets.append(nearest_higher[idx_i])
        for previous in candidate_indices[:rank]:
            if previous not in pair_targets:
                pair_targets.append(previous)

        for pair_rank, idx_j in enumerate(pair_targets):
            original_distance = dist_mat[idx_i, idx_j]
            distance_threshold = original_distance * path_distance_ratio
            state, connected, find_time, neighbor_distance = _connectivity_search(
                idx_i, idx_j, dist_mat, dc, distance_threshold)

            if state == 2:
                penalty_time = find_time - 5
                corrected_distance = neighbor_distance + distance_threshold * (
                    1 + penalty_time * distance_penalty_ratio)
                dist_mat[idx_i, idx_j] = corrected_distance
                dist_mat[idx_j, idx_i] = corrected_distance
                if corrected_distance < delta[idx_i]:
                    delta[idx_i] = corrected_distance
                    nearest_higher[idx_i] = idx_j
            elif state == 3 and pair_rank == 0:
                corrected_distance = 1.1 * np.max(dist_mat[idx_j, connected])
                dist_mat[idx_i, idx_j] = corrected_distance
                dist_mat[idx_j, idx_i] = corrected_distance
                delta[idx_i] = corrected_distance

            connectivity_records.append({
                "pair": (idx_i, idx_j),
                "state": state,
                "path_length": find_time,
                "original_distance": original_distance,
                "corrected_distance": dist_mat[idx_i, idx_j]})

    delta[sorted_indices[0]] = np.max(delta)
    gamma = den_est * delta
    cluster_centers = np.sort(np.argsort(gamma, kind="stable")[-n_clusters:])
    labels = _assign_by_nearest_higher(
        dist_mat, sorted_indices, nearest_higher, cluster_centers)

    if return_details:
        details = {
            "cluster_centers": cluster_centers,
            "1nn_higher": nearest_higher,
            "den_est": den_est,
            "delta": delta,
            "gamma": gamma,
            "dc": dc,
            "candidate_indices": candidate_indices,
            "connectivity_records": connectivity_records,
            "X_transformed": X_work}
        return labels, details
    else:
        return labels


#=======================================================================================#

def DPC_DLP(X, n_clusters, neighbor_fraction=0.009, mu=2.0, sigma_scale=0.4,
            n_graph_neighbors=10, alpha=1.0, regularization=0.0,
            n_iterations=8, scale=True, return_details=False):
    """
    Density Peak Clustering with Dynamic Label Propagation (DPC-DLP).

    Parameters
    ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.
        n_clusters : int
            Number of clusters.
        neighbor_fraction : float, optional
            Fraction p used to set k=floor(p*n). Default is 0.009 (0.9%) for
            the Aggregation dataset.
        mu : float, optional
            Scale parameter of the adaptive Gaussian graph.
        sigma_scale : float, optional
            Scale used to construct pair-specific Gaussian bandwidths.
        n_graph_neighbors : int, optional
            Number of neighbors used to construct the propagation graph.
        alpha : float, optional
            Strength of the label-correlation term.
        regularization : float, optional
            Diagonal regularization added during transition updates.
        n_iterations : int, optional
            Number of dynamic label-propagation iterations.
        scale : bool, optional
            Whether to apply feature-wise min-max scaling.
        return_details : bool, optional
            If True, also return a dictionary with intermediate quantities.

    Returns
    -------
        labels : ndarray of shape (n_samples,)
            Cluster labels.
        details : dict, optional
            Contains centers, density, delta, decision values, neighborhoods,
            propagated label scores, and the final transition matrix.

    References
    ----------
        Seyedi, S. A. et al. (2019). Dynamic graph-based label propagation for
        density peaks clustering. Expert Systems with Applications, 115, 314-328.
    """
    from scipy.sparse import csr_matrix
    from sklearn.metrics import pairwise_distances

    X, n_clusters = _validate_input(X, n_clusters)
    n_samples = X.shape[0]
    if neighbor_fraction <= 0:
        raise ValueError("neighbor_fraction must be positive.")
    if mu <= 0 or sigma_scale <= 0:
        raise ValueError("mu and sigma_scale must be positive.")
    n_graph_neighbors = int(n_graph_neighbors)
    n_iterations = int(n_iterations)
    if n_graph_neighbors < 1:
        raise ValueError("n_graph_neighbors must be positive.")
    if n_iterations < 1:
        raise ValueError("n_iterations must be positive.")

    k = int(np.floor(neighbor_fraction * n_samples))
    if k < 1:
        raise ValueError("neighbor_fraction is too small to select one neighbor.")
    k = min(k, n_samples - 1)
    seed_neighbor_count = min(max(k, 10), n_samples - 1)
    graph_neighbor_count = min(n_graph_neighbors, seed_neighbor_count)

    X_work = _minmax_scale(X) if scale else X.copy()
    dist_mat = pairwise_distances(X_work)
    distance_order = np.argsort(dist_mat, axis=1, kind="stable")
    neighbor_indices = distance_order[:, 1:seed_neighbor_count + 1]
    neighbor_distances = np.take_along_axis(dist_mat, neighbor_indices, axis=1)

    den_est = np.exp(-np.sum(neighbor_distances[:, :k]**2, axis=1) / k)
    delta, nearest_higher, _ = _nearest_higher_density(dist_mat, den_est)
    gamma = den_est * delta
    cluster_centers = np.argsort(-gamma, kind="stable")[:n_clusters]

    initial_scores = np.zeros((n_samples, n_clusters), dtype=float)
    labeled_mask = np.zeros(n_samples, dtype=bool)
    for cluster_id, center in enumerate(cluster_centers):
        initial_scores[center, cluster_id] = 1.0
        labeled_mask[center] = True
        initial_scores[neighbor_indices[center], cluster_id] = 1.0
        labeled_mask[neighbor_indices[center]] = True

    sigma_neighbor_count = min(6, n_samples)
    local_distance_sum = np.sum(
        np.take_along_axis(
            dist_mat, distance_order[:, :sigma_neighbor_count], axis=1), axis=1)
    sigma = sigma_scale * (
        local_distance_sum[:, None] + local_distance_sum[None, :]) / 10.0
    denominator = mu * np.maximum(sigma**2, np.finfo(float).eps)
    weight_mat = np.exp(-(dist_mat**2) / denominator)
    np.fill_diagonal(weight_mat, 0.0)

    rows = np.repeat(np.arange(n_samples), graph_neighbor_count)
    columns = neighbor_indices[:, :graph_neighbor_count].ravel()
    values = weight_mat[rows, columns]
    row_sums = np.bincount(rows, weights=values, minlength=n_samples)
    nonzero = row_sums[rows] > 0
    values[nonzero] /= row_sums[rows[nonzero]]
    transition_base = csr_matrix(
        (values, (rows, columns)), shape=(n_samples, n_samples))

    transition = transition_base.toarray()
    label_scores = initial_scores.copy()
    for _ in range(n_iterations):
        updated_scores = transition @ label_scores
        updated_scores[labeled_mask] = initial_scores[labeled_mask]

        label_correlation = alpha * (label_scores @ label_scores.T)
        intermediate = transition_base @ (transition + label_correlation)
        transition = (transition_base @ intermediate.T).T
        if regularization:
            transition.flat[::n_samples + 1] += regularization
        label_scores = updated_scores

    labels = np.argmax(label_scores, axis=1)

    if return_details:
        details = {
            "cluster_centers": cluster_centers,
            "1nn_higher": nearest_higher,
            "den_est": den_est,
            "delta": delta,
            "gamma": gamma,
            "neighbor_indices": neighbor_indices,
            "label_scores": label_scores,
            "transition_matrix": transition,
            "X_transformed": X_work,
            "k": k}
        return labels, details
    else:
        return labels


#=======================================================================================#

def _natural_neighbor_search(neighbor_order):
    """Find natural neighborhoods using the stable-search stopping rule."""
    n_samples = neighbor_order.shape[0]
    forward_neighbors = [[] for _ in range(n_samples)]
    reverse_neighbors = [[] for _ in range(n_samples)]
    reverse_count = np.zeros(n_samples, dtype=int)

    rank = 1
    previous_missing = 0
    stable_rounds = 0
    natural_neighbors = [[] for _ in range(n_samples)]

    while rank < n_samples:
        for idx_i in range(n_samples):
            idx_j = int(neighbor_order[idx_i, rank])
            forward_neighbors[idx_i].append(idx_j)
            reverse_neighbors[idx_j].append(idx_i)
            reverse_count[idx_j] += 1

        natural_neighbors = [
            list(set(forward_neighbors[idx_i]) & set(reverse_neighbors[idx_i]))
            for idx_i in range(n_samples)]
        missing_count = int(np.sum(reverse_count == 0))

        if missing_count == previous_missing:
            stable_rounds += 1
        else:
            stable_rounds = 1
        rank += 1

        if missing_count == 0 or stable_rounds >= 2:
            break
        previous_missing = missing_count

    return rank - 1, forward_neighbors, reverse_neighbors, natural_neighbors


def _mdnn_density(dist_mat, natural_neighbors, forward_neighbors):
    """Compute the shared-natural-neighbor density used by DPC-MDNN."""
    n_samples = dist_mat.shape[0]
    den_est = np.zeros(n_samples)
    forward_sets = [set(neighbors) for neighbors in forward_neighbors]

    for idx_i in range(n_samples):
        for idx_j in natural_neighbors[idx_i]:
            shared = np.fromiter(
                forward_sets[idx_i] & forward_sets[idx_j], dtype=int)
            if shared.size:
                denominator = (
                    np.sum(dist_mat[idx_i, shared]) +
                    np.sum(dist_mat[idx_j, shared])) * (dist_mat[idx_i, idx_j] + 0.01)
                if denominator > 0:
                    den_est[idx_i] += shared.size**2 / denominator
    return den_est


def _mdnn_neighbor_children(cores, representative, natural_neighbors):
    """Construct each core's represented points and adjacent natural neighbors."""
    children = []
    for core in cores:
        represented = np.where(representative == core)[0]
        adjacent = set(represented.tolist())
        for idx_i in represented:
            adjacent.update(natural_neighbors[idx_i])
        children.append(sorted(adjacent))
    return children


def _mdnn_representatives(forward_neighbors, den_est, natural_neighbors, dist_mat):
    """Select and consolidate representative points for DPC-MDNN."""
    n_samples = den_est.size
    representative_votes = [[] for _ in range(n_samples)]

    for idx_i in np.argsort(den_est, kind="stable"):
        neighborhood = np.asarray(forward_neighbors[idx_i], dtype=int)
        representative = neighborhood[np.argmax(den_est[neighborhood])]
        for idx_j in neighborhood:
            representative_votes[idx_j].append(int(representative))

    representative = np.full(n_samples, -1, dtype=int)
    for idx_i, votes in enumerate(representative_votes):
        candidates, counts = np.unique(votes, return_counts=True)
        max_count = np.max(counts)
        tied = candidates[counts == max_count]
        tie_score = dist_mat[idx_i, tied] * np.abs(den_est[idx_i] - den_est[tied])
        representative[idx_i] = tied[np.argmin(tie_score)]

    # Collapse representative chains to their fixed points.
    for idx_i in range(n_samples):
        path = []
        current = idx_i
        visited = set()
        while representative[current] != current and current not in visited:
            visited.add(current)
            path.append(current)
            current = representative[current]
        if current in visited:
            cycle = path[path.index(current):]
            current = cycle[np.argmax(den_est[cycle])]
            representative[current] = current
        representative[path] = current

    cores = np.where(representative == np.arange(n_samples))[0].tolist()
    children = _mdnn_neighbor_children(cores, representative, natural_neighbors)
    original_position = {core: position for position, core in enumerate(cores)}
    delete_cores = set()

    for idx_i in sorted(cores, key=lambda index: den_est[index]):
        if idx_i in delete_cores:
            continue
        for idx_j in sorted(cores, key=lambda index: den_est[index]):
            if idx_i in delete_cores:
                break
            if idx_j in delete_cores or idx_i == idx_j:
                continue

            pos_i = original_position[idx_i]
            pos_j = original_position[idx_j]
            if idx_j in children[pos_i] or idx_i in children[pos_j]:
                if den_est[idx_i] < den_est[idx_j]:
                    representative[representative == idx_i] = idx_j
                    cores.remove(idx_i)
                    delete_cores.add(idx_i)
                    children[pos_i] = sorted(set(children[pos_i]) | set(children[pos_j]))
                    break
                elif den_est[idx_i] > den_est[idx_j]:
                    representative[representative == idx_j] = idx_i
                    cores.remove(idx_j)
                    delete_cores.add(idx_j)
                    children[pos_j] = sorted(set(children[pos_i]) | set(children[pos_j]))

    contained_points = [np.where(representative == core)[0] for core in cores]
    return representative, cores, contained_points


def _mdnn_similarity(cores, den_est, neighbor_children, dist_mat, representative):
    """Compute the candidate-core dissimilarity matrix for DPC-MDNN."""
    n_cores = len(cores)
    similarity = np.zeros((n_cores, n_cores))
    if n_cores < 2:
        return similarity

    core_distance = dist_mat[np.ix_(cores, cores)]
    max_distance = np.max(core_distance)
    max_penalty = 1.0

    for idx_i in range(n_cores):
        for idx_j in range(idx_i + 1, n_cores):
            common = set(neighbor_children[idx_i]) & set(neighbor_children[idx_j])
            if not common:
                continue

            represented_i = np.where(representative == cores[idx_i])[0]
            represented_j = np.where(representative == cores[idx_j])[0]
            mean_i = np.mean(den_est[represented_i])
            mean_j = np.mean(den_est[represented_j])
            denominator = len(common)**2 * 2 * np.sqrt(mean_i * mean_j)
            if denominator <= 0:
                value = np.inf
            else:
                value = (
                    dist_mat[cores[idx_i], cores[idx_j]] *
                    max(len(neighbor_children[idx_i]), len(neighbor_children[idx_j])) *
                    np.abs(mean_i + mean_j) / denominator)
            similarity[idx_i, idx_j] = value
            similarity[idx_j, idx_i] = value
            max_penalty = max(
                max_penalty,
                len(neighbor_children[idx_i]) * len(neighbor_children[idx_j]) /
                len(common)**2)

    for idx_i in range(n_cores):
        for idx_j in range(idx_i + 1, n_cores):
            if not (set(neighbor_children[idx_i]) & set(neighbor_children[idx_j])):
                value = max_penalty * dist_mat[cores[idx_i], cores[idx_j]] * max_distance
                similarity[idx_i, idx_j] = value
                similarity[idx_j, idx_i] = value
    return similarity


def _single_link_microclusters(similarity, contained_points, n_clusters,
                               min_cluster_size, n_samples):
    """Merge candidate cores with the single-link strategy in the released code."""
    clusters = [[idx_i] for idx_i in range(similarity.shape[0])]
    sizes = [points.size for points in contained_points]

    while len(clusters) > n_clusters:
        best_pair = (0, 1)
        best_distance = np.min(similarity[np.ix_(clusters[0], clusters[1])])
        for idx_i in range(len(clusters)):
            for idx_j in range(idx_i + 1, len(clusters)):
                value = np.min(similarity[np.ix_(clusters[idx_i], clusters[idx_j])])
                if value < best_distance:
                    best_distance = value
                    best_pair = (idx_i, idx_j)

        idx_i, idx_j = best_pair
        merged = clusters[idx_i] + clusters[idx_j]
        merged_size = sizes[idx_i] + sizes[idx_j]
        del clusters[idx_j]
        del clusters[idx_i]
        del sizes[idx_j]
        del sizes[idx_i]
        clusters.append(merged)
        sizes.append(merged_size)

    retained = [
        cluster for cluster, size in zip(clusters, sizes)
        if size >= min_cluster_size * n_samples]
    return retained


def DPC_MDNN(X, n_clusters, manifold_neighbors=5, min_cluster_size=0.01,
             scale=True, return_details=False):
    """
    Density Peak Clustering based on Manifold Distance and Natural Neighbors.

    Parameters
    ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.
        n_clusters : int
            Number of final clusters.
        manifold_neighbors : int, optional
            Number of neighbors used to construct the Isomap manifold graph.
        min_cluster_size : float, optional
            Final clusters smaller than this fraction of the data are labeled noise.
        scale : bool, optional
            Whether to apply feature-wise min-max scaling.
        return_details : bool, optional
            If True, also return a dictionary with intermediate quantities.

    Returns
    -------
        labels : ndarray of shape (n_samples,)
            Cluster labels. Removed small clusters are labeled as -1.
        details : dict, optional
            Contains final centers, representative points, natural neighborhoods,
            density, manifold distances, and candidate-core similarity.

    References
    ----------
        Wang, H. et al. (2025). Improved density peak clustering with a flexible
        manifold distance and natural nearest neighbors for network intrusion
        detection. Scientific Reports, 15, Article 92509.
    """
    from scipy.sparse import SparseEfficiencyWarning
    from sklearn.manifold import Isomap

    X, n_clusters = _validate_input(X, n_clusters)
    n_samples, n_features = X.shape
    manifold_neighbors = int(manifold_neighbors)
    if not 1 <= manifold_neighbors < n_samples:
        raise ValueError("manifold_neighbors must be between 1 and n_samples - 1.")
    if not 0 <= min_cluster_size < 1:
        raise ValueError("min_cluster_size must be in [0, 1).")

    X_work = _minmax_scale(X) if scale else X.copy()
    n_components = min(2, n_features)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=SparseEfficiencyWarning)
        isomap = Isomap(
            n_components=n_components,
            n_neighbors=manifold_neighbors,
            path_method="auto")
        isomap.fit(X_work)
    manifold_distance = np.asarray(isomap.dist_matrix_)

    neighbor_order = np.argsort(manifold_distance, axis=1, kind="stable")
    natural_k, forward_neighbors, reverse_neighbors, natural_neighbors = (
        _natural_neighbor_search(neighbor_order))
    for idx_i in range(n_samples):
        forward_neighbors[idx_i].append(idx_i)

    den_est = _mdnn_density(
        manifold_distance, natural_neighbors, forward_neighbors)
    representative, cores, contained_points = _mdnn_representatives(
        forward_neighbors, den_est, natural_neighbors, manifold_distance)
    if not cores:
        raise RuntimeError("DPC-MDNN did not identify any candidate cores.")

    neighbor_children = _mdnn_neighbor_children(
        cores, representative, natural_neighbors)
    similarity = _mdnn_similarity(
        cores, den_est, neighbor_children, manifold_distance, representative)
    effective_clusters = min(n_clusters, len(cores))
    merged_core_clusters = _single_link_microclusters(
        similarity, contained_points, effective_clusters,
        min_cluster_size, n_samples)

    core_labels = np.full(len(cores), -1, dtype=int)
    cluster_centers = []
    for cluster_id, core_cluster in enumerate(merged_core_clusters):
        core_labels[core_cluster] = cluster_id
        member_cores = np.asarray(cores, dtype=int)[core_cluster]
        cluster_centers.append(member_cores[np.argmax(den_est[member_cores])])

    core_position = {core: position for position, core in enumerate(cores)}
    labels = np.array([
        core_labels[core_position[representative[idx_i]]]
        for idx_i in range(n_samples)], dtype=int)
    cluster_centers = np.asarray(cluster_centers, dtype=int)

    if return_details:
        details = {
            "cluster_centers": cluster_centers,
            "candidate_cores": np.asarray(cores, dtype=int),
            "representative": representative,
            "contained_points": contained_points,
            "natural_neighbors": natural_neighbors,
            "natural_neighbor_k": natural_k,
            "den_est": den_est,
            "manifold_distance": manifold_distance,
            "core_similarity": similarity,
            "X_transformed": X_work}
        return labels, details
    else:
        return labels

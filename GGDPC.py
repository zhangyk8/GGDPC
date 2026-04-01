"""
@author: Yikun Zhang
Last Editing: Mar 18, 2026

Description: This script contains the main functions for implementing the original Density 
Peak Clustering (DPC) and Gradient-Guided Density Peak Clustering (GGDPC) algorithm.
"""

import numpy as np
from utils import gaussian_kde, gaussian_ms_onestep

#=======================================================================================#

def DPC(X, den_est=None, dist_mat=None, den_thres=0, center_quantile=None, return_details=False):
    """
    Density Peak Clustering (DPC) algorithm.

    Parameters
    ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.
        den_est : ndarray of shape (n_samples,), optional
            Pre-computed density estimates for each point. If None, it will be computed 
            using KDE with a default bandwidth and Gaussian kernel.
        dist_mat: ndarray of shape (n_samples, n_samples), optional
            Pre-computed distance matrix. If None, it will be computed using Euclidean distance.
        den_thres : float, optional
            Density quantile threshold for classifying low-density points as noise. 
            Default is 0 (no points are considered noise).
        center_quantile : float, optional
            Quantile for identifying cluster centers. If None, the cluster centers will be
            detected via z-score normalization of the product of density and distance to the
            nearest higher-density point.
        return_details : bool, optional
            If True, also return a dictionary with additional details about the clustering.
    
    Returns
    -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point. Noise points are labeled as -1.
        details : dict, optional
            Additional details about the clustering, returned if return_details is True. 
            Contains keys:
                - "cluster_centers": Indices of identified cluster centers.
                - "1nn_higher": Index of the nearest higher-density point for each point.
                - "den_est": Density estimates for each point.
                - "delta": Distance to nearest higher-density point for each point.
                - "den_thres": The density threshold used for noise classification.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    n_samples = X.shape[0]

    if not 0 <= den_thres <= 1:
        raise ValueError("den_thres must be in [0, 1].")
    if center_quantile is not None and not 0 <= center_quantile <= 1:
        raise ValueError("center_quantile must be in [0, 1].")
    
    # Validate dist_mat if provided
    if dist_mat is not None:
        dist_mat = np.asarray(dist_mat)
        if dist_mat.shape != (n_samples, n_samples):
            raise ValueError("dist_mat must have shape (n_samples, n_samples)")

    # Compute density estimates if not provided
    if den_est is None:
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(kernel='gaussian', bandwidth='silverman').fit(X)
        den_est = np.exp(kde.score_samples(X))
    elif isinstance(den_est, str) and den_est == "cutoff":
        if dist_mat is None:
            from sklearn.metrics import pairwise_distances
            dist_mat = pairwise_distances(X)
        upper = dist_mat[np.triu_indices(n_samples, k=1)]
        dc = np.quantile(upper, 0.3)
        den_est = np.sum(dist_mat < dc, axis=1).astype(float) - 1.0
    else:
        den_est = np.asarray(den_est)
        if den_est.shape != (n_samples,):
            raise ValueError("den_est must have shape (n_samples,)")

    # Sort points by density in descending order
    sorted_indices = np.argsort(-den_est)
    # Density threshold mask
    density_cutoff = np.quantile(den_est, den_thres)
    valid_mask = den_est >= density_cutoff

    # Find the nearest neighbor with higher density
    delta = np.full(n_samples, np.inf)
    nearest_higher_density = np.full(n_samples, -1, dtype=int)
    # Assign the maximum distance to the point with the highest density value
    highest = sorted_indices[0]
    if dist_mat is not None:
        delta[highest] = np.max(dist_mat[highest]) + 1e-3

        for i in range(1, n_samples):
            idx_i = sorted_indices[i]
            higher_ind = sorted_indices[:i]
            # Compute distances to points with higher density
            dists = dist_mat[idx_i, higher_ind]
            j = np.argmin(dists)
            delta[idx_i] = dists[j]
            nearest_higher_density[idx_i] = higher_ind[j]
    else:
        delta[highest] = np.max(np.linalg.norm(X - X[highest], axis=1)) + 1e-3
        for i in range(1, n_samples):
            idx_i = sorted_indices[i]
            higher_ind = sorted_indices[:i]
            # Compute distances to points with higher density
            dists = np.linalg.norm(X[idx_i] - X[higher_ind], axis=1)
            j = np.argmin(dists)
            delta[idx_i] = dists[j]
            nearest_higher_density[idx_i] = higher_ind[j]

    # Identify cluster centers based on center_quantile or z-score of (density * delta)
    gamma = den_est * delta
    if center_quantile is not None:
        center_mask = gamma > np.quantile(gamma, center_quantile)
    else:
        # gamma_std = np.std(gamma)
        # if gamma_std == 0:
        #     center_mask = np.zeros(n_samples, dtype=bool)
        #     center_mask[np.argmax(gamma)] = True
        # else:
        #     z_scores = (gamma - np.mean(gamma)) / gamma_std
        #     center_mask = z_scores > 3  # Example threshold for z-score
        from sklearn.linear_model import LinearRegression
        log_den = np.log(den_est + 1e-12)
        log_delta = np.log(delta + 1e-12)
        X_reg = log_den.reshape(-1, 1)
        y_reg = log_delta
        reg = LinearRegression().fit(X_reg, y_reg)
        delta_pred = reg.predict(X_reg)
        residuals = log_delta - delta_pred
        residual_std = np.std(residuals)
        if residual_std == 0:
            center_mask = np.zeros(n_samples, dtype=bool)
            center_mask[np.argmax(gamma)] = True
        else:
            z_scores = residuals / residual_std
            center_mask = z_scores > 3  # Example threshold for z-score

    cluster_centers = np.where(center_mask)[0]
    if cluster_centers.size == 0:
        cluster_centers = np.array([np.argmax(gamma)])

    # Initialize all cluster labels as noise (-1)
    labels = np.full(n_samples, -1, dtype=int)
    # Assign unique labels to cluster centers
    labels[cluster_centers] = np.arange(len(cluster_centers))

    for idx_i in sorted_indices:
        if not valid_mask[idx_i]:
            continue
        if labels[idx_i] == -1:
            parent = nearest_higher_density[idx_i]
            if parent != -1:
                labels[idx_i] = labels[parent]

    if return_details:
        details = {
            "cluster_centers": cluster_centers,
            "1nn_higher": nearest_higher_density,
            "den_est": den_est, 
            "delta": delta,
            "den_thres": den_thres}
        return labels, details
    else:
        return labels


def GGDPC(X, den_est=None, grad_new=None, den_thres=0, center_quantile=None, return_details=False):
    """
    Gradient-Guided Density Peak Clustering (GGDPC) algorithm.

    Parameters
    ----------
        X : ndarray of shape (n_samples, n_features)
            Input data points.
        den_est : ndarray of shape (n_samples,), optional
            Pre-computed density estimates for each point. If None, it will be computed 
            using KDE with a default bandwidth and Gaussian kernel. If set to "cutoff", it will use 
            a simple cutoff-based density estimation based on the number of neighbors within a certain distance.
        grad_new : ndarray of shape (n_samples, n_features), optional
            Pre-computed gradient estimates for each point. If None, it will be computed using a one-step 
            mean shift update based on the KDE density estimates.
        den_thres : float, optional
            Density quantile threshold for classifying low-density points as noise. 
            Default is 0 (no points are considered noise).
        center_quantile : float, optional
            Quantile for identifying cluster centers. If None, the cluster centers will be
            detected via z-score normalization of the product of density and distance to the nearest 
            higher-density point.
        return_details : bool, optional
            If True, also return a dictionary with additional details about the clustering.
    
    Returns
    -------
        labels : ndarray of shape (n_samples,)
            Cluster labels for each point. Noise points are labeled as -1.
        details : dict, optional
            Additional details about the clustering, returned if return_details is True. 
            Contains keys:
                - "cluster_centers": Indices of identified cluster centers.
                - "gg1nn_higher": Index of the nearest higher-density point for each point based on the gradient-guided update.
                - "den_est": Density estimates for each point.
                - "grad_new": One-step gradient-guided update locations for each point.
                - "delta": Distance to nearest higher-density point for each point.
                - "den_thres": The density threshold used for noise classification.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array of shape (n_samples, n_features).")
    n_samples, n_features = X.shape
    
    if not 0 <= den_thres <= 1:
        raise ValueError("den_thres must be in [0, 1].")
    if center_quantile is not None and not 0 <= center_quantile <= 1:
        raise ValueError("center_quantile must be in [0, 1].")
    
    # Compute density estimates if not provided
    if den_est is None:
        den_est = gaussian_kde(X, X)
        if grad_new is None:
            grad_new = gaussian_ms_onestep(X, X)
    elif isinstance(den_est, str) and den_est == "cutoff":
        from sklearn.metrics import pairwise_distances
        dist_mat = pairwise_distances(X)
        upper = dist_mat[np.triu_indices(n_samples, k=1)]
        dc = np.quantile(upper, 0.3)
        den_est = np.sum(dist_mat < dc, axis=1).astype(float) - 1.0
        if grad_new is None:
            # Find the k nearest neighbors with higher densities for each point in X
            # and compute their average locations. If this set is null, then set the location as the current point 
            k_higher = np.ceil(np.log(n_samples)).astype(int)
            grad_new = np.zeros_like(X)
            sorted_indices = np.argsort(-den_est)
            for rank, idx_i in enumerate(sorted_indices):
                higher_ind = sorted_indices[:rank]
                if higher_ind.size == 0:
                    grad_new[idx_i] = X[idx_i]
                    continue
                dists = dist_mat[idx_i, higher_ind]
                k_eff = min(k_higher, higher_ind.size)
                knn_pos = np.argpartition(dists, kth=k_eff-1)[:k_eff]
                knn_idx = higher_ind[knn_pos]
                knn_dists = dists[knn_pos]
                weights = 1.0 / np.maximum(knn_dists, 1e-12)
                grad_new[idx_i] = np.average(X[knn_idx], axis=0, weights=weights)
    else:
        den_est = np.asarray(den_est)
        if den_est.shape != (n_samples,):
            raise ValueError("den_est must have shape (n_samples,)")
        if grad_new is None:
            grad_new = gaussian_ms_onestep(X, X)
    
    grad_new = np.asarray(grad_new, dtype=float)
    if grad_new.shape != X.shape:
        raise ValueError("grad_new must have shape (n_samples, n_features)")
    
    # Sort points by density in descending order
    sorted_indices = np.argsort(-den_est)
    # Density threshold mask
    density_cutoff = np.quantile(den_est, den_thres)
    valid_mask = den_est >= density_cutoff
    
    # Find the nearest neighbor with higher density among "grad_new" for each point
    delta = np.full(n_samples, np.inf)
    nearest_higher_density = np.full(n_samples, -1, dtype=int)
    # Assign the maximum distance to the point with the highest density value
    highest = sorted_indices[0]
    delta[highest] = np.max(np.linalg.norm(X - X[highest], axis=1)) + 1e-3
    for i in range(1, n_samples):
        idx_i = sorted_indices[i]
        higher_ind = sorted_indices[:i]
        # Compute distances to points with higher density
        dists = np.linalg.norm(grad_new[idx_i] - X[higher_ind], axis=1)
        j = np.argmin(dists)
        delta[idx_i] = np.linalg.norm(X[idx_i] - X[higher_ind][j])
        nearest_higher_density[idx_i] = higher_ind[j]

    # Identify cluster centers based on center_quantile or z-score of (density * delta)
    gamma = den_est * delta
    if center_quantile is not None:
        center_mask = gamma > np.quantile(gamma, center_quantile)
    else:
        # gamma_std = np.std(gamma)
        # if gamma_std == 0:
        #     center_mask = np.zeros(n_samples, dtype=bool)
        #     center_mask[np.argmax(gamma)] = True
        # else:
        #     z_scores = (gamma - np.mean(gamma)) / gamma_std
        #     center_mask = z_scores > 3  # Example threshold for z-score
        from sklearn.linear_model import LinearRegression
        log_den = np.log(den_est + 1e-12)
        log_delta = np.log(delta + 1e-12)
        X_reg = log_den.reshape(-1, 1)
        y_reg = log_delta
        reg = LinearRegression().fit(X_reg, y_reg)
        delta_pred = reg.predict(X_reg)
        residuals = log_delta - delta_pred
        residual_std = np.std(residuals)
        if residual_std == 0:
            center_mask = np.zeros(n_samples, dtype=bool)
            center_mask[np.argmax(gamma)] = True
        else:
            z_scores = residuals / residual_std
            center_mask = z_scores > 3  # Example threshold for z-score


    cluster_centers = np.where(center_mask)[0]
    if cluster_centers.size == 0:
        cluster_centers = np.array([np.argmax(gamma)])

    # Initialize all cluster labels as noise (-1)
    labels = np.full(n_samples, -1, dtype=int)
    # Assign unique labels to cluster centers
    labels[cluster_centers] = np.arange(len(cluster_centers))

    for idx_i in sorted_indices:
        if not valid_mask[idx_i]:
            continue
        if labels[idx_i] == -1:
            parent = nearest_higher_density[idx_i]
            if parent != -1:
                labels[idx_i] = labels[parent]

    if return_details:
        details = {
            "cluster_centers": cluster_centers,
            "gg1nn_higher": nearest_higher_density,
            "den_est": den_est,
            "grad_new": grad_new, 
            "delta": delta,
            "den_thres": den_thres}
        return labels, details
    else:
        return labels
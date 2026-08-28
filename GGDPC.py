"""
@author: Yikun Zhang
Last Editing: Aug 9, 2026

Description: This script contains the main functions for implementing the original Density 
Peak Clustering (DPC) and Gradient-Guided Density Peak Clustering (GGDPC) algorithm.
"""

import numpy as np

from dpc_helpers import (
    cutoff_density,
    ggdpc_dendrogram,
    ggdpc_graph_distances,
    gradient_guided_nearest_higher,
    nearest_higher_density,
    prepare_coordinates,
    propagate_labels,
    select_centers,
    resolve_bandwidth,
    validate_common_inputs,
    validate_density,
    validate_distance_matrix,
    validate_update_locations,
)
from utils import gaussian_kde, gaussian_kde_ms_onestep, gaussian_ms_onestep


# ===================================================================================== #

def DPC(X, den_est=None, dist_mat=None, bandwidth=None, den_thres=0, center_quantile=None, scaling=False, 
        cutoff_quantile=0.3, center_z=3.0, return_details=True):
    """
    Density Peak Clustering (DPC) algorithm.

    Parameters
    ----------
        X : ndarray of shape (n_samples, n_features)
            Input observations.
        den_est : ndarray of shape (n_samples,)
            Pre-computed density estimates for each point. If None, it will be computed 
            using KDE with a default bandwidth and Gaussian kernel. If set to "cutoff", it will use 
            a simple cutoff-based density estimation based on the number of neighbors within a certain distance.
        dist_mat: ndarray of shape (n_samples, n_samples)
            Pre-computed distance matrix. If None, it will be computed using Euclidean distance.
        den_thres : float
            Density quantile threshold for classifying low-density points as noise. 
            Default is 0 (no points are considered noise).
        center_quantile : float
            Quantile for identifying cluster centers. If None, the cluster centers will be
            detected via z-score normalization of the 1NN uphill distance to the
            nearest higher-density point.
        scaling : bool
            If True, the input data will be standardized before computing the kernel density estimation. 
            Default is False.
        cutoff_quantile : float
            Quantile for determining the cutoff distance in the cutoff-based density estimation. 
            Default is 0.3 (30th percentile).
        center_z : float
            Z-score threshold for identifying cluster centers based on the 1NN uphill distance. 
            Default is 3.0.
        return_details : bool
            If True, also return a dictionary with additional details about the clustering. Default is True.
    
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
                - "density_cutoff": The actual density cutoff value used for noise classification.
                - "cutoff_distance": The cutoff distance used in the cutoff-based density estimation (if applicable).
    """
    X, den_thres, center_quantile, cutoff_quantile, center_z = validate_common_inputs(X, den_thres, center_quantile,
                                                                                      scaling, cutoff_quantile, center_z)
    n_samples = X.shape[0]

    if dist_mat is not None:
        dist_mat = validate_distance_matrix(dist_mat, n_samples)

    X_work, scaler = prepare_coordinates(X, scaling=scaling)
    cutoff_distance = None
    # Compute density estimates if not provided
    if den_est is None:
        from sklearn.neighbors import KernelDensity
        h = resolve_bandwidth(X_work, bandwidth)
        kde = KernelDensity(kernel="gaussian", bandwidth=h).fit(X_work)
        den_est = np.exp(kde.score_samples(X_work))
    elif isinstance(den_est, str):
        if den_est != "cutoff":
            raise ValueError('The only supported string value for den_est is "cutoff".')
        if dist_mat is None:
            from sklearn.metrics import pairwise_distances
            dist_mat = pairwise_distances(X_work)
        den_est, cutoff_distance = cutoff_density(dist_mat, cutoff_quantile)
    else:
        den_est = validate_density(den_est, n_samples)

    den_est = validate_density(den_est, n_samples)
    # Density threshold mask
    density_cutoff = np.quantile(den_est, den_thres)
    valid_mask = den_est >= density_cutoff

    delta, nearest_higher, sorted_indices = nearest_higher_density(X, den_est, dist_mat)
    # if scaler is not None:
    #     delta = scaler.inverse_transform(delta.reshape(-1, 1)).flatten()
    cluster_centers = select_centers(delta, valid_mask, center_quantile, center_z)
    
    # from sklearn.linear_model import LinearRegression
    # log_den = np.log(den_est + 1e-12)
    # log_delta = np.log(delta + 1e-12)
    # X_reg = log_den.reshape(-1, 1)
    # y_reg = log_delta
    # reg = LinearRegression().fit(X_reg, y_reg)
    # delta_pred = reg.predict(X_reg)
    # residuals = log_delta - delta_pred
    # residual_std = np.std(residuals)
    # if residual_std == 0:
    #     center_mask = np.zeros(n_samples, dtype=bool)
    #     center_mask[np.argmax(delta)] = True
    # else:
    #     z_scores = residuals / residual_std
    #     center_mask = z_scores > 3  # Example threshold for z-score
    # cluster_centers = np.where(center_mask)[0]
    # if cluster_centers.size == 0:
    #     cluster_centers = np.array([np.argmax(delta)])

    # Assign unique labels to cluster centers
    labels = propagate_labels(sorted_indices, nearest_higher, cluster_centers, valid_mask)

    if return_details:
        details = {
            "cluster_centers": cluster_centers,
            "1nn_higher": nearest_higher,
            "den_est": den_est,
            "delta": delta,
            "den_thres": den_thres,
            "density_cutoff": density_cutoff,
        }
        if cutoff_distance is not None:
            details["cutoff_distance"] = cutoff_distance
        return labels, details
    return labels


def GGDPC(X, den_est=None, grad_new=None, h_den=None, h_grad=None, den_thres=0, center_quantile=None, scaling=False, 
          cutoff_quantile=0.3, center_z=3.0, return_details=True, graph_dist=False, dendro=False):
    """
    Gradient-Guided Density Peak Clustering (GGDPC) algorithm.

    Parameters
    ----------
        X : ndarray of shape (n_samples, n_features)
            Input observations.
        den_est : ndarray of shape (n_samples,)
            Pre-computed density estimates for each point. If None, it will be computed 
            using KDE with a default bandwidth and Gaussian kernel. If set to "cutoff", it will use 
            a simple cutoff-based density estimation based on the number of neighbors within a certain distance.
        grad_new : ndarray of shape (n_samples, n_features), optional
            Pre-computed gradient estimates for each point. If None, it will be computed using a one-step 
            mean shift update based on the KDE density estimates.
        h_den : float
            Bandwidth for the kernel density estimation. If None, it will be determined by Silverman's rule of thumb.
        h_grad : float
            Bandwidth for the gradient estimation. If None, it will be determined by Silverman's rule of thumb.
        den_thres : float
            Density quantile threshold for classifying low-density points as noise. 
            Default is 0 (no points are considered noise).
        center_quantile : float
            Quantile for identifying cluster centers. If None, the cluster centers will be
            detected via z-score normalization of the 1NN uphill distance to the
            nearest higher-density point.
        scaling : bool
            If True, the input data will be standardized before computing the kernel density estimation. 
            Default is False.
        cutoff_quantile : float
            Quantile for determining the cutoff distance in the cutoff-based density estimation. 
            Default is 0.3 (30th percentile).
        center_z : float
            Z-score threshold for identifying cluster centers based on the 1NN uphill distance. Default is 3.0.
        return_details : bool
            If True, also return a dictionary with additional details about the clustering. Default is True.
        graph_dist : bool
            If True, compute and return the graph distances between points in the GGDPC graph.
        dendro : bool
            If True, compute and return the dendrogram linkage matrix for the GGDPC graph.
    
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
                - "density_cutoff": The actual density cutoff value used for noise classification.
                - "cutoff_distance": The cutoff distance used in the cutoff-based density estimation (if applicable).
    """
    X, den_thres, center_quantile, cutoff_quantile, center_z = validate_common_inputs(X, den_thres, center_quantile,
                                                                                      scaling, cutoff_quantile, center_z)
    n_samples = X.shape[0]
    X_work, scaler = prepare_coordinates(X, scaling=scaling)

    shifted_X = None
    if grad_new is not None:
        shifted_X = validate_update_locations(grad_new, X, scaler)

    cutoff_distance = None
    dist_mat = None
    if den_est is None:
        if shifted_X is None:
            if h_grad is None and h_den is None:
                den_est, shifted_X = gaussian_kde_ms_onestep(X_work, X_work, h=None)
            else:
                den_est = gaussian_kde(X_work, X_work, h=h_den)
                shifted_X = gaussian_ms_onestep(X_work, X_work, h=h_grad)
        else:
            den_est = gaussian_kde(X_work, X_work, h=h_den)
    elif isinstance(den_est, str):
        if den_est != "cutoff":
            raise ValueError('The only supported string value for den_est is "cutoff".')

        from sklearn.metrics import pairwise_distances
        dist_mat = pairwise_distances(X_work)
        den_est, cutoff_distance = cutoff_density(dist_mat, cutoff_quantile)

        if shifted_X is None:
            # Find the k nearest neighbors for each point in X and compute their average locations. 
            # If this set is null, then set the location as the current point 
            k_eff = min(int(np.ceil(np.log(n_samples))), n_samples - 1)
            knn_distances = dist_mat.copy()
            np.fill_diagonal(knn_distances, np.inf)
            knn_indices = np.argpartition(knn_distances, kth=k_eff - 1, axis=1)[:, :k_eff]
            shifted_X = X_work[knn_indices].mean(axis=1)
    else:
        den_est = validate_density(den_est, n_samples)
        if shifted_X is None:
            shifted_X = gaussian_ms_onestep(X_work, X_work, h=h_grad)

    den_est = validate_density(den_est, n_samples)
    if shifted_X is None or not np.all(np.isfinite(shifted_X)):
        raise ValueError("The one-step update locations must be finite.")

    # Density threshold mask
    density_cutoff = np.quantile(den_est, den_thres)
    valid_mask = den_est >= density_cutoff

    delta, nearest_higher, sorted_indices = gradient_guided_nearest_higher(X, shifted_X, den_est, dist_mat)
    # if scaler is not None:
    #     delta = delta * scaler.scale_[0]  # Scale delta back to the original scale
    cluster_centers = select_centers(delta, valid_mask, center_quantile, center_z)

    # Assign unique labels to cluster centers
    labels = propagate_labels(sorted_indices, nearest_higher, cluster_centers, valid_mask)

    if return_details:
        grad_details = (
            scaler.inverse_transform(shifted_X)
            if scaler is not None
            else shifted_X.copy()
        )
        details = {
            "cluster_centers": cluster_centers,
            "gg1nn_higher": nearest_higher,
            "den_est": den_est,
            "grad_new": grad_details,
            "delta": delta,
            "den_thres": den_thres,
            "density_cutoff": density_cutoff,
        }
        if cutoff_distance is not None:
            details["cutoff_distance"] = cutoff_distance
        if graph_dist:
            details["graph_dist"] = ggdpc_graph_distances(X_work, nearest_higher, sorted_indices)
        if dendro:
            details["dendrogram"] = ggdpc_dendrogram(X_work, nearest_higher)
        return labels, details
    return labels


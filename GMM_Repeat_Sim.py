#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Yikun Zhang
Last Editing: August 26, 2026

Comparative study of GGDPC and other DPC variants on Gaussian mixture data with 
different sample sizes.
"""

import numpy as np
import time
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from GGDPC import DPC, GGDPC
from DPC_variants import DPC_KNN_PCA, SNN_DPC, DPC_CE, DPC_DLP, DPC_MDNN
from utils import sample_gaussian_mixture

import sys

job_id = int(sys.argv[1])
print(job_id)

#=======================================================================================#

results = {}

def record_result(name, true_labels, pred_labels, elapsed):
    ari = adjusted_rand_score(true_labels, pred_labels)
    results[name] = {
        "Method": name,
        "ARI": ari,
        "Clusters Found": np.unique(pred_labels[pred_labels != -1]).size,
        "Noise Points": np.sum(pred_labels == -1),
        "Runtime (s)": elapsed}
    print(f"Adjusted Rand Index: {ari:.4f}")
    print(f"Runtime: {elapsed:.3f} seconds")

for n in [500, 1000, 2000, 5000, 8000]:
    X_dat, true_labels = sample_gaussian_mixture(n_samples=n, 
                                                 mu_lst=np.array([[0, 0], [1, 0]]), 
                                                 sigma_lst=np.array([0.3**2, 0.3**2]), 
                                                 random_state=job_id, return_labels=True)

    n_clusters = 2
    center_quantile = 1 - n_clusters / len(X_dat)

    # Original DPC
    start = time.perf_counter()
    h = None
    DPC_labels, DPC_det = DPC(X_dat, den_est=None, dist_mat=None, bandwidth=h, den_thres=0, 
                              center_quantile=center_quantile, scaling=False, center_z=4, return_details=True)
    elapsed = time.perf_counter() - start

    record_result("DPC", true_labels, DPC_labels, elapsed)


    ## GGDPC
    start = time.perf_counter()
    h = None
    GGDPC_labels, GGDPC_det = GGDPC(X_dat, den_est=None, grad_new=None, den_thres=0, 
                                    h_den=h, h_grad=h, center_quantile=center_quantile, 
                                    scaling=False, center_z=4, return_details=True, 
                                    graph_dist=True, dendro=False)
    elapsed = time.perf_counter() - start

    record_result("GGDPC", true_labels, GGDPC_labels, elapsed)


    ## DPC-KNN-PCA
    k_values = range(2, min(20, len(X_dat) - 1) + 1)
    k_scores = {}

    for k in k_values:
        labels_k, details_k = DPC_KNN_PCA(
            X_dat,
            n_clusters=n_clusters,
            k=k,
            n_components=2,
            return_details=True
        )

        # Evaluate against the ground-truth cluster labels
        k_scores[k] = adjusted_rand_score(true_labels, labels_k)
    # Obtain the oracle optimal k based on the ARI scores
    optimal_k = max(k_scores, key=k_scores.get)

    # Run the final model
    start = time.perf_counter()
    DPC_KNN_PCA_labels, DPC_KNN_PCA_det = DPC_KNN_PCA(
        X_dat, n_clusters=n_clusters, k=optimal_k, n_components=2, return_details=True)
    elapsed = time.perf_counter() - start

    record_result("DPC-KNN-PCA", true_labels, DPC_KNN_PCA_labels, elapsed)


    ## SNN-DPC
    k_values = range(2, min(20, len(X_dat)) + 1)
    k_scores = {}

    for k in k_values:
        labels_k, details_k = SNN_DPC(X_dat, n_clusters=n_clusters, k=k, scale=True, return_details=True)

        # Evaluate against the ground-truth cluster labels
        k_scores[k] = adjusted_rand_score(true_labels, labels_k)

    optimal_k = max(k_scores, key=k_scores.get)

    # Run the final model
    start = time.perf_counter()
    SNN_DPC_labels, SNN_DPC_det = SNN_DPC(
        X_dat, n_clusters=n_clusters, k=optimal_k, scale=True, return_details=True)
    elapsed = time.perf_counter() - start

    record_result("SNN-DPC", true_labels, SNN_DPC_labels, elapsed)


    ## DPC-CE
    start = time.perf_counter()
    DPC_CE_labels, DPC_CE_det = DPC_CE(
        X_dat, n_clusters=n_clusters, dc_percent=2.0, path_distance_ratio=0.25,
        distance_penalty_ratio=0.3, return_details=True)
    elapsed = time.perf_counter() - start

    record_result("DPC-CE", true_labels, DPC_CE_labels, elapsed)


    ## DPC-DLP
    start = time.perf_counter()
    DPC_DLP_labels, DPC_DLP_det = DPC_DLP(
        X_dat, n_clusters=n_clusters, neighbor_fraction=0.01,
        return_details=True)
    elapsed = time.perf_counter() - start

    record_result("DPC-DLP", true_labels, DPC_DLP_labels, elapsed)


    ## DPC-MDNN
    neighbor_values = range(2, min(20, len(X_dat) - 1) + 1)
    neighbor_scores = {}

    for manifold_neighbors in neighbor_values:
        labels_candidate = DPC_MDNN(X_dat, n_clusters=n_clusters,
                                    manifold_neighbors=manifold_neighbors, 
                                    scale=True, return_details=False)

        neighbor_scores[manifold_neighbors] = adjusted_rand_score(true_labels, labels_candidate)

    # Select the smallest value if multiple values have the same maximum ARI
    optimal_neighbors = max(neighbor_scores, key=neighbor_scores.get)

    # Run the final model
    start = time.perf_counter()
    DPC_MDNN_labels, DPC_MDNN_det = DPC_MDNN(
        X_dat, n_clusters=n_clusters, manifold_neighbors=optimal_neighbors, scale=True, return_details=True)
    elapsed = time.perf_counter() - start

    record_result("DPC-MDNN", true_labels, DPC_MDNN_labels, elapsed)

    summary = pd.DataFrame(results.values())
    summary.to_csv(f"./Results/results_GMM_{n}_samples_job_{job_id}_new.csv", index=False)
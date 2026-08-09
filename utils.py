"""
@author: Yikun Zhang
Last Editing: Mar 18, 2026

Description: Utility functions for the Gradient-Guided Density Peak Clustering 
(GGDPC) algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

#=====================================================================================#

def sample_gaussian_mixture(n_samples, mu_lst, sigma_lst, weights=None, random_state=None, 
                            return_labels=False):
    """
    Sample points from a mixture of Gaussian distributions.

    Parameters
    ----------
        n_samples : int
            Total number of samples to generate.
        mu_lst : list of array-like
            List of mean vectors for each Gaussian component. Each element should be a 1D 
            array or a scalar (for 1D).
        sigma_lst : list of array-like
            List of covariance matrices or variances for each Gaussian component. For 1D,
            each element can be a scalar (variance) or a 1x1 matrix. For higher dimensions, 
            each element can be a scalar (isotropic), a 1D array (diagonal), or a 2D array 
            (full covariance).
        weights : array-like, optional
            Mixing weights for each component. If None, components are weighted equally.
        random_state : int or np.random.Generator, optional
            Random seed or random number generator for reproducibility.
        return_labels : bool, optional
            If True, also return the component labels for each sample.

    Returns
    -------
        samples : ndarray
            Generated samples from the Gaussian mixture, shape (n_samples, n_features).
        labels : ndarray, optional
            Component labels for each sample, shape (n_samples,). 
            Returned if return_labels is True.
    """
    if len(mu_lst) != len(sigma_lst):
        raise ValueError("mu_lst and sigma_lst must have the same length.")
    n_components = len(mu_lst)
    if n_components == 0:
        raise ValueError("mu_lst and sigma_lst cannot be empty.")

    rng = np.random.default_rng(random_state)

    if weights is None:
        weights = np.full(n_components, 1.0 / n_components)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_components,):
            raise ValueError("weights must have length equal to len(mu_lst).")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")
        if weights.sum() == 0:
            raise ValueError("weights must sum to a positive value.")
        weights = weights / weights.sum()

    component_ids = rng.choice(n_components, size=n_samples, p=weights)

    first_mu = np.atleast_1d(mu_lst[0]).astype(float)
    dim = first_mu.size
    samples = np.empty((n_samples, dim), dtype=float)

    for k in range(n_components):
        idx = np.where(component_ids == k)[0]
        if idx.size == 0:
            continue

        mu = np.atleast_1d(mu_lst[k]).astype(float)
        if mu.size != dim:
            raise ValueError("All means in 'mu_lst' must have the same dimension.")
        sigma = np.asarray(sigma_lst[k], dtype=float)

        if dim == 1:
            if sigma.ndim == 0:
                std = np.sqrt(sigma)
            elif sigma.shape == (1, 1):
                std = np.sqrt(float(sigma[0, 0]))
            else:
                raise ValueError("For 1D, each sigma must be a scalar or 1*1 matrix.")
            samples[idx, 0] = rng.normal(loc=mu[0], scale=std, size=idx.size)
        else:
            if sigma.ndim == 0:
                cov = np.eye(dim) * sigma
            elif sigma.shape == (dim,):
                cov = np.diag(sigma)
            elif sigma.shape == (dim, dim):
                cov = sigma
            else:
                raise ValueError(f"For {dim}D, each sigma must be scalar, length-{dim} diagonal, or {dim}x{dim}.")
            samples[idx] = rng.multivariate_normal(mean=mu, cov=cov, size=idx.size)

    samples = samples.squeeze()
    if return_labels:
        return samples, component_ids
    return samples


def gaussian_kde(x, data, h=None, verbose=False):
    """
    The d-dim Euclidean kernel density estimator with Gaussian kernel.

    Parameters
    ----------
        x : ndarray of shape (m, d)
            Query points.
        data : ndarray of shape (n, d)
            Sample points.
        h : float, optional
            Bandwidth. If None, Silverman's rule of thumb is used.
        verbose : bool, default=False
            Whether to print the bandwidth.

    Returns
    -------
        f_hat : ndarray of shape (m,)
            KDE evaluated at query points.
    """
    x = np.asarray(x, dtype=float)
    data = np.asarray(data, dtype=float)

    if x.ndim != 2 or data.ndim != 2:
        raise ValueError("x and data must both be 2D arrays.")
    if x.shape[1] != data.shape[1]:
        raise ValueError("x and data must have the same number of columns.")

    n, d = data.shape
    if h is None:
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.mean(np.std(data, axis=0))
    h = float(h)
    if h <= 0:
        raise ValueError("h must be positive.")
    if verbose:
        print(f"The current bandwidth is {h}.\n")

    # Shape: (m, n, d)
    diff = (x[:, None, :] - data[None, :, :]) / h
    # Shape: (m, n)
    sqdist = np.sum(diff**2, axis=2)

    # Gaussian kernel average
    kernel_vals = np.exp(-sqdist/2)
    f_hat = np.mean(kernel_vals, axis=1) / ((2 * np.pi) ** (d / 2) * h**d)

    return f_hat


def gaussian_ms_onestep(x, data, h=None, verbose=False):
    """
    The d-dim one-step mean shift update under Gaussian kernel.

    Parameters
    ----------
        x : ndarray of shape (m, d)
            Query points.
        data : ndarray of shape (n, d)
            Sample points.
        h : float, optional
            Bandwidth. If None, Silverman's rule of thumb is used.
        verbose : bool, default=False
            Whether to print the bandwidth.

    Returns
    -------
        ms_new : ndarray of shape (m, d)
            One-step iterations of the Gaussian mean shift algorithm from "x".
    """
    x = np.asarray(x, dtype=float)
    data = np.asarray(data, dtype=float)

    if x.ndim != 2 or data.ndim != 2:
        raise ValueError("x and data must both be 2D arrays.")
    if x.shape[1] != data.shape[1]:
        raise ValueError("x and data must have the same number of columns.")

    n, d = data.shape
    if h is None:
        h = (4/(d+2))**(1/(d+4))*(n**(-1/(d+4)))*np.mean(np.std(data, axis=0))
    h = float(h)
    if h <= 0:
        raise ValueError("h must be positive.")
    if verbose:
        print(f"The current bandwidth is {h}.\n")

    # Shape: (m, n, d)
    diff = (x[:, None, :] - data[None, :, :]) / h
    # Shape: (m, n)
    sqdist = np.sum(diff**2, axis=2)

    # Gaussian kernel average
    kernel_vals = np.exp(-sqdist/2)
    f_hat = np.sum(kernel_vals, axis=1).reshape(-1,1)
    ms_new = np.dot(kernel_vals, data) / f_hat
    
    return ms_new


def plot_clusters(X, clu, start, end, centers, directed=False, title=None):
    segments = np.stack([X[start], X[end]], axis=1)

    # plot all nodes
    pts = np.vstack([X[start], X[end]])
    plt.scatter(X[:, 0], X[:, 1], c=clu, cmap='viridis', s=10)
    plt.scatter(X[centers, 0], X[centers, 1], c='red', marker='X', s=70, label='Cluster Centers')
    if directed:
        plt.quiver(X[start, 0], X[start, 1], X[end, 0] - X[start, 0], X[end, 1] - X[start, 1], 
                   angles='xy', scale_units='xy', scale=1, color='black', width=0.003, alpha=0.7)
    else:
        lc = LineCollection(segments, colors="black", linewidths=1, alpha=0.7)
        plt.gca().add_collection(lc)
    # plt.gca().autoscale()
    plt.legend()
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()
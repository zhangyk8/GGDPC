# Gradient-Guided Density Peak Clustering (GGDPC)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the Python implementation of gradient-guided density peak clustering (GGDPC), together with the original density peak clustering (DPC) algorithm and several DPC variants.

**Paper Reference**: Y. Zhang and Y.-C. Chen. *Gradient-Guided Density Peak Clustering* (2026+).

## Overview

GGDPC retains DPC's simple graph construction but uses one gradient ascent step to guide each nearest-higher-density search. The resulting uphill paths are more stable and better reflect the geometry of the population gradient flow.

<p align="center">
  <img src="Figures/GGDPC_overview.png"
       width="900"
       alt="Four complementary views of GGDPC on the Old Faithful data: graph, density waterfall, decision diagram, and dendrogram">
</p>

<p align="center">
  <em>GGDPC on pairs of consecutive eruption durations in the Old Faithful data: the directed graph, density waterfall, decision diagram, and induced dendrogram; see `Old_Faithful_Data.ipynb` for details.</em>
</p>

### GGDPC Algorithm At a Glance

Given observations $\mathbf{X}_1,\ldots,\mathbf{X}_n$, GGDPC:

1. Estimates the density and a one-step gradient ascent update at every observation.
2. Links each observation to the higher-density observation closest to its updated location.
3. Uses the distance from the original observation to its parent as the gradient-guided 1NN uphill distance.
4. Identifies cluster centers from unusually large uphill distances.
5. Assigns each remaining observation by following the directed graph to a selected center.

## Requirements

- Python >= 3.10 (earlier Python 3 versions may also work)
- [NumPy](https://numpy.org/)
- [SciPy](https://scipy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Matplotlib](https://matplotlib.org/)
- Optional: [pandas](https://pandas.pydata.org/), [statsmodels](https://www.statsmodels.org/), and [JupyterLab](https://jupyter.org/) for the notebooks and simulation study

## File Descriptions

### Core Modules

| File | Description |
| --- | --- |
| `GGDPC.py` | Main implementations of GGDPC and the original DPC |
| `dpc_helpers.py` | Parent searches, center selection, label propagation, graph distances, and dendrogram construction |
| `utils.py` | KDE, one-step mean shift, synthetic-data generation, and plotting utilities |
| `DPC_variants.py` | Implementations of DPC-KNN-PCA, SNN-DPC, DPC-CE, DPC-DLP, and DPC-MDNN |

### Examples and Visualization

| File | Description |
| --- | --- |
| `Old_Faithful_Data.ipynb` | Old Faithful case study and the four panels shown above (Figure 1 in the paper) |
| `GMM_Data.ipynb` | Two-component Gaussian-mixture example, method comparisons, and result visualization (Figure 2 in the paper) |

### Simulation Study

| File | Description |
| --- | --- |
| `GMM_Repeat_Sim.py` | Runs repeated Gaussian-mixture simulations |
| `Syn_Res.py` | Aggregates the simulation outputs |
| `Syn_Results/` | Contains the combined, mean, and standard-deviation result tables |
| `Figures/` | Contains the publication figures and README overview image |

The simulation scripts have corresponding `.sbatch` files for submission to a Slurm cluster.

## Usage

### Basic Example

The following example uses the two-component Gaussian mixture from `GMM_Data.ipynb`. Run it from the repository root so the local modules can be imported.

```python
import numpy as np

from GGDPC import GGDPC
from utils import sample_gaussian_mixture

X_dat, _ = sample_gaussian_mixture(n_samples=1500, mu_lst=np.array([[0, 0], [1, 0]]), sigma_lst=np.array([0.3**2, 0.3**2]), random_state=0, return_labels=True)

labels, details = GGDPC(X_dat, center_z=4)

cluster_ids = np.unique(labels[labels != -1])
print("Clusters found:", cluster_ids.size)
print("Center indices:", details["cluster_centers"])
```

GGDPC finds two clusters in this example without being given the number of clusters. By default, `GGDPC` returns `(labels, details)`; observations classified as noise receive label `-1`.

### Main Arguments of `GGDPC()`

- `center_z` controls the automatic center-selection threshold, while `center_quantile` can instead specify an uphill-distance quantile.
- `den_thres` marks observations below a chosen density quantile as noise.
- `h_den` and `h_grad` control the bandwidth for density and gradient estimations, respectively.
- `graph_dist=True` adds cumulative GGDPC path lengths from observations to the graph root, while `dendro=True` adds a SciPy linkage matrix.

With `return_details=True`, the returned dictionary includes the selected centers, parent links, density estimates, one-step update locations, and uphill distances.

## License

GGDPC is MIT licensed, as found in the [LICENSE](LICENSE) file.

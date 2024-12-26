# Assumption-Lean Post-Integrated Inference with Negative Control Outcomes

This repository contains the reproducibility materials for post-integrated inference (PII) procedure from the paper [Du2024b].


# Files

## Simulation

- `ex1-simu-PSI.py`: Simulation.
- `ex2-simu_conv_rate.py`: Evaluate the convergence rate in simulation.
- `Plot-simu.ipynb`: Plot the results of simulation.

## Real data

- `data/LUHMES.h5`: Preprocessed LUHMES data can be obtained from [[Du2024a]](https://github.com/jaydu1/CausalMultiOutcomes/blob/main/data/LUHMES/LUHMES.h5) and stored to this folder.
- `ex3-LUHMES_PSI.py`: Apply PSI to the LUHMES data.
- `ex4-LUHMES_RUV.R`: Apply RUV and CATE to the LUHMES data.
- `Plot-LUHMES.ipynb`: Plot the results of real data and save common discoveries for GO analysis.
- `ex3-GO.R`: GO analysis of results.

## Utility functions

- `PII.py`: Post-integrated inference.
- `cate`: The implementation of CATE [Wang2017]. The original CRAN package is built under an older version of R, which is not compatible with certain new R packages.


# Dependencies

The dependencies for running PII method are listed in `environment.yml` and can be installed by running 
```bash
conda env create -f environment.yml
```

The Python dependencies for reproducibility are listed below:

Package | Version
---|---
h5py | 3.10.0
joblib | 1.3.2
matplotlib-base | 3.8.3
matplotlib-venn | 1.1.1
numba | 0.59.0
numpy | 1.26.4
pandas | 2.2.1
python | 3.12.2
scikit-learn | 1.4.1.post1
scipy | 1.12.0
statsmodels | 0.14.4
tqdm | 4.66.2
upsetplot | 0.9.0

The R dependencies for reproducibility are listed below:

Package | Version
---|---
bioconductor-annotationdbi | 1.64.1
bioconductor-clusterprofiler | 4.10.0
bioconductor-matrixgenerics | 1.14.0
bioconductor-org.hs.eg.db | 3.18.0
bioconductor-rrvgo | 1.14.0
bioconductor-sva | 3.50.0
r-base | 4.3.3
r-biocmanager | 1.30.22
r-dbplyr | 2.4.0
r-dplyr | 1.1.4
r-dqrng | 0.3.2
r-dtplyr | 1.3.1
r-esabcv | 1.2.1.1
r-essentials | 4.3.1
r-ggplot2 | 3.5.0
r-hdf5r | 1.3.11
r-mass | 7.3_60
r-matrix | 1.6_5
r-patchwork | 1.2.0
r-plyr | 1.8.9
r-purrr | 1.0.2
r-ruv | 0.9.7.1
r-seurat | 5.0.1
r-stringr | 1.5.1
r-survival | 3.5_8
r-tidyr | 1.3.1
r-tidyverse | 2.0.0



# Reference

- [Wang2017] Wang, J., Zhao, Q., Hastie, T., & Owen, A. B. (2017). Confounder adjustment in multiple hypothesis testing. Annals of statistics, 45(5), 1863.
- [Du2024a] Du, J. H., Zeng, Z., Kennedy, E. H., Wasserman, L., & Roeder, K. (2024). Causal inference for genomic data with multiple heterogeneous outcomes. arXiv preprint arXiv:2404.09119.
- [Du2024b] Du, J. H., Roeder, K., & Wasserman, L. (2024). Assumption-Lean Post-Integrated Inference with Negative Control Outcomes. arXiv preprint arXiv:2410.04996.


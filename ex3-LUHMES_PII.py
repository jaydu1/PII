import sys
import os
import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn_ensemble_cv import reset_random_seeds, Ensemble, ECV
from PII import PII, fit_logistic, fit_rf, fit_rf_ind
from functools import partial
import h5py

# Set the result path
path_result = 'results/LUHMES/'

# Set the number of cores for parallel processing
n_cores = len(os.sched_getaffinity(0))
n_cores = int(n_cores // 2 + 1)
os.environ["OMP_NUM_THREADS"] = f"{n_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_cores}"
os.environ["MKL_NUM_THREADS"] = f"{n_cores}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_cores}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_cores}"
os.environ["NUMBA_CACHE_DIR"] = '/tmp/numba_cache'
os.makedirs(path_result, exist_ok=True)

# Load data from HDF5 file
with h5py.File('data/LUHMES/LUHMES.h5', 'r') as f:
    print(f.keys())
    counts = np.array(f['counts'], dtype='float')
    sgRNA = np.array(f['sgRNA'], dtype='S32').astype(str)
    gene_names = np.array(f['gene_names'], dtype='S32').astype(str)
    var_features = pd.DataFrame(np.array(f['var.features']))
    var_features.index = gene_names
    cell_ids = np.array(f['cell_ids'], dtype='S32').astype(str)
    covariates = pd.DataFrame(np.array(f['covariates'])).astype(float).values
    pt_state = covariates[:, -1]
    covariates = np.c_[np.log(np.sum(counts, axis=1)), covariates[:, :-1]]
    id_cell = (~np.char.startswith(sgRNA, 'RELN'))
    counts = counts[id_cell]
    sgRNA = sgRNA[id_cell]
    cell_ids = cell_ids[id_cell]
    covariates = covariates[id_cell]
    pt_state = pt_state[id_cell]
    print(np.unique(covariates[:, -1:]))
    covariates = np.c_[
        pt_state,
        covariates[:, :-1],
        OneHotEncoder(drop='first', sparse_output=False).fit_transform(covariates[:, -1:])
    ]

# Define gRNA names
gRNA_names = ['Nontargeting', 'ADNP', 'ARID1B', 'ASH1L', 'CHD2', 'CHD8', 'CTNND2', 'DYRK1A',
              'HDAC5', 'MECP2', 'MYT1L', 'POGZ', 'PTEN', 'SETD5']

# One-hot encode perturbed gRNA
perturbed = np.array([s.split('_')[0] for s in sgRNA])
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False, categories=[gRNA_names])
perturbed_one_hot_encoded = one_hot_encoder.fit_transform(perturbed.reshape(-1, 1))
covariates = np.c_[covariates, perturbed_one_hot_encoded]

print(counts.shape, len(gRNA_names), cell_ids.shape, covariates.shape)
print(perturbed_one_hot_encoded.shape)

# Standardize covariates
scaler = StandardScaler()
X = scaler.fit_transform(covariates)
Y = counts

n, p = Y.shape
d = X.shape[1]

print(X.shape, Y.shape)

# Select features based on variance
C = np.sort(np.argsort(var_features['vst.variance.standardized'].values)[:4000])
CC = np.where(var_features['vst.variance.standardized'] > 1.)[0]
print(len(C), len(CC))

# Normalize counts
Y_norm = np.log1p(Y[:, C] / np.sum(Y[:, C], axis=1, keepdims=True) * 1e4)

# Get embedding method and number of components from command line arguments
embedding = sys.argv[1]
r = int(sys.argv[2])
print(embedding, r)

# Perform PCA or load precomputed embedding
if embedding == 'PCA':
    Y_scale = StandardScaler().fit_transform(Y_norm) / np.sqrt(n)
    pca = PCA().fit(Y_scale)
    variance_threshold = 0.9
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    print(pca.singular_values_[:50])
    U_hat = pca.transform(Y_scale)[:, :r]
else:
    with h5py.File(f'results/LUHMES/output_{embedding}_{r}.h5', 'r') as f:
        print(f.keys())
        U_hat = np.array(f['U_hat'], dtype='float').T

U = U_hat
print(U_hat.shape)

# Define partial functions for PII
X_U = partial(fit_rf_ind, M=25, M_max=100, verbose=0)
Y_XU = partial(fit_rf_ind, M=25, M_max=100, verbose=0)
gY_U = partial(fit_rf_ind, M=25, M_max=100, verbose=0)

# Run PII
res = PII(Y, X, C, U, X_U, Y_XU, gY_U, CC=CC, link='log', verbose=100, crossfit=1)

# Save results
with open(f'results/LUHMES/output_PII_{embedding}_{r}.npz', 'wb') as f:
    np.savez(f, **res)

import sys
import os
import numpy as np
import pandas as pd

import scipy as sp
from scipy.linalg import svd
from scipy.special import expit
from scipy.stats import t, norm
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from PII import PII, fit_logistic, fit_rf, fit_rf_ind

from joblib import Parallel, delayed
from functools import reduce
from tqdm import tqdm

# Set the number of threads for parallel processing
n_cores = len(os.sched_getaffinity(0))
n_cores = int(n_cores // 2 + 1)
os.environ["OMP_NUM_THREADS"] = f"{n_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_cores}"
os.environ["MKL_NUM_THREADS"] = f"{n_cores}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_cores}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_cores}"
os.environ["NUMBA_CACHE_DIR"] = '/tmp/numba_cache'

# Create results directory if it doesn't exist
path_result = 'results/ex3/'
os.makedirs(path_result, exist_ok=True)

# Simulation parameters
n_simu = 50
p = 1000
s = 0.2
d = 1
r = 10
rho = 1.0
sigma_U = 1.0
sigma_UE = 1.0 # sigma_UE = 0.8

def compute_similarity_matrix(X):
    """Compute similarity matrix for X."""
    return np.dot(X, X.T)

def compute_projection_matrix_svd(A):
    """Compute projection matrix P for the subspace defined by A using SVD."""
    U, _, _ = svd(A, full_matrices=False)
    P = np.dot(U, U.T)
    return P

def generate_data(m, n, p, s, seed, rho=rho, sigma_U=sigma_U, sigma_UE=sigma_UE):
    """Generate synthetic data for the simulation."""
    np.random.seed(seed)
    X = np.random.normal(size=(n, d))
    B = rho * np.random.binomial(1, s, size=(1, p))
    
    BX = np.random.uniform(size=(d, r), low=-1, high=1)
    U = X @ BX + sigma_UE * np.random.normal(size=(n, r))
    BU = np.random.uniform(size=(r, p), low=-1, high=1) / np.sqrt(r)
    func = lambda u: sigma_U * (BU.T @ u)

    Theta = X @ B + np.apply_along_axis(func, 1, U)
    prob = expit(Theta)
    Y = np.random.binomial(1, prob)
    
    X_test = np.random.normal(size=(m, d))
    U_test_2 = X_test @ BX
    eps = sigma_UE * np.random.normal(size=(m, r))
    U_test = U_test_2 + eps
    Theta = X_test @ B + np.apply_along_axis(func, 1, U_test)
    Y_test = expit(Theta)
    Y_test_2 = np.random.binomial(1, Y_test)

    BX_U = BX.T / np.linalg.norm(BX)**2
    noise = np.mean((eps @ BX_U)**2)

    C = np.random.choice(np.where(B[0, :] == 0)[0], int(p / 2), replace=False)
    return Y[:, C], Y, X, U, C, B, Y_test, Y_test_2, X_test, U_test, noise

def run_simu(m, n, p, s, seed, r=r):
    """Run a single simulation."""
    Y_C, Y, X, U, C, B, Y_test, Y_test_2, X_test, U_test, noise = generate_data(m, n, p, s, seed, rho=rho, sigma_U=sigma_U, sigma_UE=sigma_UE)
    Y = np.r_[Y, Y_test_2]
    U = np.r_[U, U_test]
    
    CC = np.setdiff1d(np.arange(p), C)

    res = []
    for _U, has_U in zip([None, U], [False, True]):
        if _U is None:
            _U = lambda Y, Y_test: PCA(n_components=r).fit_transform(Y_test)
        else:
            _U = lambda Y, Y_test: U
        U_hat = _U(Y_C, Y[:, C])
        U_train, U_test = U_hat[:n], U_hat[n:]

        _X_pred = fit_rf(U_train, X, U_test)
        mse_X = np.maximum(np.mean((_X_pred - X_test)**2) - noise, 1e-2)

        XU_hat = np.hstack((X, U_train))
        XU_hat_test = np.hstack((X_test, U_test))
        _Y_pred = fit_rf(XU_hat, Y[:n, CC], XU_hat_test)
        mse_Y = np.max(np.mean((_Y_pred - Y_test[:, CC])**2, axis=0))

        res.append([has_U, n, p, s, mse_X, mse_Y])

    return res

# Run simulations in parallel
with Parallel(n_jobs=8, verbose=0, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
    for n in [100, 150, 200, 300, 400]:
        m = 2000
        df_res = pd.DataFrame()

        res = parallel(
            delayed(run_simu)(m, n, p, s, seed) 
            for seed in tqdm(np.arange(n_simu), desc='seed')
        )    

        res = pd.DataFrame(np.concatenate(res, axis=0), columns=[
            'has_U', 'n', 'p', 's', 'mse_X', 'mse_Y'
        ])
        df_res = pd.concat([df_res, res], axis=0)

        df_res.to_csv(f'{path_result}res_{n}_{p}_sigma_{sigma_UE:.01f}.csv', index=False)

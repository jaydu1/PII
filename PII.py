import numpy as np
from scipy.special import logit
from scipy.stats import t
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.model_selection import KFold


def check_and_print(variable, name, verbose):
    if variable is None:
        return fit_rf
    elif isinstance(variable, (int, float, np.number, np.ndarray)):
        if verbose > 0:
            print(f"Using input {name}...")
        return lambda *args: variable
    elif callable(variable):
        if verbose > 0:
            print(f"Estimating {name}...")
        return variable
    else:
        if verbose > 0:
            print(f"{name} is neither a numeric variable nor a function")
        return None
        
def PII(Y, X, C, U, 
    X_U=None, Y_XU=None, gY_U=None,
    Y_C=None, CC=None, link='identity', 
    crossfit=1, verbose=0):
    '''
    Batch corrected inference (PII) with negative control outcomes.

    Parameters
    ----------
    Y : array-like of shape (n, p)
        Outcome matrix.
    X : array-like of shape (n, p)
        Feature matrix.
    C : array-like of shape (p, )
        Indices of control outcomes.
    U : array-like of shape (n, r) or callable
        Estimated embedding or a callable function that takes Y[:,C] and output U.
    X_U : array-like of shape (n, d) or callable
        Estimated regression function E[X|U] or a callable function that takes (U,X) and output E[X|U].
    Y_XU : array-like of shape (n, p) or callable
        Estimated regression function E[Y|X,U] or a callable function that takes ([X,U],Y) and output E[Y|X,U].
    gY_U : array-like of shape (n, p) or callable
        Estimated regression function E[g(E[Y|X,U])|U] or a callable function that takes (U,g(E[Y|X,U])) and output E[g(E[Y|X,U])|U].
    Y_C : array-like of shape (n, p), optional
        Control outcomes. The default is None and uses the control outcomes from observations Y. Alternatively, one can supply extra negative control outcomes of shape [m, p-len(C)].
    link : str, optional
        Link function. The default is 'identity'. Other options are 'logit' and 'log'.


    Returns
    -------
    result : dict
        A dictionary containing the following
        - hbeta : array-like of shape (p, )
            Estimated coefficients.
        - var_beta : array-like of shape (p, )
            Estimated variance of coefficients.
        - t_values : array-like of shape (p, )
            t-values.
        - p_values : array-like of shape (p, )
            p-values.
        - U_hat : array-like of shape (n, r)
            Estimated embedding.
    '''
    n, p = Y.shape
    # Input: (Y, X) and a set of control outcomes C

    if link == 'logit':
        g = logit; gp = lambda p: 1 / (p * (1 - p))
    elif link == 'identity':
        g = lambda x: x; gp = lambda x: 1
    elif link == 'log':
        g = np.log; gp = lambda x: 1/x

    if CC is None:
        CC = np.setdiff1d(np.arange(p), C)

    # Check if U is a numeric variable or a function    
    U = check_and_print(U, "embedding", verbose)

    if Y_C is None:
        Y_C = Y[:,C]
    U_hat = U(Y_C, Y[:,C])

    XU_hat = np.hstack((X, U_hat))

    mu = np.zeros((n, len(CC)))
    X_pred = np.zeros((n, X.shape[1]))
    Y_pred = np.zeros((n, len(CC)))
    if crossfit > 1:
        assert callable(X_U) and callable(Y_XU), "X_U and Y_XU must be callable functions for crossfit > 1"
        kf = KFold(n_splits=crossfit)
        iters = kf.split(Y)
    else:
        iters = [(np.arange(n), np.arange(n))]

    for train_index, test_index in iters:
        # Step 2: estimate nuisance functions
        X_U = check_and_print(X_U, "E[X|U]", verbose)
        _X_pred = X_U(U_hat[train_index], X[train_index], U_hat[test_index])

        Y_XU = check_and_print(Y_XU, "E[Y|X,U]", verbose)
        _Y_pred = Y_XU(XU_hat[train_index], Y[:,CC][train_index], XU_hat[test_index])

        if link == 'logit':
            _Y_pred = np.clip(_Y_pred, 1e-5, 1-1e-5)
        elif link == 'log':
            _Y_pred = np.clip(_Y_pred, 1e-3, np.inf)

        gY_U = check_and_print(gY_U, "E[g(E[Y|X,U])|U]", verbose)
        Y_U_pred = gY_U(U_hat[train_index], g(_Y_pred), U_hat[test_index])

        # Step 3: estimate the estimand
        _mu = gp(_Y_pred) * (Y[:,CC][test_index] - _Y_pred) + g(_Y_pred) - Y_U_pred
        
        X_pred[test_index] = _X_pred
        Y_pred[test_index] = _Y_pred
        mu[test_index] = _mu

    # linear_reg = LinearRegression(fit_intercept=False)
    linear_reg = Lasso(fit_intercept=False, alpha=1e-4)
    linear_reg.fit(X - X_pred, mu)
    hbeta = linear_reg.coef_.T

    hSigma = np.einsum('ij,ik->ijk', X - X_pred, X - X_pred)
    psi = np.matmul((X - X_pred)[:, :, np.newaxis], mu[:, np.newaxis, :]) - hSigma @ hbeta
    varphi = np.linalg.solve(np.mean(hSigma, axis=0) + 1e-4*np.identity(hSigma.shape[-1]), psi)
    var_beta = np.mean(varphi**2, axis=0)/(n-1)

    # Degrees of freedom
    df = n - 1
    t_values = hbeta / np.sqrt(var_beta)
    p_values = 2 * (1 - t.cdf(np.abs(t_values), df))

    return {
        'hbeta': hbeta,
        'var_beta': var_beta,
        't_values': t_values,
        'p_values': p_values,
        'U_hat': U_hat,
        'X_pred': X_pred,
        'Y_pred': Y_pred,
        'Y_U_pred': Y_U_pred
    }





from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from tqdm import tqdm

def fit_logistic(X, Y, X_test=None, n_jobs=-1, verbose=0):
    # Fit logistic regression of Y on X and U_hat    
    X_test = X if X_test is None else X_test
    def _fit(X, y, X_test):
        logistic_reg = LogisticRegression(C=1e4)
        logistic_reg.fit(X, y)
        return logistic_reg.predict_proba(X_test)[:,1]
    
    iterable = np.arange(Y.shape[1])
    if verbose > 0:
        print('Fitting logistic regression')
        iterable = tqdm(iterable, desc="j")

    with Parallel(n_jobs=n_jobs, verbose=verbose, temp_folder='~/tmp/', timeout=99999, max_nbytes=None) as parallel:
        Y_pred = parallel(delayed(_fit)(X, Y[:,j], X_test) for j in iterable)
    Y_pred = np.array(Y_pred).T

    return Y_pred



from sklearn_ensemble_cv import reset_random_seeds, Ensemble, ECV
from sklearn.tree import DecisionTreeRegressor
def fit_rf(X, y, X_test=None, M=25, M_max=100, 
    grid_regr = {'max_depth': [1,3,5]},
    grid_ensemble = {'max_samples': np.linspace(0.25, 1., 4)},
    verbose=0):
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    res_ecv, info_ecv = ECV(
        X, y, DecisionTreeRegressor, grid_regr, grid_ensemble, 
        kwargs_ensemble={'verbose':verbose},
        # kwargs_regr={'min_samples_split': 0.05, 'min_samples_leaf': 0.05},
        M=M, M0=M, M_max=M_max, return_df=True
    )
    info_ecv['best_params_ensemble']['n_estimators'] = info_ecv['best_n_estimators_extrapolate']
    regr = Ensemble(
        estimator=DecisionTreeRegressor(**info_ecv['best_params_regr']),
        **info_ecv['best_params_ensemble']).fit(X, y)
        
    if X_test is None:
        X_test = X
    return regr.predict(X_test).reshape(-1, y.shape[1])


def fit_rf_ind(X, Y, *args, **kwargs):
    Y_hat = Parallel(n_jobs=-1)(delayed(fit_rf)(X, Y[:,j], *args, **kwargs)
        for j in tqdm(range(Y.shape[1])))
    Y_pred = np.concatenate(Y_hat, axis=-1)
    return Y_pred
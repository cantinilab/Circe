# Code from https://github.com/skggm/skggm


"""Metrics for cross validation with Gaussian graphical models."""
import numpy as np
from sklearn.utils.extmath import fast_logdet


def log_likelihood(covariance, precision):
    """Computes the log-likelihood between the covariance and precision
    estimate.

    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance

    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested

    Returns
    -------
    log-likelihood
    """
    assert covariance.shape == precision.shape
    dim, _ = precision.shape
    log_likelihood_ = (
        -np.sum(covariance * precision)
        + fast_logdet(precision)
        - dim * np.log(2 * np.pi)
    )
    log_likelihood_ /= 2.
    return log_likelihood_


def kl_loss(covariance, precision):
    """Computes the KL divergence between precision estimate and
    reference covariance.

    The loss is computed as:

        Trace(Theta_1 * Sigma_0) - log(Theta_0 * Sigma_1) - dim(Sigma)

    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance

    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the covariance model to be tested

    Returns
    -------
    KL-divergence
    """
    assert covariance.shape == precision.shape
    dim, _ = precision.shape
    logdet_p_dot_c = fast_logdet(np.dot(precision, covariance))
    return 0.5 * (np.sum(precision * covariance) - logdet_p_dot_c - dim)


def quadratic_loss(covariance, precision):
    """Computes ...

    Parameters
    ----------
    covariance : 2D ndarray (n_features, n_features)
        Maximum Likelihood Estimator of covariance

    precision : 2D ndarray (n_features, n_features)
        The precision matrix of the model to be tested

    Returns
    -------
    Quadratic loss
    """
    assert covariance.shape == precision.shape
    dim, _ = precision.shape
    return np.trace((np.dot(covariance, precision) - np.eye(dim)) ** 2)


def cov_with_appended_zeros(X, m, *, ddof=1, rowvar=False, return_mean=False):
    """
    Covariance of X once `m` zero rows are appended.

    Parameters
    ----------
    X : array_like
        Data without the zero rows.
        Shape is (n_samples, n_features) if rowvar=False,
        or (n_features, n_samples) if rowvar=True.
    m : int
        Number of missing all-zero observations.
    ddof : {0,1}, optional
        0 – population covariance, 1 – unbiased (default).
    rowvar : bool, optional
        Same meaning as in numpy.cov.  Default False.
    return_mean : bool, optional
        Also return the corrected mean if True.

    Returns
    -------
    cov_full : ndarray
        Corrected covariance matrix.
    mean_full : ndarray, optional
        Corrected mean vector (only if return_mean=True).
    """
    X = np.asarray(X)
    cov_n = np.cov(X, rowvar=rowvar, ddof=ddof)

    # fast exit when nothing has to be added
    if m == 0 and not return_mean:
        return cov_n

    # compute the mean only if needed
    mean_n = X.mean(axis=1 if rowvar else 0)
    if m == 0:                      # mean requested, but no correction
        return (cov_n, mean_n) if return_mean else cov_n

    n = X.shape[1] if rowvar else X.shape[0]
    N = n + m

    # outer product of the mean with itself; orientation does not matter
    S_new = (n - ddof) * cov_n + (n * m) / N * np.outer(mean_n, mean_n)
    cov_full = S_new / (N - ddof)

    if return_mean:
        mean_full = (n / N) * mean_n
        return cov_full, mean_full

    return cov_full

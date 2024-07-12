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
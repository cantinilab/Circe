# Code from https://github.com/skggm/skggm


from __future__ import absolute_import

import sys
import time
import operator
import numpy as np
from functools import partial

from sklearn.covariance import EmpiricalCovariance
from sklearn.utils import check_array, as_float_array, deprecated
from numpy.testing import assert_array_almost_equal
from joblib import Parallel, delayed
from sklearn.model_selection import cross_val_score, RepeatedKFold

import circe.pyquic
from .inverse_covariance import (
    InverseCovarianceEstimator,
    _init_coefs,
    _compute_error,
    _validate_path,
)


def quic(
    S,
    lam,
    mode="default",
    tol=1e-6,
    max_iter=1000,
    Theta0=None,
    Sigma0=None,
    path=None,
    msg=0,
):
    """Fits the inverse covariance model according to the given training
    data and parameters.

    Parameters
    -----------
    S : 2D ndarray, shape (n_features, n_features)
        Empirical covariance or correlation matrix.

    Other parameters described in `class InverseCovariance`.

    Returns
    -------
    Theta :
    Sigma :
    opt :
    cputime :
    iters :
    dGap :
    """
    assert mode in ["default", "path", "trace"], "mode = 'default', 'path' or 'trace'."

    Sn, Sm = S.shape
    if Sn != Sm:
        raise ValueError("Input data must be square. S shape = {}".format(S.shape))
        return

    # Regularization parameter matrix L.
    if isinstance(lam, float):
        _lam = np.empty((Sn, Sm))
        _lam[:] = lam
        _lam[np.diag_indices(Sn)] = 0.  # make sure diagonal is zero
    else:
        assert lam.shape == S.shape, "lam, S shape mismatch."
        _lam = as_float_array(lam, copy=False, ensure_all_finite=False)

    # Defaults.
    optSize = 1
    iterSize = 1
    if mode == "trace":
        optSize = max_iter

    # Default Theta0, Sigma0 when both are None.
    if Theta0 is None and Sigma0 is None:
        Theta0 = np.eye(Sn)
        Sigma0 = np.eye(Sn)

    assert Theta0 is not None, "Theta0 and Sigma0 must both be None or both specified."
    assert Sigma0 is not None, "Theta0 and Sigma0 must both be None or both specified."
    assert Theta0.shape == S.shape, "Theta0, S shape mismatch."
    assert Sigma0.shape == S.shape, "Theta0, Sigma0 shape mismatch."
    Theta0 = as_float_array(Theta0, copy=False, ensure_all_finite=False)
    Sigma0 = as_float_array(Sigma0, copy=False, ensure_all_finite=False)

    if mode == "path":
        assert path is not None, "Please specify the path scaling values."

        # path must be sorted from largest to smallest and have unique values
        check_path = sorted(set(path), reverse=True)
        assert_array_almost_equal(check_path, path)

        path_len = len(path)
        optSize = path_len
        iterSize = path_len

        # Note here: memory layout is important:
        # a row of X/W holds a flattened Sn x Sn matrix,
        # one row for every element in _path_.
        Theta = np.empty((path_len, Sn * Sn))
        Theta[0, :] = Theta0.ravel()
        Sigma = np.empty((path_len, Sn * Sn))
        Sigma[0, :] = Sigma0.ravel()
    else:
        path = np.empty(1)
        path_len = len(path)

        Theta = np.empty(Theta0.shape)
        Theta[:] = Theta0
        Sigma = np.empty(Sigma0.shape)
        Sigma[:] = Sigma0

    # Cython fix for Python3
    # http://cython.readthedocs.io/en/latest/src/tutorial/strings.html
    quic_mode = mode
    if sys.version_info[0] >= 3:
        quic_mode = quic_mode.encode("utf-8")

    # Run QUIC.
    opt = np.zeros(optSize)
    cputime = np.zeros(optSize)
    dGap = np.zeros(optSize)
    iters = np.zeros(iterSize, dtype=np.uint32)
    circe.pyquic.quic(
        quic_mode,
        Sn,
        S,
        _lam,
        path_len,
        path,
        tol,
        msg,
        max_iter,
        Theta,
        Sigma,
        opt,
        cputime,
        iters,
        dGap,
    )

    if optSize == 1:
        opt = opt[0]
        cputime = cputime[0]
        dGap = dGap[0]

    if iterSize == 1:
        iters = iters[0]

    # reshape Theta, Sigma in path mode
    Theta_out = Theta
    Sigma_out = Sigma
    if mode == "path":
        Theta_out = []
        Sigma_out = []
        for lidx in range(path_len):
            Theta_out.append(np.reshape(Theta[lidx, :], (Sn, Sn)))
            Sigma_out.append(np.reshape(Sigma[lidx, :], (Sn, Sn)))

    return Theta_out, Sigma_out, opt, cputime, iters, dGap


class QuicGraphicalLasso(InverseCovarianceEstimator):
    """
    Computes a sparse inverse covariance matrix estimation using quadratic
    approximation.

    The inverse covariance is estimated the sample covariance estimate
    $S$ as an input such that:

    $T_hat = max_{\Theta} logdet(Theta) - Trace(ThetaS) - \lambda|\Theta|_1 $

    Parameters
    -----------
    lam : scalar or 2D ndarray, shape (n_features, n_features) (default=0.5)
        Regularization parameters per element of the inverse covariance matrix.

        If a scalar lambda is used, a penalty matrix will be generated
        containing lambda for all values in both upper and lower triangles
        and zeros along the diagonal.  This differs from the scalar graphical
        lasso by the diagonal. To replicate the scalar formulation you must
        manualy pass in lam * np.ones((n_features, n_features)).

    mode : one of 'default', 'path', or 'trace'
        Computation mode.

    tol : float (default=1e-6)
        Convergence threshold.

    max_iter : int (default=1000)
        Maximum number of Newton iterations.

    Theta0 : 2D ndarray, shape (n_features, n_features) (default=None)
        Initial guess for the inverse covariance matrix. If not provided, the
        diagonal identity matrix is used.

    Sigma0 : 2D ndarray, shape (n_features, n_features) (default=None)
        Initial guess for the covariance matrix. If not provided the diagonal
        identity matrix is used.

    path : array of floats (default=None)
        In "path" mode, an array of float values for scaling lam.
        The path must be sorted largest to smallest.  This class will auto sort
        this, in which case indices correspond to self.path_

    method : 'quic' or 'bigquic', ... (default=quic)
        Currently only 'quic' is supported.

    verbose : integer
        Used in quic routine.

    score_metric : one of 'log_likelihood' (default), 'frobenius', 'spectral',
                  'kl', or 'quadratic'
        Used for computing self.score().

    init_method : one of 'corrcoef', 'cov', 'spearman', 'kendalltau',
        or a custom function.
        Computes initial covariance and scales lambda appropriately.
        Using the custom function extends graphical model estimation to
        distributions beyond the multivariate Gaussian.
        The `spearman` or `kendalltau` options extend inverse covariance
        estimation to nonparanormal and transelliptic graphical models.
        Custom function must return ((n_features, n_features) ndarray, float)
        where the scalar parameter will be used to scale the penalty lam.

    auto_scale : bool
        If True, will compute self.lam_scale_ = max off-diagonal value when
        init_method='cov'.
        If false, then self.lam_scale_ = 1.
        lam_scale_ is used to scale user-supplied self.lam during fit.

    Methods
    ----------
    lam_at_index(lidx) :  Compute the scaled lambda used at index lidx.
        The parameter lidx is ignored when mode='default'.  Can use self.lam_
        for convenience in this case.

    Attributes
    ----------
    covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated covariance matrix
        If mode='path', this is 2D ndarray, shape (len(path), n_features ** 2)

    precision_ : 2D ndarray, shape (n_features, n_features)
        Estimated pseudo-inverse matrix.
        If mode='path', this is 2D ndarray, shape (len(path), n_features ** 2)

    sample_covariance_ : 2D ndarray, shape (n_features, n_features)
        Estimated sample covariance matrix

    lam_ : (float) or 2D ndarray, shape (n_features, n_features)
        When mode='default', this is the lambda used in fit (lam * lam_scale_)

    lam_scale_ : (float)
        Additional scaling factor on lambda (due to magnitude of
        sample_covariance_ values).

    path_ : None or array of floats
        Sorted (largest to smallest) path.  This will be None if not in path
        mode.

    opt_ :

    cputime_ :

    iters_ :

    duality_gap_ :

    """

    def __init__(
        self,
        lam=0.5,
        mode="default",
        tol=1e-6,
        max_iter=1000,
        Theta0=None,
        Sigma0=None,
        path=None,
        method="quic",
        verbose=0,
        score_metric="log_likelihood",
        init_method="corrcoef",
        auto_scale=True,
    ):
        # quic-specific params
        self.lam = lam
        self.mode = mode
        self.tol = tol
        self.max_iter = max_iter
        self.Theta0 = Theta0
        self.Sigma0 = Sigma0
        self.method = method
        self.verbose = verbose
        self.path = path

        if self.mode == "path" and path is None:
            raise ValueError("path required in path mode.")
            return

        super(QuicGraphicalLasso, self).__init__(
            score_metric=score_metric, init_method=init_method, auto_scale=auto_scale
        )

    def fit(self, X, y=None, **fit_params):
        """Fits the inverse covariance model according to the given training
        data and parameters.

        Parameters
        -----------
        X : 2D ndarray, shape (n_features, n_features)
            Input data.

        Returns
        -------
        self
        """

        # To satisfy sklearn 
        if y is not None and len(y) == 1:
            raise ValueError("Cannot fit with just 1 sample.")
    
        # quic-specific outputs
        self.opt_ = None
        self.cputime_ = None
        self.iters_ = None
        self.duality_gap_ = None

        # these must be updated upon self.fit()
        self.sample_covariance_ = None
        self.lam_scale_ = None
        self.is_fitted_ = False

        self.path_ = _validate_path(self.path)
        X = check_array(X, accept_sparse=True, ensure_min_features=2, estimator=self)
        X = as_float_array(X, copy=False, ensure_all_finite=False)
        self.init_coefs(X)
        if self.method == "quic":
            (
                self.precision_,
                self.covariance_,
                self.opt_,
                self.cputime_,
                self.iters_,
                self.duality_gap_,
            ) = quic(
                self.sample_covariance_,
                self.lam * self.lam_scale_,
                mode=self.mode,
                tol=self.tol,
                max_iter=self.max_iter,
                Theta0=self.Theta0,
                Sigma0=self.Sigma0,
                path=self.path_,
                msg=self.verbose,
            )
        else:
            raise NotImplementedError("Only method='quic' has been implemented.")

        self.is_fitted_ = True
        self.n_features_in_ = X.shape[1]
        return self

    def lam_at_index(self, lidx):
        """Compute the scaled lambda used at index lidx.
        """
        if self.path_ is None:
            return self.lam * self.lam_scale_

        return self.lam * self.lam_scale_ * self.path_[lidx]

    @property
    def lam_(self):
        if self.path_ is not None:
            print("lam_ is an invalid parameter in path mode, " "use self.lam_at_index")
        return self.lam_at_index(0)

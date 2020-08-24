#-------------------------------------------------------------------------------
#
#  Adapted from Scipy's KDE estimator, with copyright notice as below.
#
#  URL: https://github.com/scipy/scipy/blob/master/scipy/stats/kde.py
#
#-------------------------------------------------------------------------------
#
#  Define classes for (uni/multi)-variate kernel density estimation.
#
#  Currently, only Gaussian kernels are implemented.
#
#  Written by: Robert Kern
#
#  Date: 2004-08-09
#
#  Modified: 2005-02-10 by Robert Kern.
#              Contributed to SciPy
#            2005-10-07 by Robert Kern.
#              Some fixes to match the new scipy_core
#
#  Copyright 2004-2005 by Enthought, Inc.
#
#-------------------------------------------------------------------------------

import warnings
from scipy.special import logsumexp, ndtr
import numpy as np

__all__ = ['kde']


class kde(object):
    """Representation of a kernel-density estimate using Gaussian kernels.
    
    Kernel density estimation is a way to estimate the probability density
    function (PDF) of a random variable in a non-parametric way.
    `kde` works for both uni-variate and multi-variate data.  It includes 
    automatic bandwidth determination.  The estimation works best for a 
    unimodal distribution; bimodal or multi-modal distributions tend to be
    oversmoothed.

    Parameters
    ----------
    dataset : array_like
        Datapoints to estimate from. In case of univariate data this is a 1-D
        array, otherwise a 2-D array with shape (# of data, # of dim).
    bw_method : str, scalar or callable, optional
        The method used to calculate the estimator bandwidth.  This can be
        'scott', 'silverman', a scalar constant or a callable.  If a scalar,
        this will be used directly as `kde.factor`.  If a callable, it should
        take a `kde` instance as only parameter and return a scalar.
        If None (default), 'scott' is used.  See Notes for more details.
    weights : array_like, optional
        weights of datapoints. This must be the same shape as dataset.
        If None (default), the samples are assumed to be equally weighted
    
    Attributes
    ----------
    dataset : ndarray
        The dataset with which `kde` was initialized.
    d : int
        Number of dimensions.
    n : int
        Number of datapoints.
    neff : int
        Effective number of datapoints.
    factor : float
        The bandwidth factor, obtained from `kde.covariance_factor`, with which
        the covariance matrix is multiplied.
    covariance : ndarray
        The covariance matrix of `dataset`, scaled by the calculated bandwidth
        (`kde.factor`).
    inv_cov : ndarray
        The inverse of `covariance`.
    
    Methods
    -------
    __call__
    pdf
    logpdf
    cdf
    resample
    set_bandwidth
    covariance_factor
    
    Notes
    -----
    Bandwidth selection strongly influences the estimate obtained from the KDE
    (much more so than the actual shape of the kernel).  Bandwidth selection
    can be done by a "rule of thumb", by cross-validation, by "plug-in
    methods" or by other means; see [3]_, [4]_ for reviews.  `gaussian_kde`
    uses a rule of thumb, the default is Scott's Rule.

    Scott's Rule [1]_, implemented as `scotts_factor`, is::

        n**(-1./(d+4)),
    
    with ``n`` the number of data points and ``d`` the number of dimensions.
    In the case of unequally weighted points, `scotts_factor` becomes::

        neff**(-1./(d+4)),
    
    with ``neff`` the effective number of datapoints.
    Silverman's Rule [2]_, implemented as `silverman_factor`, is::

        (n * (d + 2) / 4.)**(-1. / (d + 4)).
    
    or in the case of unequally weighted points::

        (neff * (d + 2) / 4.)**(-1. / (d + 4)).
    
    Good general descriptions of kernel density estimation can be found in [1]_
    and [2]_, the mathematics for this multi-dimensional implementation can be
    found in [1]_.
    
    With a set of weighted samples, the effective number of datapoints ``neff``
    is defined by::

        neff = sum(weights)^2 / sum(weights^2)
    
    as detailed in [5]_.
    
    References
    ----------
    .. [1] D.W. Scott, "Multivariate Density Estimation: Theory, Practice, and
           Visualization", John Wiley & Sons, New York, Chicester, 1992.
    .. [2] B.W. Silverman, "Density Estimation for Statistics and Data
           Analysis", Vol. 26, Monographs on Statistics and Applied Probability,
           Chapman and Hall, London, 1986.
    .. [3] B.A. Turlach, "Bandwidth Selection in Kernel Density Estimation: A
           Review", CORE and Institut de Statistique, Vol. 19, pp. 1-33, 1993.
    .. [4] D.M. Bashtannyk and R.J. Hyndman, "Bandwidth selection for kernel
           conditional density estimation", Computational Statistics & Data
           Analysis, Vol. 36, pp. 279-298, 2001.
    .. [5] Gray P. G., 1969, Journal of the Royal Statistical Society.
           Series A (General), 132, 272
    
    """
    def __init__(self, dataset, bw_method=None, bw_factor=None, weights=None):
        if dataset.ndim == 1:
            self.dataset = dataset[:, np.newaxis]
        elif dataset.ndim == 2:
            self.dataset = dataset
        else:
            raise ValueError("`dataset` should be a 1-d or 2-d array.")
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.n, self.d = self.dataset.shape

        if weights is not None:
            self._weights = np.atleast_1d(weights).astype(float)
            self._weights /= np.sum(self._weights)
            if self.weights.ndim != 1:
                raise ValueError("`weights` input should be one-dimensional.")
            if len(self._weights) != self.n:
                raise ValueError("`weights` input should be of length n")
            self._neff = 1 / np.sum(self._weights**2)

        self.bw_factor = bw_factor if bw_factor is not None else 1. ##### ##### TODO: type check for this
        self.set_bandwidth(bw_method=bw_method)

    def scotts_factor(self):
        """Compute Scott's factor.
        
        Returns
        -------
        s : float
            Scott's factor.
        
        """
        return np.power(self.neff, -1. / (self.d + 4))

    def silverman_factor(self):
        """Compute the Silverman factor.
        
        Returns
        -------
        s : float
            The silverman factor.
        
        """
        return np.power(self.neff * (self.d + 2.) / 4., -1. / (self.d + 4.))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor
    covariance_factor.__doc__ = """Computes the coefficient (`kde.factor`) that
        multiplies the data covariance matrix to obtain the kernel covariance
        matrix. The default is `scotts_factor`.  A subclass can overwrite this
        method to provide a different method, or set it through a call to
        `kde.set_bandwidth`."""

    def set_bandwidth(self, bw_method=None):
        """Compute the estimator bandwidth with given method.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `kde.factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `kde.covariance_factor` method is kept.
        
        """
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, str):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif hasattr(bw_method, '__call__'):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    def _compute_covariance(self):
        """Compute the covariance matrix for each Gaussian kernel using
        covariance_factor().

        """
        self.factor = self.covariance_factor() * self.bw_factor
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            self._data_covariance = np.atleast_2d(np.cov(self.dataset, rowvar=False,
                                            bias=False, aweights=self.weights))
            self._data_inv_cov = np.linalg.inv(self._data_covariance)

        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = np.sqrt(np.linalg.det(2 * np.pi * self.covariance))

    def _diff(self, x):
        """Utility for evaluating pdf, logpdf and cdf_1d."""
        points = x[:, np.newaxis] if x.ndim == 1 else x

        m, d = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = points.reshape((self.d, 1))
                d = self.d
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        return points[np.newaxis, :, :] - self.dataset[:, np.newaxis, :]
        # (# of data, # of points, # of dim)

    def pdf(self, x):
        """Evaluate the estimated pdf on a set of points.
        
        Parameters
        ----------
        x : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.
        
        Returns
        -------
        values : (# of points,)-array
            The values at each point.
        
        Raises
        ------
        ValueError
            If the dimensionality of the input points is different than
            the dimensionality of the KDE.
        
        """
        diff = self._diff(x)
        energy = np.einsum("lmi,ij,lmj->lm", diff, self.inv_cov / 2, diff)
        result = (self.weights @ np.exp(-energy)) / self._norm_factor

        return result

    __call__ = pdf

    def logpdf(self, x):
        """Evaluate the log of the estimated pdf on a provided set of points.
        
        Parameters
        ----------
        x : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.
        
        Returns
        -------
        values : (# of points,)-array
            The values at each point.
        
        Raises
        ------
        ValueError
            If the dimensionality of the input points is different than
            the dimensionality of the KDE.
        
        """
        diff = self._diff(x)
        energy = np.einsum("lmi,ij,lmj->lm", diff, self.inv_cov / 2, diff)
        result = logsumexp(-energy.T, b=self.weights / self._norm_factor,
                           axis=1)

        return result

    def cdf(self, x):
        """Evaluate the estimated cdf on a set of 1-d points.
        
        Parameters
        ----------
        x : (# of points)-array
            Alternatively, a scalar can be passed in and
            treated as a single point.
        
        Returns
        -------
        values : (# of points,)-array
            The values at each point.
        
        Raises
        ------
        NotImplementedError
            If KDE is not 1-d.
        ValueError
            If the dimensionality of the input points is different than
            the dimensionality of the KDE.
        
        """
        if self.d != 1:
            msg = "currently only supports cdf for 1-d kde"
            raise NotImplementedError(msg)

        diff = self._diff(x)[:, :, 0]
        diff_scaled = diff / np.asscalar(self.covariance)**0.5
        cum_energy = ndtr(diff_scaled)
        result = self.weights @ cum_energy

        return result

    def resample(self, size=None):
        """Randomly sample a dataset from the estimated pdf.
        
        Parameters
        ----------
        size : int, optional
            The number of samples to draw.  If not provided, then the size is
            the same as the effective number of samples in the underlying
            dataset.
        
        Returns
        -------
        resample : (self.d, `size`) ndarray
            The sampled dataset.
        
        """
        if size is None:
            size = int(self.neff)

        norm = np.random.multivariate_normal(np.zeros(self.d),
                                             self.covariance, size=size)
        indices = np.random.choice(self.n, size=size, p=self.weights)
        means = self.dataset[indices, :]

        return means + norm

    @property
    def weights(self):
        try:
            return self._weights
        except AttributeError:
            self._weights = np.ones(self.n) / self.n
            return self._weights

    @property
    def neff(self):
        try:
            return self._neff
        except AttributeError:
            self._neff = 1 / np.sum(self.weights**2)
            return self._neff

import numpy as np
import scipy.linalg
import warnings
from ...utils.random import check_state

__all__ = ['QuadMetric', 'QuadMetricDiag', 'QuadMetricFull',
           'QuadMetricDiagAdapt', 'QuadMetricFullAdapt']

# TODO: finish docstring of QuadMetricDiag and QuadMetricFull
# TODO: implement low-rank adaptive metric?
#       https://github.com/pymc-devs/pymc3/pull/3596


class QuadMetric:
    """Base class for implementing quadratic metrics."""
    def velocity(self, x, out=None):
        raise NotImplementedError('Abstract method')

    def energy(self, x, velocity=None):
        raise NotImplementedError('Abstract method')

    def random(self, x):
        raise NotImplementedError('Abstract method')

    def velocity_energy(self, x, v_out):
        raise NotImplementedError('Abstract method')

    def update(self, sample, warmup):
        """Use a new sample during tuning to update.

        This can be used by adaptive QuadMetric to change the mass matrix.
        """
        pass

    def raise_ok(self):
        """Check if the mass matrix is ok, and raise ValueError if not.

        Raises
        ------
        ValueError if any standard deviations are 0 or infinite

        Returns
        -------
        None
        """
        return None

    def reset(self):
        pass


class QuadMetricDiag(QuadMetric):
    """
    Quadratic metric using a fixed diagonal covariance.
    
    Parameters
    ----------
    var : 1-d array_like of float
       Diagonal of the covariance matrix for the quad metric
    """
    def __init__(self, var):
        var = np.atleast_1d(var).astype(np.float)
        if var.ndim != 1:
            raise ValueError('var should be a 1-d array.')
        if not np.all(var > 0):
            raise ValueError(
                'the input diagonal covariance is not positive definite.')
        std = var ** 0.5
        self._std = std
        self._inv_std = 1. / std
        self._var = var.copy()
        self._n = len(self._var)

    def velocity(self, x, out=None):
        """Compute the velocity at the given momentum."""
        return np.multiply(self._var, x, out=out)

    def energy(self, x, velocity=None):
        """Compute the kinetic energy at the given momentum."""
        if velocity is not None:
            return 0.5 * x.dot(velocity)
        return 0.5 * x.dot(self._var * x)
    
    def random(self, random_state):
        """Draw a random value for the momentum."""
        vals = check_state(random_state).normal(size=self._n)
        return self._inv_std * vals

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at the given momentum."""
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)


class QuadMetricFull(QuadMetric):
    """
    Quadratic metric using a fixed full-rank covariance.
    
    Parameters
    ----------
    cov : 2-d array_like of float
        The covariance matrix for the quad metric
    """
    def __init__(self, cov):
        cov = np.atleast_2d(cov).astype(np.float)
        if cov.ndim != 2:
            raise ValueError('cov should be a 2-d array.')
        if not np.all(scipy.linalg.eigvalsh(cov) > 0):
            raise ValueError('the input covariance is not positive definite.')
        self._cov = np.copy(cov)
        self._chol = scipy.linalg.cholesky(self._cov, lower=True)
        self._n = len(self._cov)

    def velocity(self, x, out=None):
        """Compute the velocity at the given momentum."""
        return np.dot(self._cov, x, out=out)
    
    def energy(self, x, velocity=None):
        """Compute the kinetic energy at the given momentum."""
        if velocity is None:
            velocity = self.velocity(x)
        return 0.5 * x.dot(velocity)

    def random(self, random_state):
        """Draw a random value for the momentum."""
        vals = check_state(random_state).normal(size=self._n)
        return scipy.linalg.solve_triangular(self._chol.T, vals,
                                             overwrite_b=True)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at the given momentum."""
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)   


class QuadMetricDiagAdapt(QuadMetricDiag):
    """
    Adapt a diagonal mass matrix using the sample variances.
    
    Parameters
    ----------
    n : positive int
        The dimensionality of the problem.
    initial_mean : 1-d array_like
        Initial guess of the sample mean.
    initial_var: 1-d array_like or None, optional
        Initial guess of the sample variance. Set to ones by default.
    """
    def __init__(self, n, initial_mean, initial_var=None, initial_weight=0,
                 adaptation_window=101):
        # Set up a diagonal mass matrix
        if initial_var is not None and initial_var.ndim != 1:
            raise ValueError('Initial diagonal must be one-dimensional.')
        if initial_mean.ndim != 1:
            raise ValueError('Initial mean must be one-dimensional.')
        if initial_var is not None and len(initial_var) != n:
            raise ValueError('Wrong shape for initial_var: expected %s got %s'
                             % (n, len(initial_var)))
        if len(initial_mean) != n:
            raise ValueError('Wrong shape for initial_mean: expected %s got %s'
                             % (n, len(initial_mean)))

        if initial_var is None:
            initial_var = np.ones(n)
            initial_weight = 1

        self._n = n
        self._var = np.array(initial_var, copy=True, dtype=np.float)
        self._std = np.sqrt(initial_var)
        self._inv_std = 1. / self._std
        self._foreground_var = _WeightedVariance(
            self._n, initial_mean, initial_var, initial_weight)
        self._background_var = _WeightedVariance(self._n)
        self._n_samples = 0
        self._adaptation_window = adaptation_window

    def _update_from_weightvar(self, weightvar):
        weightvar.current_variance(out=self._var)
        np.sqrt(self._var, out=self._std)
        np.divide(1, self._std, out=self._inv_std)

    def update(self, sample, warmup):
        """Use a new sample during tuning to update."""
        if not warmup:
            return

        window = self._adaptation_window

        self._foreground_var.add_sample(sample, weight=1)
        self._background_var.add_sample(sample, weight=1)
        self._update_from_weightvar(self._foreground_var)

        if self._n_samples > 0 and self._n_samples % window == 0:
            self._foreground_var = self._background_var
            self._background_var = _WeightedVariance(self._n)

        self._n_samples += 1

    def raise_ok(self):
        """Check if the mass matrix is ok, and raise ValueError if not.

        Raises
        ------
        ValueError if any standard deviations are 0 or infinite

        Returns
        -------
        None
        """
        if np.any(self._std == 0):
            index = np.where(self._std == 0)[0]
            errmsg = ['Mass matrix contains zeros on the diagonal.']
            for ii in index:
                errmsg.append('The deviation of var {} is zero.'.format(ii))
            raise ValueError('\n'.join(errmsg))

        if np.any(~np.isfinite(self._std)):
            index = np.where(~np.isfinite(self._std))[0]
            errmsg = ['Mass matrix contains non-finite values on the diagonal.']
            for ii in index:
                errmsg.append(
                    'The deviation of var {} is non-finite.'.format(ii))
            raise ValueError('\n'.join(errmsg))


class QuadMetricFullAdapt(QuadMetricFull):
    """
    Adapt a dense mass matrix using the sample covariances.
    
    Parameters
    ----------
    n : positive int
        The dimensionality of the problem.
    initial_mean : 1-d array_like
        Initial guess of the sample mean.
    initial_cov: 2-d array_like or None, optional
        Initial guess of the sample covariance. Set to identity by default.
    
    Notes
    -----
    If the parameter `doubling` is `True`, the adaptation window is doubled
    every time it is passed. This can lead to better convergence of the mass
    matrix estimation.
    """
    def __init__(self, n, initial_mean, initial_cov=None, initial_weight=0,
                 adaptation_window=101, update_window=1, doubling=True):
        warnings.warn("QuadPotentialFullAdapt is an experimental feature")

        if initial_cov is not None and initial_cov.ndim != 2:
            raise ValueError("Initial covariance must be two-dimensional.")
        if initial_mean.ndim != 1:
            raise ValueError("Initial mean must be one-dimensional.")
        if initial_cov is not None and initial_cov.shape != (n, n):
            raise ValueError("Wrong shape for initial_cov: expected %s got %s"
                             % (n, initial_cov.shape))
        if len(initial_mean) != n:
            raise ValueError("Wrong shape for initial_mean: expected %s got %s"
                             % (n, len(initial_mean)))

        if initial_cov is None:
            initial_cov = np.eye(n)
            initial_weight = 1
        
        self._n = n
        self._cov = np.array(initial_cov, copy=True, dtype=np.float)
        self._chol = scipy.linalg.cholesky(self._cov, lower=True)
        self._chol_error = None
        self._foreground_cov = _WeightedCovariance(
            self._n, initial_mean, initial_cov, initial_weight)
        self._background_cov = _WeightedCovariance(self._n)
        self._n_samples = 0

        self._doubling = doubling
        self._adaptation_window = int(adaptation_window)
        self._update_window = int(update_window)
        self._previous_update = 0

    def _update_from_weightvar(self, weightvar):
        weightvar.current_covariance(out=self._cov)
        try:
            self._chol = scipy.linalg.cholesky(self._cov, lower=True)
        except (scipy.linalg.LinAlgError, ValueError) as error:
            self._chol_error = error

    def update(self, sample, warmup):
        """Use a new sample during tuning to update."""
        if not warmup:
            return

        # Steps since previous update
        delta = self._n_samples - self._previous_update

        self._foreground_cov.add_sample(sample, weight=1)
        self._background_cov.add_sample(sample, weight=1)

        # Update the covariance matrix and recompute the Cholesky factorization
        # every "update_window" steps
        if (delta + 1) % self._update_window == 0:
            self._update_from_weightvar(self._foreground_cov)

        # Reset the background covariance if we are at the end of the adaptation window.
        if delta >= self._adaptation_window:
            self._foreground_cov = self._background_cov
            self._background_cov = _WeightedCovariance(self._n)

            self._previous_update = self._n_samples
            if self._doubling:
                self._adaptation_window *= 2

        self._n_samples += 1

    def raise_ok(self, vmap):
        if self._chol_error is not None:
            raise ValueError("{0}".format(self._chol_error))


class _WeightedVariance:
    """Online algorithm for computing mean and variance."""
    def __init__(self, nelem, initial_mean=None, initial_variance=None,
                 initial_weight=0):
        self.n_samples = float(initial_weight)
        if initial_mean is None:
            self.mean = np.zeros(nelem)
        else:
            self.mean = np.array(initial_mean, copy=True)
        if initial_variance is None:
            self.raw_var = np.zeros(nelem)
        else:
            self.raw_var = np.array(initial_variance, copy=True)

        self.raw_var[:] *= self.n_samples

        if self.raw_var.shape != (nelem,):
            raise ValueError('Invalid shape for initial variance.')
        if self.mean.shape != (nelem,):
            raise ValueError('Invalid shape for initial mean.')

    def add_sample(self, x, weight):
        x = np.asarray(x)
        self.n_samples += 1
        old_diff = x - self.mean
        self.mean[:] += old_diff / self.n_samples
        new_diff = x - self.mean
        self.raw_var[:] +=  weight * old_diff * new_diff

    def current_variance(self, out=None):
        if self.n_samples == 0:
            raise ValueError('Can not compute variance without samples.')
        if out is not None:
            return np.divide(self.raw_var, self.n_samples, out=out)
        else:
            return self.raw_var / self.n_samples

    def current_mean(self):
        return self.mean.copy()


class _WeightedCovariance:
    """Online algorithm for computing mean and covariance.
    This implements the `Welford's algorithm
    <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_ based
    on the implementation in `the Stan math library
    <https://github.com/stan-dev/math>`_.
    """
    def __init__(self, nelem, initial_mean=None, initial_covariance=None,
                 initial_weight=0):
        self.n_samples = float(initial_weight)
        if initial_mean is None:
            self.mean = np.zeros(nelem)
        else:
            self.mean = np.array(initial_mean, copy=True)
        if initial_covariance is None:
            self.raw_cov = np.eye(nelem)
        else:
            self.raw_cov = np.array(initial_covariance, copy=True)

        self.raw_cov[:] *= self.n_samples

        if self.raw_cov.shape != (nelem, nelem):
            raise ValueError("Invalid shape for initial covariance.")
        if self.mean.shape != (nelem,):
            raise ValueError("Invalid shape for initial mean.")

    def add_sample(self, x, weight):
        x = np.asarray(x)
        self.n_samples += 1
        old_diff = x - self.mean
        self.mean[:] += old_diff / self.n_samples
        new_diff = x - self.mean
        self.raw_cov[:] += weight * new_diff[:, None] * old_diff[None, :]

    def current_covariance(self, out=None):
        if self.n_samples == 0:
            raise ValueError("Can not compute covariance without samples.")
        if out is not None:
            return np.divide(self.raw_cov, self.n_samples - 1, out=out)
        else:
            return self.raw_cov / (self.n_samples - 1)

    def current_mean(self):
        return self.mean.copy()

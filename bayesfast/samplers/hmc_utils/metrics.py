import numpy as np
import scipy.linalg
from ...utils.random_utils import check_state

__all__ = ['QuadMetric', 'QuadMetricDiag', 'QuadMetricFull', 
           'QuadMetricDiagAdapt']


class QuadMetric:
    
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
    """Quad metric using a fixed diagonal covariance."""

    def __init__(self, v):
        """Use a vector to represent a diagonal matrix for the covariance.

        Parameters
        ----------
        v : vector, 0 <= ndim <= 1
           Diagonal of the covariance matrix for the quad metric
        """
        v = np.atleast_1d(v)
        if v.ndim != 1:
            raise ValueError('v should be a 1-d array.')
        if not np.all(v > 0):
            raise ValueError(
                'the input diagonal covariance is not positive definite.')
        s = v ** .5
        self.s = s
        self.inv_s = 1. / s
        self.v = v.copy()

    def velocity(self, x, out=None):
        """Compute the velocity at the given momentum."""
        if out is not None:
            np.multiply(x, self.v, out=out)
            return
        return self.v * x

    def energy(self, x, velocity=None):
        """Compute the kinetic energy at the given momentum."""
        if velocity is not None:
            return 0.5 * np.dot(x, velocity)
        return .5 * x.dot(self.v * x)
    
    def random(self, random_state):
        """Draw a random value for the momentum."""
        return check_state(random_state).normal(size=self.s.shape) * self.inv_s

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at the given momentum."""
        np.multiply(x, self.v, out=v_out)
        return 0.5 * np.dot(x, v_out)


class QuadMetricFull(QuadMetric):
    """Quad metric using a fixed full-rank covariance."""

    def __init__(self, cov):
        """Compute the lower cholesky decomposition of the covariance.

        Parameters
        ----------
        cov : matrix, ndim = 2
            The covariance matrix for the quad metric
        """
        cov = np.atleast_2d(cov)
        if cov.ndim != 2:
            raise ValueError('cov should be a 2-d array.')
        if not np.all(scipy.linalg.eigvalsh(cov) > 0):
            raise ValueError('the input covariance is not positive definite.')
        self.cov = np.copy(cov)
        self.L = scipy.linalg.cholesky(cov, lower=True)

    def velocity(self, x, out=None):
        """Compute the velocity at the given momentum."""
        return np.dot(self.cov, x, out=out)
    
    def energy(self, x, velocity=None):
        """Compute the kinetic energy at the given momentum."""
        if velocity is None:
            velocity = self.velocity(x)
        return .5 * x.dot(velocity)

    def random(self, random_state):
        """Draw a random value for the momentum."""
        n = check_state(random_state).normal(size=self.L.shape[0])
        return scipy.linalg.solve_triangular(self.L.T, n)

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at the given momentum."""
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)


class QuadMetricDiagAdapt(QuadMetric):
    """Adapt a diagonal mass matrix from the sample variances."""

    def __init__(self, n, initial_mean, initial_diag=None, initial_weight=0,
                 adaptation_window=101):
        """Set up a diagonal mass matrix."""
        if initial_diag is not None and initial_diag.ndim != 1:
            raise ValueError('Initial diagonal must be one-dimensional.')
        if initial_mean.ndim != 1:
            raise ValueError('Initial mean must be one-dimensional.')
        if initial_diag is not None and len(initial_diag) != n:
            raise ValueError('Wrong shape for initial_diag: expected %s got %s'
                             % (n, len(initial_diag)))
        if len(initial_mean) != n:
            raise ValueError('Wrong shape for initial_mean: expected %s got %s'
                             % (n, len(initial_mean)))

        if initial_diag is None:
            initial_diag = np.ones(n)
            initial_weight = 1

        self._n = n
        self._var = np.array(initial_diag, copy=True)
        self._stds = np.sqrt(initial_diag)
        self._inv_stds = 1. / self._stds
        self._foreground_var = _WeightedVariance(
            self._n, initial_mean, initial_diag, initial_weight)
        self._background_var = _WeightedVariance(self._n)
        self._n_samples = 0
        self.adaptation_window = adaptation_window

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
        return self._inv_stds * vals

    def velocity_energy(self, x, v_out):
        """Compute velocity and return kinetic energy at the given momentum."""
        self.velocity(x, out=v_out)
        return 0.5 * np.dot(x, v_out)

    def _update_from_weightvar(self, weightvar):
        weightvar.current_variance(out=self._var)
        np.sqrt(self._var, out=self._stds)
        np.divide(1, self._stds, out=self._inv_stds)

    def update(self, sample, warmup):
        """Use a new sample during tuning to update."""
        if not warmup:
            return

        window = self.adaptation_window

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
        if np.any(self._stds == 0):
            index = np.where(self._stds == 0)[0]
            errmsg = ['Mass matrix contains zeros on the diagonal.']
            for ii in index:
                errmsg.append('The deviation of var {} is zero.'.format(ii))
            raise ValueError('\n'.join(errmsg))

        if np.any(~np.isfinite(self._stds)):
            index = np.where(~np.isfinite(self._stds))[0]
            errmsg = ['Mass matrix contains non-finite values on the diagonal.']
            for ii in index:
                errmsg.append(
                    'The deviation of var {} is non-finite.'.format(ii))
            raise ValueError('\n'.join(errmsg))


class _WeightedVariance:
    """Online algorithm for computing mean of variance."""

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

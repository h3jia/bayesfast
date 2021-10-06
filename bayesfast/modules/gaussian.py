from ..core.module import ModuleBase
import numpy as np
from scipy.stats import multivariate_normal, mvn, norm

__all__ = ['Gaussian']


class Gaussian(ModuleBase):
    """
    A univariate or multivariate Gaussian distribution.
    
    Parameters
    ----------
    mean : 1-d array_like of float
        The mean of the Gaussian.
    cov : 1-d or 2-d array_like of float
        If 1-d, will be interpreted as the diagonal of the covariance matrix.
        If 2-d, will be interpreted as the full covariance matrix.
    input_vars : str or 1-d array_like of str, optional
        Name(s) of input variable(s). Will first be concatenated as one single
        variable. Set to ``'__var__'`` by default.
    output_vars : str or 1-d array_like of str, optional
        Name of output variable. Should contain only 1 variable here. Set to
        ``'__var__'`` by default.
    delete_vars : str or 1-d array_like of str, optional
        Name(s) of variable(s) to be deleted from the dict during runtime. Set
        to ``()`` by default.
    lower : 1-d array_like of float, or None, optional
        If not None, will be used to compute the correct normalization of a
        truncated Gaussian. Set to ``None`` by default.
    upper : 1-d array_like of float, or None, optional
        If not None, will be used to compute the correct normalization of a
        truncated Gaussian. Set to ``None`` by default.
    label : str or None, optional
        The label of the module used in ``print_summary``. Set to ``None`` by
        default.
    """
    def __init__(self, mean, cov, input_vars='__var__', output_vars='__var__',
                 delete_vars=(), lower=None, upper=None, label=None):
        self.mean = mean
        self.cov = cov
        self.lower = lower
        self.upper = upper
        super().__init__(
            input_vars=input_vars, output_vars=output_vars,
            delete_vars=delete_vars, input_shapes=-1, output_shapes=None,
            input_scales=None, label=label)

    _input_min_length = 1

    _input_max_length = np.inf

    _output_min_length = 1

    _output_max_length = 1

    def _reset_norm(self):
        self._norm_0 = None
        self._norm_1 = None

    def _compute_norm(self):
        try:
            dim = self.mean.shape[0]
            assert self.mean.shape == (dim,)
            assert self.lower is None or self.lower.shape == (dim,)
            assert self.upper is None or self.upper.shape == (dim,)
            assert self.cov.shape == (dim, dim)
            if self.lower is None:
                lower = np.full_like(self.mean, -np.inf)
            else:
                lower = self.lower
            if self.upper is None:
                upper = np.full_like(self.mean, np.inf)
            else:
                upper = self.upper
            assert np.all(lower <= upper)
            if self._var is None:
                self._norm_0 = multivariate_normal.logpdf(
                    x=self.mean, mean=self.mean, cov=self.cov)
                self._norm_1 = -np.log(
                    mvn.mvnun(lower, upper, self.mean, self.cov)[0])
            else:
                self._norm_0 = np.sum(norm.logpdf(
                    x=self.mean, loc=self.mean, scale=np.sqrt(self._var)))
                cdf_1 = norm.cdf(x=upper, loc=self.mean,
                                 scale=np.sqrt(self._var))
                cdf_0 = norm.cdf(x=lower, loc=self.mean,
                                 scale=np.sqrt(self._var))
                self._norm_1 = -np.sum(np.log(cdf_1 - cdf_0))
        except:
            self._reset_norm()
            raise

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, m):
        try:
            m = np.atleast_1d(m)
            assert m.ndim == 1
            self._mean = m
            self._mean.flags.writeable = False # TODO: PropertyArray?
        except Exception:
            raise ValueError('invalid value for mean.')
        self._reset_norm()

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, c):
        try:
            c = np.atleast_1d(c)
            assert c.ndim == 1 or c.ndim == 2
            if c.ndim == 2:
                assert c.shape[0] == c.shape[1]
                self._cov = c
                self._cov_inv = np.linalg.inv(c)
                self._var = None
                self._var_inv = None
            elif c.ndim == 1:
                self._var = c
                self._var_inv = 1 / c
                self._cov = np.diag(self._var)
                self._cov_inv = np.diag(self._var_inv)
                self._var.flags.writeable = False # TODO: PropertyArray?
                self._var_inv.flags.writeable = False # TODO: PropertyArray?
            self._cov.flags.writeable = False # TODO: PropertyArray?
            self._cov_inv.flags.writeable = False # TODO: PropertyArray?
        except Exception:
            raise ValueError('invalid value for cov.')
        self._reset_norm()

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, l):
        if l is None:
            self._lower = None
        else:
            try:
                l = np.atleast_1d(l)
                assert l.ndim == 1
                self._lower = l
                self._lower.flags.writeable = False # TODO: PropertyArray?
            except Exception:
                raise ValueError('invalid value for lower.')
        self._reset_norm()

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, u):
        if u is None:
            self._upper = None
        else:
            try:
                u = np.atleast_1d(u)
                assert u.ndim == 1
                self._upper = u
                self._upper.flags.writeable = False # TODO: PropertyArray?
            except Exception:
                raise ValueError('invalid value for upper.')
        self._reset_norm()

    def _input_check(self, x):
        if self._norm_0 is None or self._norm_1 is None:
            self._compute_norm()
        x = np.asarray(x)
        assert x.shape[-1] == self.mean.shape[0]
        return x

    def _fun(self, x):
        x = self._input_check(x)
        delta = x - self.mean
        if self._var_inv is None:
            dcd = np.einsum('...i,ij,...j->...', delta, self._cov_inv, delta)
        else:
            dcd = np.einsum('...i,i,...i->...', delta, self._var_inv, delta)
        return -0.5 * dcd + self._norm_0 + self._norm_1

    def _jac(self, x):
        x = self._input_check(x)
        if self._var_inv is None:
            return -np.einsum('...i,ij->...j', x - self.mean, self._cov_inv)
        else:
            return -np.einsum('...i,i->...i', x - self.mean, self._var_inv)

    def _fun_and_jac(self, x):
        x = self._input_check(x)
        delta = x - self.mean
        if self._var_inv is None:
            j = -np.einsum('...i,ij->...j', delta, self._cov_inv)
        else:
            j = -np.einsum('...i,i->...i', delta, self._var_inv)
        f = 0.5 * np.einsum('...i,...i->...', delta, j)
        return f + self._norm_0 + self._norm_1, j

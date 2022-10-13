from ..core.surrogate import Surrogate
from ..core.density import ModuleCache
from ._poly import *
import numpy as np
import jax.numpy as jnp
from jax import jit
from jax.lax import cond
from scipy.linalg import lstsq

__all__ = ['PolyConfig']


_all_order = ['linear', 'quadratic', 'cubic-2', 'cubic-3']


class PolyConfig:
    """
    Configuring a basic PolyModel block with a certain polynomial order.

    Parameters
    ----------
    order : str or 1-d array_like of str
        Specifying the order of the polynomial. Should be one of ``'linear'``, ``'quadratic'``,
        ``'cubic-2'`` and ``'cubic-3'``, or a 1-d array_like of them, which will initialize a list
        of PolyConfig(s) with the corresponding order(s). Can also have prefix of ``'<'`` or
        ``'<='``, like ``'<=quadratic'``, which will be interpreted as ``['linear', 'quadratic']``.
    input_indices : int or 1-d array_like of int
        The indices activated for the input of the polymonial. If a builtin int is given, will be
        interpreted as ``np.arange(input_indices)``. Please do not use wrap around indices.
    output_indices : int or 1-d array_like of int
        The indices to which the polynomial output is written. If a builtin int is given, will be
        interpreted as ``np.arange(output_indices)``. Please do not use wrap around indices.
    """
    def __new__(cls, order, input_indices, output_indices):
        if isinstance(order, str):
            if order[0] == '<':
                # return a list of PolyConfig(s), with the highest order determined below
                if order[1] == '=':
                    if order[2:] == 'linear':
                        o_max = 1
                    elif order[2:] == 'quadratic':
                        o_max = 2
                    elif order[2:] == 'cubic-2':
                        o_max = 3
                    elif order[2:] == 'cubic-3':
                        o_max = 4
                    else:
                        raise ValueError('invalid value for order.')
                elif order[1:] == 'quadratic':
                    o_max = 1
                elif order[1:] == 'cubic-2':
                    o_max = 2
                elif order[1:] == 'cubic-3':
                    o_max = 3
                else:
                    raise ValueError('invalid value for order.')
                order = _all_order[:o_max]
            else:
                # return a single PolyConfig
                return super(PolyConfig, cls).__new__(cls)
        if hasattr(order, '__iter__'):
            return [cls(oi, input_indices, output_indices) for oi in order]
        else:
            raise ValueError('invalid value for order.')

    def __init__(self, order, input_indices, output_indices):
        self._set_order(order)
        self._set_input_indices(input_indices)
        self._set_output_indices(output_indices)
        self._clear_coef()

    def __getnewargs__(self):
        # https://stackoverflow.com/questions/59828469/python-multiprocessing-pool-map-causes-error-in-new
        return self.order, self.input_indices, self.output_indices

    def _clear_coef(self):
        self._coef = np.zeros(self._A_shape)

    @property
    def order(self):
        """
        The order of this polynomial.
        """
        return self._order

    def _set_order(self, od):
        if od in ('linear', 'quadratic', 'cubic-2', 'cubic-3'):
            self._order = od
        else:
            raise ValueError('order should be one of ("linear", "quadratic", "cubic-2", "cubic-3"),'
                             ' instead of "{}".'.format(od))

    @property
    def input_indices(self):
        """
        The indices activated for the input of the polymonial.
        """
        return self._input_indices

    def _set_input_indices(self, ii):
        if isinstance(ii, int):
            self._input_indices = np.arange(ii)
        else:
            ii = np.asarray(ii, dtype=int).reshape(-1)
            if ii.size != np.unique(ii).size:
                raise ValueError('the elements in input_indices should be unique.')
            self._input_indices = ii

    @property
    def output_indices(self):
        """
        The indices to which the polynomial output is written.
        """
        return self._output_indices

    def _set_output_indices(self, oi):
        if isinstance(oi, int):
            self._output_indices = np.arange(oi)
        else:
            oi = np.asarray(oi, dtype=int).reshape(-1)
            if oi.size != np.uniquei(oi).size:
                raise ValueError('the elements in output_indices should be unique.')
            self._output_indices = oi

    @property
    def input_size(self):
        """
        The size of input indices.
        """
        return self._input_indices.size

    @property
    def output_size(self):
        """
        The size of output indices.
        """
        return self._output_indices.size

    @property
    def _A_shape(self):
        """
        The shape of all the coefficients. Note that not all the elements are independent.
        It's organized for the convenience of polynomial evaluation.
        """
        if self._order == 'linear':
            return (self.output_size, self.input_size + 1)
        elif self._order == 'quadratic':
            return (self.output_size, self.input_size, self.input_size)
        elif self._order == 'cubic-2':
            return (self.output_size, self.input_size, self.input_size)
        elif self._order == 'cubic-3':
            return (self.output_size, self.input_size, self.input_size, self.input_size)
        else:
            raise RuntimeError('unexpected value "{}" for self.order.'.format(self._order))

    @property
    def _a_shape(self):
        """
        The shape of all independent coefficients for one single output vairable.
        This would be the shape of lstsq regression.
        """
        if self._order == 'linear':
            return (self.input_size + 1,)
        elif self._order == 'quadratic':
            return (self.input_size * (self.input_size + 1) // 2,)
        elif self._order == 'cubic-2':
            return (self.input_size * self.input_size,)
        elif self._order == 'cubic-3':
            return (self.input_size * (self.input_size - 1) * (self.input_size - 2) // 6,)
        else:
            raise RuntimeError('unexpected value "{}" for self.order.'.format(self._order))

    def _set(self, a, i):
        """
        Set the polynomial coefficients based on the lstsq regression result.
        """
        a = np.ascontiguousarray(a)
        i = int(i)
        if a.shape != self._a_shape:
            raise ValueError('shape of a {} does not match the expected shape {}.'.format(a.shape,
                             self._a_shape))
        if not 0 <= i < self.output_size:
            raise ValueError('i = {} out of range for self.output_size = {}.'.format(i,
                             self.output_size))
        if self._order == 'linear':
            coefi = a
        else:
            coefi = np.zeros(self._A_shape[1:])
            if self._order == 'quadratic':
                _set_quadratic(a, coefi, self.input_size)
            elif self._order == 'cubic-2':
                _set_cubic_2(a, coefi, self.input_size)
            elif self._order == 'cubic-3':
                _set_cubic_3(a, coefi, self.input_size)
            else:
                raise RuntimeError('unexpected value "{}" for self.order.'.format(self._order))
        self._coef[i] = coefi


class PolyModel(Surrogate):
    """
    Polynomial surrogate model, currently up to cubic order.

    Parameters
    ----------
    configs : PolyConfig or 1-d array_like of PolyConfig
        Configuring the PolyModel.
    use_jit : bool, optional
        Whether to apply jit with jax. If True, the model will be jitted once it's fitted with data.
        Set to ``True`` by default.
    use_trust : bool, optional
        Whether to define a trust region and do linear extrapolation outside it. Set to ``True`` by
        default.
    chi2_a : float or None, optional
        The absolute value of the max chi2 that defines the trust region. Will be superseded if
        ``chi2_r`` is not None. Set to ``None`` by default.
    chi2_r : float or None, optional
        The relative value of the max chi2 that defines the trust region. Set to ``1.0`` by default.
    center_max : bool, optional
        Whether to use the model evaluated at the max logp as the central value for extrapolation.
        Set to ``True`` by default.
    """
    def __init__(self, configs, use_jit=True, use_trust=True, chi2_a=None, chi2_r=1.0,
                 center_max=True):
        self._set_configs(configs)
        self._use_jit = bool(use_jit)
        self._set_trust(use_trust, chi2_a, chi2_r, center_max)
        self._build_recipe()
        self._ready = False

    @property
    def configs(self):
        """
        A list of basic PolyModel blocks, each with a certain polynomial order.
        """
        return self._configs

    @property
    def n_config(self):
        """
        The number of basic PolyModel blocks.
        """
        return len(self._configs)

    @property
    def use_jit(self):
        """
        Whether to apply jit with jax.
        """
        return self._use_jit

    @property
    def use_trust(self):
        """
        Whether to define a trust region and do linear extrapolation outside it.
        """
        return self._use_trust

    @property
    def chi2_a(self):
        """
        The absolute value of the max chi2 that defines the trust region.
        """
        return self._chi2_a

    @property
    def chi2_r(self):
        """
        The relative value of the max chi2 that defines the trust region.
        """
        return self._chi2_r

    @property
    def center_max(self):
        """
        Whether to use the model evaluated at the max logp as the central value for extrapolation.
        """
        return self._center_max

    def _set_configs(self, configs):
        if isinstance(configs, PolyConfig):
            configs = [configs]
        if (hasattr(configs, '__iter__') and len(configs) > 0 and
           all(isinstance(cf, PolyConfig) for cf in configs)):
            self._configs = list(configs)
        else:
            raise ValueError('invalid value for configs.')

    def _set_trust(self, use_trust, chi2_a, chi2_r, center_max):
        self._use_trust = bool(use_trust)
        if chi2_a is None:
            self._chi2_a = None
        else:
            try:
                self._chi2_a = float(chi2_a)
            except Exception:
                raise ValueError('invalid value for chi2_a.')
        if chi2_r is None:
            if use_trust and chi2_a is None:
                raise ValueError('if use_trust, chi2_a and chi2_r cannot both be None.')
            self._chi2_r = None
        else:
            try:
                self._chi2_r = float(chi2_r)
            except Exception:
                raise ValueError('invalid value for chi2_r.')
        self._center_max = bool(center_max)

    def _build_recipe(self):
        self._input_size = np.max([np.max(cf.input_size) for cf in self._configs])
        self._output_size = np.max([np.max(cf.output_size) for cf in self._configs])
        r = np.full((self._output_size, 4), -1)
        # r has shape (# of output variables, 4), if r[i, j] = k for j = (0, 1, 2, 3)
        # it means for the i-th output variables, the (linear, quadratic, cubic-2, cubic-3) term
        # comes from the k-th PolyConfig
        for i, conf in enumerate(self._configs):
            if conf.order == 'linear':
                if np.any(r[conf.output_indices, 0] >= 0):
                    raise ValueError('multiple linear PolyConfig(s) share at least one common '
                                     'output variable. Please check PolyConfig #{}.'.format(i))
                r[conf.output_indices, 0] = i
            elif conf.order == 'quadratic':
                if np.any(r[conf.output_indices, 1] >= 0):
                    raise ValueError('multiple quadratic PolyConfig(s) share at least one common '
                                     'output variable. Please check PolyConfig #{}.'.format(i))
                r[conf.output_indices, 1] = i
            elif conf.order == 'cubic-2':
                if np.any(r[conf.output_indices, 2] >= 0):
                    raise ValueError('multiple cubic_2 PolyConfig(s) share at least one common '
                                     'output variable. Please check PolyConfig #{}.'.format(i))
                r[conf.output_indices, 2] = i
            elif conf.order == 'cubic-3':
                if np.any(r[conf.output_indices, 3] >= 0):
                    raise ValueError('multiple cubic_3 PolyConfig(s) share at least one common '
                                     'output variable. Please check PolyConfig #{}.'.format(i))
                r[conf.output_indices, 3] = i
            else:
                raise RuntimeError('unexpected value of conf.order for PolyConfig #{}.'.format(i))
        if np.any(np.all(r < 0, axis=1)):
            raise ValueError('no PolyConfig has output for variable(s) '
                             '{}.'.format(np.argwhere(np.any(np.all(r < 0, axis=1))).flatten()))
        self._recipe = r

    def forward(self, *args, **kwargs):
        """
        Evaluate the PolyModel.
        """
        if not self._ready:
            raise RuntimeError('this PolyModel has not been fitted yet.')
        return self._eval(*args, **kwargs)

    def _eval(self, *args, **kwargs):
        x = self.f_pre(*args, **kwargs)
        if self.use_trust:
            y = cond(jnp.einsum('j,jk,k->', x - self._mu, self._hess, x - self._mu) <= self._chi2,
                     self.f_poly, self.f_poly_ex, x)
        else:
            y = self.f_poly(x)
        return self.f_post(y)

    def f_pre(self, x):
        """
        Utility to concat the raw input as a single 1-d array.
        """
        return x

    def f_poly(self, x):
        """
        Evaluate the polynomial(s).
        """
        f = jnp.zeros(self._output_size)
        for conf in self._configs:
            if conf.order == 'linear':
                f = f.at[conf.output_indices].add(self._linear(conf, x[conf.input_indices]))
            elif conf.order == 'quadratic':
                f = f.at[conf.output_indices].add(self._quadratic(conf, x[conf.input_indices]))
            elif conf.order == 'cubic-2':
                f = f.at[conf.output_indices].add(self._cubic_2(conf, x[conf.input_indices]))
            elif conf.order == 'cubic-3':
                f = f.at[conf.output_indices].add(self._cubic_3(conf, x[conf.input_indices]))
            else:
                raise RuntimeError('unexpected value for conf.order.')
        return f

    def f_poly_ex(self, x):
        """
        Evaluate the polynomial(s) outside the trust region with linear extrapolation.
        """
        chi = jnp.einsum('j,jk,k->', x - self._mu, self._hess, x - self._mu)**0.5
        x0 = (self._chi * x + (chi - self._chi) * self._mu) / chi
        f_x0 = self.f_poly(x0)
        return (chi * f_x0 - (chi - self._chi) * self._f_mu) / self._chi

    def f_post(self, y):
        """
        Utility to split the output into several variables.
        """
        return y

    def fi_post(self, z):
        """
        Inverse of f_post, used for fitting.
        """
        return z

    @staticmethod
    def _linear(config, x):
        """
        Evaluate a linear block.
        """
        return jnp.dot(config._coef[:, 1:], x) + config._coef[:, 0]

    @staticmethod
    def _quadratic(config, x):
        """
        Evaluate a quadratic block.
        """
        # TODO: rewrite this in a more efficient way to avoid the 0 elements
        return jnp.einsum('ijk,j,k->i', config._coef, x, x)

    @staticmethod
    def _cubic_2(config, x):
        """
        Evaluate a cubic-2 block.
        """
        return jnp.einsum('ijk,j,k->i', config._coef, x * x, x)

    @staticmethod
    def _cubic_3(cls, config, x):
        """
        Evaluate a cubic-3 block.
        """
        # TODO: rewrite this in a more efficient way to avoid the 0 elements
        return jnp.einsum('ijkl,j,k,l->i', config._coef, x, x, x)

    def fit(self, module_caches, add_dicts=(), logp_key=None, f_w=None):
        x = np.array([self.f_pre(*mc.args, **mc.kwargs) for mc in module_caches], dtype=np.float64)
        y = np.array([self.fi_post(mc.returns) for mc in module_caches], dtype=np.float64)
        if not (x.ndim == 2 and x.shape[-1] == self._input_size):
            raise ValueError('x should be a 2-d array, with shape (# of points, # of input_size), '
                             'instead of {}.'.format(x.shape))
        if not (y.ndim == 2 and y.shape[-1] == self._output_size):
            raise ValueError('y should be a 2-d array, with shape (# of points, # of output_size), '
                             'instead of {}.'.format(y.shape))
        if not x.shape[0] == y.shape[0]:
            raise ValueError('x and y have different # of points.')
        if x.shape[0] < self.n_param:
            raise ValueError('I need at least {} points, but you only gave me '
                             '{}.'.format(self.n_param, x.shape[0]))

        if logp_key is not None:
            logp = np.array([ad[logp_key] for ad in add_dicts]).reshape(-1)
            if not x.shape[0] == logp.shape[0]:
                raise ValueError('x and logp have inconsistent shapes.')
        else:
            if self.use_trust and self.center_max:
                raise ValueError('logp_key must be given if self.use_trust and self.center_max are '
                                 'both True.')
            logp = None
        if f_w is None:
            w = None
        else:
            w = np.asarray(f_w(x=x, y=y, logp=logp, module_caches=module_caches,
                           add_dicts=add_dicts), dtype=np.float64)
            if not (w.ndim == 1 and w.shape[0] == x.shape[0]):
                raise ValueError('invalid shape for w.')

        for c in self.configs:
            c._clear_coef()

        for i in range(self._output_size):
            A = np.empty((x.shape[0], 0))
            j_l, j_q, j_c2, j_c3 = self._recipe[i]
            k = [0]
            if j_l >= 0:
                _A = np.empty((x.shape[0], self.configs[j_l]._a_shape[0]))
                _x = np.ascontiguousarray(x[..., self.configs[j_l].input_indices])
                _A[:, 0] = 1
                _A[:, 1:] = _x
                k.append(k[-1] + self.configs[j_l]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            if j_q >= 0:
                _A = np.empty((x.shape[0], self.configs[j_q]._a_shape[0]))
                _x = np.ascontiguousarray(x[..., self.configs[j_q].input_indices])
                _lsq_quadratic(_x, _A, x.shape[0], self.configs[j_q].input_size)
                k.append(k[-1] + self.configs[j_q]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            if j_c2 >= 0:
                _A = np.empty((x.shape[0], self.configs[j_c2]._a_shape[0]))
                _x = np.ascontiguousarray(x[..., self.configs[j_c2].input_indices])
                _lsq_cubic_2(_x, _A, x.shape[0], self.configs[j_c2].input_size)
                k.append(k[-1] + self.configs[j_c2]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            if j_c3 >= 0:
                _A = np.empty((x.shape[0], self.configs[j_c3]._a_shape[0]))
                _x = np.ascontiguousarray(x[..., self.configs[j_c3].input_indices])
                _lsq_cubic_3(_x, _A, x.shape[0], self.configs[j_c3].input_size)
                k.append(k[-1] + self.configs[j_c3]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            b = np.copy(y[:, i])
            if w is not None:
                b *= w
                A *= w[:, np.newaxis]

            # np.savez('/global/homes/h/hejia/debug-poly.npz', A=A, b=b)
            lsq = lstsq(A, b)[0]
            p = 0
            if j_l >= 0:
                q = int(np.argwhere(self.configs[j_l].output_indices == i))
                self.configs[j_l]._set(lsq[k[p]:k[p + 1]], q)
                p += 1
            if j_q >= 0:
                q = int(np.argwhere(self.configs[j_q].output_indices == i))
                self.configs[j_q]._set(lsq[k[p]:k[p + 1]], q)
                p += 1
            if j_c2 >= 0:
                q = int(np.argwhere(self.configs[j_c2].output_indices == i))
                self.configs[j_c2]._set(lsq[k[p]:k[p + 1]], q)
                p += 1
            if j_c3 >= 0:
                q = int(np.argwhere(self.configs[j_c3].output_indices == i))
                self.configs[j_c3]._set(lsq[k[p]:k[p + 1]], q)
                p += 1

        # for c in self.configs:
        #     c._coef = jnp.asarray(c._coef)

        if self.use_trust:
            self._mu = jnp.mean(x, axis=0)
            self._hess = jnp.linalg.inv(jnp.atleast_2d(jnp.cov(x, rowvar=False)))
            # atleast_2d so it also works for 1-dim problems
            if isinstance(self.chi2_r, float):
                chi2_all = jnp.einsum('ij,jk,ik->i', x - self._mu, self._hess, x - self._mu)
                if 0. <= self.chi2_r <= 1.:
                    self._chi2 = jnp.quantile(chi2_all, self.chi2_r)
                else:
                    chi2_min = jnp.min(chi2_all)
                    chi2_max = jnp.max(chi2_all)
                    self._chi2 = (chi2_max - chi2_min) * self.chi2_r + chi2_min
            else:
                if not isinstance(self.chi2_a, float):
                    raise RuntimeError('neither self.chi2_r nor chi2_a is a float.')
                self._chi2 = self.chi2_a
            assert self._chi2 >= 0.
            self._chi = self._chi2**0.5
            if self.center_max:
                mu_f = x[jnp.argmax(logp)]
            else:
                mu_f = self._mu
            self._f_mu = self.f_poly(mu_f)

        if self.use_jit:
            self.apply_jit()
        self._ready = True

    @property
    def n_param(self):
        """
        The number of free parameters. Required to determine the number of fitting samples.
        """
        return np.sum([conf._a_shape[0] for conf in self.configs])

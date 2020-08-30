import copy
from ..core.module import *
from ..core.density import *
from ._poly import *
import numpy as np
from scipy.linalg import lstsq
from collections import namedtuple

__all__ = ['PolyConfig', 'PolyModel']

# TODO: check the fit mechanism


BoundOptions = namedtuple('BoundOptions',
                          ('use_bound', 'alpha', 'alpha_p', 'center_max'))


class PolyConfig:
    """
    Configuring the PolyModel.
    
    Parameters
    ----------
    order : str
        Specifying the order of the polynomial model. Should be one of 'linear',
        'quadratic', 'cubic-2' and 'cubic-3'.
    input_mask : None or 1-d array_like, optional
        The indice of input variables that are activated. If `None`, will be
        understood as np.arange(input_size), i.e. all the variables are
        activated. Set to `None` by default.
    output_mask : None or 1-d array_like, optional
        The indice of output variables that are activated. If `None`, will be
        understood as np.arange(output_size), i.e. all the variables are
        activated. Set to `None` by default.
    """
    def __init__(self, order, input_mask=None, output_mask=None):
        if order in ('linear', 'quadratic', 'cubic-2', 'cubic-3'):
            self._order = order
        else:
            raise ValueError(
                'order should be one of ("linear", "quadratic", "cubic-2", '
                '"cubic-3"), instead of "{}".'.format(order))
        self._set_input_mask(input_mask)
        self._set_output_mask(output_mask)
        self._coef = None

    @property
    def order(self):
        return self._order

    @property
    def input_mask(self):
        return self._input_mask

    def _set_input_mask(self, im):
        if im is None:
            self._input_mask = None
        else:
            self._input_mask = np.sort(np.unique(np.asarray(im, dtype=np.int)))
            # we do not allow directly modify the elements of input_mask here
            # as it cannot trigger the unique/sort check
            self._input_mask.flags.writeable = False # TODO: PropertyArray?

    @property
    def output_mask(self):
        return self._output_mask

    def _set_output_mask(self, om):
        if om is None:
            self._output_mask = None
        else:
            self._output_mask = np.sort(np.unique(np.asarray(om, dtype=np.int)))
            # we do not allow directly modify the elements of output_mask here
            # as it cannot trigger the unique/sort check
            self._output_mask.flags.writeable = False # TODO: PropertyArray?

    @property
    def input_size(self):
        return self._input_mask.size if self._input_mask is not None else None

    @property
    def output_size(self):
        return self._output_mask.size if self._output_mask is not None else None

    @property
    def _A_shape(self):
        """
        The shape of all the coefficients.
        Note that not all the elements are independent.
        It's organized for the convenience of polynomial evaluation.
        """
        if self._input_mask is None or self._output_mask is None:
            raise RuntimeError('you have not defined self.input_mask and/or '
                               'self.output_mask yet.')
        if self._order == 'linear':
            return (self.output_size, self.input_size + 1)
        elif self._order == 'quadratic':
            return (self.output_size, self.input_size, self.input_size)
        elif self._order == 'cubic-2':
            return (self.output_size, self.input_size, self.input_size)
        elif self._order == 'cubic-3':
            return (self.output_size, self.input_size, self.input_size,
                    self.input_size)
        else:
            raise RuntimeError(
                'unexpected value "{}" for self.order.'.format(self._order))

    @property
    def _a_shape(self):
        """
        The shape of all independent coefficients for one single output vairable.
        This would be the shape of lstsq regression.
        """
        if self._input_mask is None or self._output_mask is None:
            raise RuntimeError('you have not defined self.input_mask and/or '
                               'self.output_mask yet.')
        if self._order == 'linear':
            return (self.input_size + 1,)
        elif self._order == 'quadratic':
            return (self.input_size * (self.input_size + 1) // 2,)
        elif self._order == 'cubic-2':
            return (self.input_size * self.input_size,)
        elif self._order == 'cubic-3':
            return (self.input_size * (self.input_size - 1) *
                    (self.input_size - 2) // 6,)
        else:
            raise RuntimeError(
                'unexpected value "{}" for self.order.'.format(self._order))

    def _set(self, a, i):
        if self._input_mask is None or self._output_mask is None:
            raise RuntimeError('you have not defined self.input_mask and/or '
                               'self.output_mask yet.')
        a = np.ascontiguousarray(a)
        i = int(i)
        if a.shape != self._a_shape:
            raise ValueError('shape of a {} does not match the expected shape '
                             '{}.'.format(a.shape, self._a_shape))
        if not 0 <= i < self.output_size:
            raise ValueError('i = {} out of range for self.output_size = '
                             '{}.'.format(i, self.output_size))
        if self._order == 'linear':
            coefi = a
        else:
            coefi = np.empty(self._A_shape[1:])
            if self._order == 'quadratic':
                _set_quadratic(a, coefi, self.input_size)
            elif self._order == 'cubic-2':
                _set_cubic_2(a, coefi, self.input_size)
            elif self._order == 'cubic-3':
                _set_cubic_3(a, coefi, self.input_size)
            else:
                raise RuntimeError(
                    'unexpected value of self.order "{}".'.format(self._order))
        if self._coef is None:
            self._coef = np.zeros(self._A_shape)
        self._coef[i] = coefi


class PolyModel(Surrogate):
    """
    Polynomial surrogate model, currently up to cubic order.
    
    Parameters
    ----------
    configs : str, PolyConfig, or 1-d array_like of them
        Determining the configuration of the model. If str, should be one of
        ('linear', 'quadratic', 'cubic-2', 'cubic-3'). Note that 'quadratic'
        will be interpreted as ['linear', 'quadratic'], i.e. up to quadratic
        order; similar for `cubic-2` and `cubic-3`.
    bound_options : dict, optional
        Keyword arguments to be passed to `self.set_bound_options`. Set to `{}`
        by default.
    args : array_like, optional
        Additional arguments to be passed to `Surrogate.__init__`.
    kwargs : dict, optional
        Additional keyword arguments to be passed to `Surrogate.__init__`.
    """
    def __init__(self, configs, bound_options={}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(configs, str):
            if configs == 'linear':
                configs = ['linear']
            elif configs == 'quadratic':
                configs = ['linear', 'quadratic']
            elif configs == 'cubic-2':
                configs = ['linear', 'quadratic', 'cubic-2']
            elif configs == 'cubic-3':
                configs = ['linear', 'quadratic', 'cubic-2', 'cubic-3']
            else:
                raise ValueError('if configs is a str, it should be "linear", '
                                 '"quadratic", "cubic-2" or "cubic-3".')
        if isinstance(configs, PolyConfig):
            configs = [configs]
        if hasattr(configs, '__iter__'):
            self._configs = []
            for conf in configs:
                if isinstance(conf, str):
                    conf = PolyConfig(conf)
                if isinstance(conf, PolyConfig):
                    if conf._input_mask is None:
                        conf._set_input_mask(np.arange(self._input_size))
                    if conf._output_mask is None:
                        conf._set_output_mask(np.arange(self._output_size))
                    self._configs.append(conf)
                else:
                    raise ValueError('invalid value for the #{} element of '
                                     'configs.'.format(i))
        else:
            raise ValueError('invalid value for configs.')
        self._configs = tuple(self._configs)
        self._build_recipe()
        if isinstance(bound_options, dict):
            self.set_bound_options(**bound_options)
        else:
            raise ValueError('bound_options should be a dict.')

    @property
    def configs(self):
        return self._configs

    @property
    def n_config(self):
        return len(self._configs)

    @property
    def bound_options(self):
        return BoundOptions(self._use_bound, self._alpha, self._alpha_p,
                            self._center_max)

    def set_bound_options(self, use_bound=True, alpha=None, alpha_p=100.,
                          center_max=True):
        self._use_bound = bool(use_bound)
        if alpha is None:
            self._alpha = None
        else:
            try:
                alpha = float(alpha)
                assert alpha > 0
                self._alpha = alpha
            except Exception:
                raise ValueError('invalid value for alpha.')
        if alpha_p is None:
            if alpha is None:
                raise ValueError('alpha and alpha_p cannot both be None.')
            self._alpha_p = None
        else:
            try:
                alpha_p = float(alpha_p)
                assert alpha_p > 0
                self._alpha_p = alpha_p
            except Exception:
                raise ValueError('invalid value for alpha_p.')
        self._center_max = bool(center_max)

    def _set_bound(self, x, logp=None):
        try:
            x = np.ascontiguousarray(x)
            assert x.shape[-1] == self._input_size and x.ndim == 2
        except Exception:
            raise ValueError('invalid value for x.')
        self._mu = np.mean(x, axis=0)
        self._hess = np.linalg.inv(np.cov(x, rowvar=False))
        if self._alpha_p is not None:
            _beta = np.einsum('ij,jk,ik->i', x - self._mu, self._hess,
                              x - self._mu)**0.5
            if self._alpha_p < 100.:
                self._alpha = np.percentile(_beta, self._alpha_p)
            else:
                self._alpha = np.max(_beta) * self._alpha_p / 100.
        if self._center_max:
            try:
                logp = np.asarray(logp)
                assert x.shape[0] == logp.shape[0] and logp.ndim == 1
            except Exception:
                raise ValueError('invalid value for logp.')
            mu_f = x[np.argmax(logp)]
        else:
            mu_f = self._mu
        try:
            self._use_bound = False # to avoid using bound during fun evaluation
            self._f_mu = self._fun(mu_f)
        finally:
            self._use_bound = True

    @property
    def recipe(self):
        return self._recipe

    def _build_recipe(self):
        rr = np.full((self._output_size, 4), -1)
        for ii, conf in enumerate(self._configs):
            if conf.order == 'linear':
                if np.any(rr[conf._output_mask, 0] >= 0):
                    raise ValueError(
                        'multiple linear PolyConfig(s) share at least one '
                        'common output variable. Please check your PolyConfig '
                        '#{}.'.format(ii))
                rr[conf._output_mask, 0] = ii
            elif conf.order == 'quadratic':
                if np.any(rr[conf._output_mask, 1] >= 0):
                    raise ValueError(
                        'multiple quadratic PolyConfig(s) share at least one '
                        'common output variable. Please check your PolyConfig '
                        '#{}.'.format(ii))
                rr[conf._output_mask, 1] = ii
            elif conf.order == 'cubic-2':
                if np.any(rr[conf._output_mask, 2] >= 0):
                    raise ValueError(
                        'multiple cubic_2 PolyConfig(s) share at least one '
                        'common output variable. Please check your PolyConfig '
                        '#{}.'.format(ii))
                rr[conf._output_mask, 2] = ii
            elif conf.order == 'cubic-3':
                if np.any(rr[conf._output_mask, 3] >= 0):
                    raise ValueError(
                        'multiple cubic_3 PolyConfig(s) share at least one '
                        'common output variable. Please check your PolyConfig '
                        '#{}.'.format(ii))
                rr[conf._output_mask, 3] = ii
            else:
                raise RuntimeError('unexpected value of conf.order for '
                                   'PolyConfig #{}.'.format(ii))
        if np.any(np.all(rr < 0, axis=1)):
            raise ValueError(
                'no PolyConfig has output for variable(s) {}.'.format(
                np.argwhere(np.any(np.all(rr < 0, axis=1))).flatten()))
        self._recipe = rr
        self._recipe.flags.writeable = False # TODO: PropertyArray?

    @classmethod
    def _linear(cls, config, x_in, target):
        if target == 'fun':
            return np.dot(config._coef[:, 1:], x_in) + config._coef[:, 0]
        elif target == 'jac':
            return config._coef[:, 1:]
        elif target == 'fun_and_jac':
            ff = np.dot(config._coef[:, 1:], x_in) + config._coef[:, 0]
            jj = config._coef[:, 1:]
            return ff, jj
        else:
            raise ValueError(
                'target should be one of ("fun", "jac", "fun_and_jac"), '
                'instead of "{}".'.format(target))

    @classmethod
    def _quadratic(cls, config, x_in, target):
        if target == 'fun':
            out_f = np.empty(config.output_size)
            _quadratic_f(x_in, config._coef, out_f, config.output_size,
                         config.input_size)
            return out_f
        elif target == 'jac':
            out_j = np.empty((config.output_size, config.input_size))
            _quadratic_j(x_in, config._coef, out_j, config.output_size,
                         config.input_size)
            return out_j
        elif target == 'fun_and_jac':
            out_f = np.empty(config.output_size)
            _quadratic_f(x_in, config._coef, out_f, config.output_size,
                         config.input_size)
            out_j = np.empty((config.output_size, config.input_size))
            _quadratic_j(x_in, config._coef, out_j, config.output_size,
                         config.input_size)
            return out_f, out_j
        else:
            raise ValueError(
                'target should be one of ("fun", "jac", "fun_and_jac"), '
                'instead of "{}".'.format(target))

    @classmethod
    def _cubic_2(cls, config, x_in, target):
        if target == 'fun':
            out_f = np.empty(config.output_size)
            _cubic_2_f(x_in, config._coef, out_f, config.output_size,
                       config.input_size)
            return out_f
        elif target == 'jac':
            out_j = np.empty((config.output_size, config.input_size))
            _cubic_2_j(x_in, config._coef, out_j, config.output_size,
                       config.input_size)
            return out_j
        elif target == 'fun_and_jac':
            out_f = np.empty(config.output_size)
            _cubic_2_f(x_in, config._coef, out_f, config.output_size,
                       config.input_size)
            out_j = np.empty((config.output_size, config.input_size))
            _cubic_2_j(x_in, config._coef, out_j, config.output_size,
                       config.input_size)
            return out_f, out_j
        else:
            raise ValueError(
                'target should be one of ("fun", "jac", "fun_and_jac"), '
                'instead of "{}".'.format(target))

    @classmethod
    def _cubic_3(cls, config, x_in, target):
        if target == 'fun':
            out_f = np.empty(config.output_size)
            _cubic_3_f(x_in, config._coef, out_f, config.output_size,
                       config.input_size)
            return out_f
        elif target == 'jac':
            out_j = np.empty((config.output_size, config.input_size))
            _cubic_3_j(x_in, config._coef, out_j, config.output_size,
                       config.input_size)
            return out_j
        elif target == 'fun_and_jac':
            out_f = np.empty(config.output_size)
            _cubic_3_f(x_in, config._coef, out_f, config.output_size,
                       config.input_size)
            out_j = np.empty((config.output_size, config.input_size))
            _cubic_3_j(x_in, config._coef, out_j, config.output_size,
                       config.input_size)
            return out_f, out_j
        else:
            raise ValueError(
                'target should be one of ("fun", "jac", "fun_and_jac"), '
                'instead of "{}".'.format(target))

    @classmethod
    def _eval_one(cls, config, x, target='fun'):
        x_in = np.ascontiguousarray(x[config.input_mask])
        if config.order == 'linear':
            return cls._linear(config, x_in, target)
        elif config.order == 'quadratic':
            return cls._quadratic(config, x_in, target)
        elif config.order == 'cubic-2':
            return cls._cubic_2(config, x_in, target)
        elif config.order == 'cubic-3':
            return cls._cubic_3(config, x_in, target)
        else:
            raise RuntimeError('unexpected value of config.order.')

    def _fun(self, x):
        if (self._use_bound and not self._all_linear and
            np.dot(np.dot(x - self._mu, self._hess), x - self._mu)**0.5 >
            self._alpha):
            return self._fj_bound(x, 'fun')
        else:
            ff = np.zeros(self._output_size)
            for conf in self._configs:
                ff[conf._output_mask] += self._eval_one(conf, x, 'fun')
            return ff

    def _jac(self, x):
        if (self._use_bound and not self._all_linear and
            np.dot(np.dot(x - self._mu, self._hess), x - self._mu)**0.5 >
            self._alpha):
            return self._fj_bound(x, 'jac')
        else:
            jj = np.zeros((self._output_size, self._input_size))
            for conf in self._configs:
                jj[conf._output_mask[:, np.newaxis],
                   conf._input_mask] += self._eval_one(conf, x, 'jac')
            return jj

    def _fun_and_jac(self, x):
        if (self._use_bound and not self._all_linear and
            np.dot(np.dot(x - self._mu, self._hess), x - self._mu)**0.5 >
            self._alpha):
            return self._fj_bound(x, 'fun_and_jac')
        else:
            ff = np.zeros(self._output_size)
            jj = np.zeros((self._output_size, self._input_size))
            for conf in self._configs:
                _f, _j = self._eval_one(conf, x, 'fun_and_jac')
                ff[conf._output_mask] += _f
                jj[conf._output_mask[:, np.newaxis], conf._input_mask] += _j
            return ff, jj

    def _fj_bound(self, x, target='fun'):
        beta = np.dot(np.dot(x - self._mu, self._hess), x - self._mu)**0.5
        x_0 = (self._alpha * x + (beta - self._alpha) * self._mu) / beta
        ff_0 = np.zeros(self._output_size)
        for conf in self._configs:
            ff_0[conf._output_mask] += self._eval_one(conf, x_0, 'fun')
        if target != 'jac':
            ff = (beta * ff_0 - (beta - self._alpha) * self._f_mu) / self._alpha
            if target == 'fun':
                return ff
        grad_beta = np.dot(self._hess, x - self._mu) / beta
        jj_0 = np.zeros((self._output_size, self._input_size))
        for conf in self._configs:
            jj_0[conf._output_mask[:, np.newaxis],
                 conf._input_mask] += self._eval_one(conf, x_0, 'jac')
        jj = jj_0 + np.outer((ff_0 - self._f_mu) / self._alpha -
                             np.dot(jj_0, x - self._mu) / beta, grad_beta)
        if target == 'jac':
            return jj
        elif target == 'fun_and_jac':
            return ff, jj
        raise ValueError(
                'target should be one of ("fun", "jac", "fun_and_jac"), '
                'instead of "{}".'.format(target))

    def fit(self, x, y, logp=None):
        x = np.asarray(x)
        y = np.asarray(y)
        if not (x.ndim == 2 and x.shape[-1] == self._input_size):
            raise ValueError(
                'x should be a 2-d array, with shape (# of points, # of '
                'input_size), instead of {}.'.format(x.shape))
        if not (y.ndim == 2 and y.shape[-1] == self._output_size):
            raise ValueError(
                'y should be a 2-d array, with shape (# of points, # of '
                'output_size), instead of {}.'.format(y.shape))
        if not x.shape[0] == y.shape[0]:
            raise ValueError('x and y have different # of points.')
        if x.shape[0] < self.n_param:
            raise ValueError('I need at least {} points, but you only gave me '
                             '{}.'.format(self.n_param, x.shape[0]))
        for ii in range(self._output_size):
            A = np.empty((x.shape[0], 0))
            jj_l, jj_q, jj_c2, jj_c3 = self._recipe[ii]
            kk = [0]
            if jj_l >= 0:
                _A = np.empty((x.shape[0], self._configs[jj_l]._a_shape[0]))
                _x = np.ascontiguousarray(x[...,
                                            self._configs[jj_l]._input_mask])
                _A[:, 0] = 1
                _A[:, 1:] = _x
                kk.append(kk[-1] + self._configs[jj_l]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            if jj_q >= 0:
                _A = np.empty((x.shape[0], self._configs[jj_q]._a_shape[0]))
                _x = np.ascontiguousarray(x[...,
                                              self._configs[jj_q]._input_mask])
                _lsq_quadratic(_x, _A, x.shape[0],
                               self._configs[jj_q].input_size)
                kk.append(kk[-1] + self._configs[jj_q]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            if jj_c2 >= 0:
                _A = np.empty((x.shape[0], self._configs[jj_c2]._a_shape[0]))
                _x = np.ascontiguousarray(x[...,
                                            self._configs[jj_c2]._input_mask])
                _lsq_cubic_2(_x, _A, x.shape[0],
                             self._configs[jj_c2].input_size)
                kk.append(kk[-1] + self._configs[jj_c2]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            if jj_c3 >= 0:
                _A = np.empty((x.shape[0], self._configs[jj_c3]._a_shape[0]))
                _x = np.ascontiguousarray(x[...,
                                            self._configs[jj_c3]._input_mask])
                _lsq_cubic_3(_x, _A, x.shape[0],
                             self._configs[jj_c3].input_size)
                kk.append(kk[-1] + self._configs[jj_c3]._a_shape[0])
                A = np.concatenate((A, _A), axis=-1)
            b = np.copy(y[:, ii])
            lsq = lstsq(A, b)[0]
            pp = 0
            if jj_l >= 0:
                qq = int(np.argwhere(self._configs[jj_l]._output_mask == ii))
                self._configs[jj_l]._set(lsq[kk[pp]:kk[pp + 1]], qq)
                pp += 1
            if jj_q >= 0:
                qq = int(np.argwhere(self._configs[jj_q]._output_mask == ii))
                self._configs[jj_q]._set(lsq[kk[pp]:kk[pp + 1]], qq)
                pp += 1
            if jj_c2 >= 0:
                qq = int(np.argwhere(self._configs[jj_c2]._output_mask == ii))
                self._configs[jj_c2]._set(lsq[kk[pp]:kk[pp + 1]], qq)
                pp += 1
            if jj_c3 >= 0:
                qq = int(np.argwhere(self._configs[jj_c3]._output_mask == ii))
                self._configs[jj_c3]._set(lsq[kk[pp]:kk[pp + 1]], qq)
                pp += 1
        if self._use_bound and not self._all_linear:
            self._set_bound(x, logp)

    @property
    def n_param(self):
        return np.sum([conf._a_shape[0] for conf in self._configs])

    @property
    def _all_linear(self):
        return all(conf.order == 'linear' for conf in self._configs)

import copy
from ..core.module import *
from ..core.density import *
from ._poly import *
from scipy.linalg import lstsq

__all__ = ['PolyConfig', 'PolyModel']


class PolyConfig:
    
    def __init__(self, order, input_mask, output_mask, coef=None):
        if order in ('linear', 'quad', 'cubic_2', 'cubic_3'):
            self._order = order
        else:
            raise ValueError(
                'order should be one of ("linear", "quad", "cubic_2", '
                '"cubic_3"), instead of "{}".'.format(order))
        self._input_mask = np.sort(np.unique(np.asarray(input_mask, 
                                                        dtype=np.int)))
        self._output_mask = np.sort(np.unique(np.asarray(output_mask, 
                                                         dtype=np.int)))
        self.coef = coef
    
    @property
    def order(self):
        return self._order
    
    @property
    def input_mask(self):
        return self._input_mask.copy()
    
    @property
    def output_mask(self):
        return self._output_mask.copy()
    
    @property
    def input_size(self):
        return self._input_mask.size
    
    @property
    def output_size(self):
        return self._output_mask.size
    
    @property
    def _A_shape(self):
        if self._order == 'linear':
            return (self.output_size, self.input_size + 1)
        elif self._order == 'quad':
            return (self.output_size, self.input_size, self.input_size)
        elif self._order == 'cubic_2':
            return (self.output_size, self.input_size, self.input_size)
        elif self._order == 'cubic_3':
            return (self.output_size, self.input_size, self.input_size, 
                    self.input_size)
        else:
            raise RuntimeError(
                'unexpected value of self.order "{}".'.format(self._order))
    
    @property
    def _a_shape(self):
        if self._order == 'linear':
            return (self.input_size + 1,)
        elif self._order == 'quad':
            return (self.input_size * (self.input_size + 1) // 2,)
        elif self._order == 'cubic_2':
            return (self.input_size * self.input_size,)
        elif self._order == 'cubic_3':
            return (self.input_size * (self.input_size - 1) * 
                    (self.input_size - 2) // 6,)
        else:
            raise RuntimeError(
                'unexpected value of self.order "{}".'.format(self._order))
    
    @property
    def coef(self):
        return self._coef
    
    @coef.setter
    def coef(self, A):
        if A is not None:
            if A.shape != self._A_shape:
                raise ValueError(
                    'shape of the coef matrix {} does not match the expected '
                    'shape {}.'.format(A.shape, self._A_shape))
            self._coef = np.copy(A)
        else:
            self._coef = None
            
    def _set(self, a, ii):
        ii = int(ii)
        if a.shape != self._a_shape:
            raise ValueError('shape of a {} does not match the expected shape '
                             '{}.'.format(a.shape, self._a_shape))
        if not 0 <= ii <= self.output_size:
            raise ValueError('ii = {} out of range for self.output_size = '
                             '{}.'.format(ii, self.output_size))
        if self._order == 'linear':
            coefii = a
        else:
            coefii = np.empty(self._A_shape[1:])
            if self._order == 'quad':
                _set_quad(a, coefii, self.input_size)
            elif self._order == 'cubic_2':
                _set_cubic_2(a, coefii, self.input_size)
            elif self._order == 'cubic_3':
                _set_cubic_3(a, coefii, self.input_size)
            else:
                raise RuntimeError(
                    'unexpected value of self.order "{}".'.format(self._order))
        if self._coef is None:
            self._coef = np.zeros(self._A_shape)
        self._coef[ii] = coefii


class PolyModel(Surrogate):
    
    def __init__(self, configs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(configs, PolyConfig):
            self._configs = [configs]
        elif hasattr(configs, '__iter__'):
            self._configs = []
            for conf in configs:
                if isinstance(conf, PolyConfig):
                    self._configs.append(conf)
                else:
                    raise ValueError(
                        'not all the element(s) in configs are PolyConfig(s).')
        else:
            raise ValueError(
                'configs should be a PolyConfig, or consist of PolyConfig(s).')
        self._build_recipe()
        self._use_bound = False
        
    # TODO: is it really bad to allow users to reach configs?
    @property
    def configs(self):
        return copy.deepcopy(self._configs)
    
    @property
    def n_config(self):
        return len(self._config)
    
    @property
    def alpha(self):
        try:
            return self._alpha
        except:
            return None
    
    @alpha.setter
    def alpha(self, a):
        try:
            a = float(a)
            assert a > 0
            self._alpha = a
        except:
            raise ValueError('alpha should be a positive float.')
    
    @property
    def mu(self):
        try:
            return self._mu.copy()
        except:
            return None
    
    @property
    def hess(self):
        try:
            return self._hess.copy()
        except:
            return None
    
    @property
    def f_mu(self):
        try:
            return self._f_mu.copy()
        except:
            return None
    
    @property
    def use_bound(self):
        return self._use_bound
    
    @use_bound.setter
    def use_bound(self, ub):
        self._use_bound = bool(ub)
    
    def set_bound(self, x, alpha=None, alpha_p=99, mu_f=None):
        try:
            x = np.ascontiguousarray(x)
            assert x.shape[-1] == self._input_size
        except:
            raise ValueError('invalid value for x.')
        x = x.reshape((-1, self._input_size))
        self._mu = np.mean(x, axis=0)
        self._hess = np.linalg.inv(np.cov(x, rowvar=False))
        if alpha is None:
            if (self.alpha is None) or (alpha_p is not None):
                try:
                    alpha_p = float(alpha_p)
                    assert alpha_p > 0
                except:
                    raise ValueError('alpha_p should be a positive float.')
                _beta = np.einsum('ij,jk,ik->i', x - self._mu, self._hess, 
                                  x - self._mu)**0.5
                if alpha_p < 100:
                    self.alpha = np.percentile(_beta, alpha_p)
                else:
                    self.alpha = np.max(_beta) * alpha_p / 100
            else:
                pass
        else:
            self.alpha = alpha
        if mu_f is None:
            mu_f = self._mu
        else:
            mu_f = np.asarray(mu_f)
            if not mu_f.shape == (self._input_size,):
                raise ValueError(
                    'mu_f should be a 1-d array with shape ({},), or None if '
                    'you want to use the sample mean, instead of '
                    '{}.'.format(self._input_size, mu_f.shape))
        self._f_mu = self._fun(mu_f)
        self._use_bound = True
    
    @property
    def recipe(self):
        return self._recipe.copy()
    
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
            elif conf.order == 'quad':
                if np.any(rr[conf._output_mask, 1] >= 0):
                    raise ValueError(
                        'multiple quad PolyConfig(s) share at least one common '
                        'output variable. Please check your PolyConfig '
                        '#{}.'.format(ii))
                rr[conf._output_mask, 1] = ii
            elif conf.order == 'cubic_2':
                if np.any(rr[conf._output_mask, 2] >= 0):
                    raise ValueError(
                        'multiple cubic_2 PolyConfig(s) share at least one '
                        'common output variable. Please check your PolyConfig '
                        '#{}.'.format(ii))
                rr[conf._output_mask, 2] = ii
            elif conf.order == 'cubic_3':
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
    def _quad(cls, config, x_in, target):
        if target == 'fun':
            out_f = np.empty(config.output_size)
            _quad_f(x_in, config._coef, out_f, config.output_size, 
                    config.input_size)
            return out_f
        elif target == 'jac':
            out_j = np.empty((config.output_size, config.input_size))
            _quad_j(x_in, config._coef, out_j, config.output_size, 
                    config.input_size)
            return out_j
        elif target == 'fun_and_jac':
            out_f = np.empty(config.output_size)
            _quad_f(x_in, config._coef, out_f, config.output_size, 
                    config.input_size)
            out_j = np.empty((config.output_size, config.input_size))
            _quad_j(x_in, config._coef, out_j, config.output_size, 
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
        elif config.order == 'quad':
            return cls._quad(config, x_in, target)
        elif config.order == 'cubic_2':
            return cls._cubic_2(config, x_in, target)
        elif config.order == 'cubic_3':
            return cls._cubic_3(config, x_in, target)
        else:
            raise RuntimeError('unexpected value of config.order.')
    
    def _fun(self, x):
        if not x.shape == (self._input_size,):
            raise ValueError('shape of x should be {}, instead of '
                             '{}.'.format((self._input_size,), x.shape))
        if self._use_bound and np.dot(np.dot(x - self._mu, self._hess), 
                                      x - self._mu)**0.5 > self._alpha:
            return self._fj_bound(x, 'fun')
        else:
            ff = np.zeros(self._output_size)
            for conf in self._configs:
                ff[conf._output_mask] += self._eval_one(conf, x, 'fun')
            return ff
    
    def _jac(self, x):
        if not x.shape == (self._input_size,):
            raise ValueError('shape of x should be {}, instead of '
                             '{}.'.format((self._input_size,), x.shape))
        if self._use_bound and np.dot(np.dot(x - self._mu, self._hess), 
                                      x - self._mu)**0.5 > self._alpha:
            return self._fj_bound(x, 'jac')            
        else:
            jj = np.zeros((self._output_size, self._input_size))
            for conf in self._configs:
                jj[conf._output_mask[:, np.newaxis], 
                   conf._input_mask] += self._eval_one(conf, x, 'jac')
            return jj
    
    def _fun_and_jac(self, x):
        if not x.shape == (self._input_size,):
            raise ValueError('shape of x should be {}, instead of '
                             '{}.'.format((self._input_size,), x.shape))
        ff = np.zeros(self._output_size)
        jj = np.zeros((self._output_size, self._input_size))
        if self._use_bound and np.dot(np.dot(x - self._mu, self._hess), 
                                      x - self._mu)**0.5 > self._alpha:
            return self._fj_bound(x, 'fun_and_jac')
        else:
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
    
    def fit(self, x, y, w=None, use_bound=True, bound_kwargs={}):
        # w_min=None, w_max=None
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
        if w is not None:
            w = np.asarray(w).flatten()
            if not w.shape[0] == x.shape[0]:
                raise ValueError('x and w have different # of points.')
            """if w_min is None:
                w_min = 1 / w.shape[0]**0.5
            elif w_min is False:
                w_min = 0.
            else:
                try:
                    w_min = float(w_min)
                except:
                    raise ValueError('invalid value for w_min.')
            if w_max is None:
                w_max = w.shape[0]**0.5
            elif w_max is False:
                w_max = np.inf
            else:
                try:
                    w_max = float(w_max)
                except:
                    raise ValueError('invalid value for w_max.')
            w = np.clip(w, w_min, w_max)"""
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
                _lsq_quad(_x, _A, x.shape[0], self._configs[jj_q].input_size)
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
            if w is not None:
                b *= w
                A *= w[:, np.newaxis]
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
        if use_bound:
            self.set_bound(x, **bound_kwargs)
    
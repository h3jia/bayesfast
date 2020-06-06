import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.decomposition import FastICA
from ..utils import check_client
from ..utils.kde import kde
from ..utils.cubic import cubic_spline
from ..utils.random import check_state, multivariate_normal
from distributed import Client
from itertools import starmap
import copy
import warnings

try:
    from getdist import plots, MCSamples
    HAS_GETDIST = True
except:
    HAS_GETDIST = False

__all__ = ['SIT']


class SIT:
    """
    Sliced Iterative Transform.
    
    Parameters
    ----------
    
    """
    def __init__(self, n_iter=10, client=None, bw_factor=1., m_ica=20000,
                 random_state=None, cubic_options={},
                 ica_options={'max_iter': 80}, random_options={}):
        self._data = None
        self._cubic = []
        self.n_iter = n_iter
        self.client = client
        self.bw_factor = bw_factor
        self.m_ica = m_ica
        self.random_state = random_state
        self.cubic_options = cubic_options
        self.ica_options = ica_options
        self.random_options = random_options
        
    def __getstate__(self):
        """We need this to make self._client work correctly."""
        self_dict = self.__dict__.copy()
        del self_dict['_client']
        return self_dict
    
    @property
    def data(self):
        return self._data.copy()
    
    @property
    def data_init(self):
        return self._data_init.copy()
    
    @property
    def dim(self):
        return self._data.shape[-1]
    
    @property
    def weights(self):
        return self._weights.copy()
    
    @property
    def n_iter(self):
        return self._n_iter
    
    @n_iter.setter
    def n_iter(self, n):
        try:
            n = int(n)
            assert n > 0
        except:
            raise ValueError('n_iter should be a positive int.')
        self._n_iter = n
    
    @property
    def i_iter(self):
        return len(self._cubic)
    
    def add_iter(self, n):
        self.n_iter = self.n_iter + n
    
    @property
    def client(self):
        return self._client
    
    @client.setter
    def client(self, clt):
        if isinstance(clt, (int, Client)) or clt is None:
            self._client = clt
        else:
            raise ValueError('invalid value for client.')
    
    @property
    def bw_factor(self):
        return self._bw_factor
    
    @bw_factor.setter
    def bw_factor(self, bw):
        try:
            bw = float(bw)
            assert bw > 0
        except:
            raise ValueError('bw_factor should be a positive float.')
        self._bw_factor = bw
    
    @property
    def m_ica(self):
        return self._m_ica
    
    @m_ica.setter
    def m_ica(self, m):
        try:
            m = int(m)
            assert m > 0
        except:
            raise ValueError('m_ica should be a positive int.')
        self._m_ica = m
    
    @property
    def random_state(self):
        return self._random_state
    
    @random_state.setter
    def random_state(self, state):
        if state is None:
            self._random_state = None
        else:
            self._random_state = check_state(state)
    
    @property
    def cubic_options(self):
        return self._cubic_options
    
    @cubic_options.setter
    def cubic_options(self, co):
        try:
            self._cubic_options = dict(co)
        except:
            raise ValueError('cubic_options should be a dict.')
    
    @property
    def ica_options(self):
        return self._ica_options
    
    @ica_options.setter
    def ica_options(self, io):
        try:
            self._ica_options = dict(io) 
        except:
            raise ValueError('ica_optionss should be a dict.')
    
    @property
    def random_options(self):
        return self._random_options
    
    @random_options.setter
    def random_options(self, ro):
        try:
            self._random_method = dict(ro)
        except:
            raise ValueError('random_optipms should be a dict.')
    
    def _gaussianize_1d(self, x):
        k = kde(x, bw_factor=self._bw_factor, weights=self._weights)
        c = cubic_spline(x, lambda xx: norm.ppf(k.cdf(xx)),
                         **self._cubic_options)
        return c
    
    def _gaussianize_nd(self, x):
        foo = self._client.map(self._gaussianize_1d, x.T)
        map_result = self._client.gather(foo)
        self._cubic.append(map_result)
        y = np.array([map_result[i](x[:, i]) for i in range(self.dim)]).T
        return y
    
    def _ica(self, x):
        io = self._ica_options.copy()
        if not 'random_state' in io:
            io['random_state'] = self._random_state
        ica = FastICA(**io)
        if self._m_ica is None:
            ica.fit(x)
        else:
            n_ica = min(x.shape[0], self.m_ica)
            ica.fit(x[self._random_state.choice(x.shape[0], n_ica, False)])
        y = ica.transform(x)
        m = np.mean(x, axis=0)
        s = np.std(y, axis=0)
        y /= s
        A = ica.components_ / s[:, np.newaxis]
        B = np.linalg.inv(A)
        return y, A, B, m
    
    def _init_data(self, data, weights):
        if data is None:
            if self._data is None:
                raise ValueError('you have not given me the data to fit.')
        
        else:
            try:
                data = np.array(data)
            except:
                raise ValueError('invalid value for data.')
            if data.ndim == 1:
                self._data = data[:, np.newaxis]
            elif data.ndim == 2:
                self._data = data
            elif data.ndim >= 3:
                self._data = data.reshape((-1, data.shape[-1]))
            else:
                raise ValueError('invalid value for data.ndim.')
            self._data_init = self._data.copy()
            
            _n = self._data.shape[0]
            if weights is not None:
                try:
                    weights = np.asarray(weights)
                    assert weights.shape == (_n,)
                except:
                    raise ValueError('invalid value for weights.')
                self._weights = weights
            else:
                self._weights = np.ones(_n) / _n
            
            self._cubic = []
            self._A = np.zeros((0, self.dim, self.dim))
            self._B = np.zeros((0, self.dim, self.dim))
            self._m = np.zeros((0, self.dim))
            self._logdetA = np.zeros(0)
    
    def fit(self, data=None, weights=None, n_run=None, plot=0):
        self._init_data(data, weights)
        self._random_state = check_state(self._random_state)
        
        try:
            plot = int(plot)
        except:
            raise ValueError('plot should be an int.')
        if (not HAS_GETDIST) and (plot != 0):
            plot = 0
            warnings.warn('you have not installed getdist, so I can only do '
                          'plot=0.', RuntimeWarning)
        
        if n_run is None:
            n_run = self.n_iter - self.i_iter
        else:
            try:
                n_run = int(n_run)
                assert n_run > 0
            except:
                raise ValueError('invalid value for n_run.')
            if n_run > self.n_iter - self.i_iter:
                self.n_iter = self.i_iter + n_run
        
        try:
            old_client = self._client
            self._client, _new_client = check_client(self._client)
            for i in range(n_run):
                try:
                    y, A, B, m = self._ica(self._data)
                except:
                    warnings.warn(
                        "we found that sometimes ica goes wrong, but actually "
                        "it can work if we use a different random seed, so "
                        "let's give it one more chance.", RuntimeWarning)
                    y, A, B, m = self._ica(self._data)
                self._data = self._gaussianize_nd(y)
                self._A = np.concatenate((self._A, A[np.newaxis]), axis=0)
                self._B = np.concatenate((self._B, B[np.newaxis]), axis=0)
                self._m = np.concatenate((self._m, m[np.newaxis]), axis=0)
                self._logdetA = np.append(self._logdetA, 
                                          np.log(np.abs(np.linalg.det(A))))
                finite_index = np.isfinite(self._data).all(axis=1)
                if len(finite_index) < self._data.shape[0]:
                    warnings.warn(
                        'inf encountered for some data points. We will remove '
                        'these inf points for now.', RuntimeWarning)
                self._data = self._data[finite_index, :]
                if (plot > 0) and (not i % plot):
                    self.triangle_plot()
            if plot < 0:
                self.triangle_plot()
        finally:
            if _new_client:
                self._client.cluster.close()
                self._client.close()
                self._client = old_client
    
    def triangle_plot(self):
        if not HAS_GETDIST:
            raise RuntimeError(
                'you need to install getdist to get the triangle plot.')
        samples = MCSamples(samples=self._data)
        g = plots.getSubplotPlotter()
        g.triangle_plot([samples,], filled=True, contour_args={'alpha':0.8},
                        diag1d_kwargs={'normalized':True})
        plt.show()
        print("triangle plot at iteration " + str(iteration))
        print("\n---------- ---------- ---------- ---------- ----------\n")
        
    def sample(self, n, use_client=False):
        n = int(n)
        if n <= 0:
            raise ValueError('n should be positive.')
        ro = self._random_options.copy()
        if not 'random_state' in ro:
            ro['random_state'] = self._random_state
        y = multivariate_normal(np.zeros(self.dim), np.eye(self.dim), n, **ro)
        x, log_j = self.backward_transform(y, use_client)
        return x, log_j, y
    
    def _do_evaluate(self, c, x):
        return c.evaluate(x)
    
    def _do_derivative(self, c, x):
        return c.derivative(x)
    
    def _do_solve(self, c, x):
        return c.solve(x)
    
    def forward_transform(self, x, use_client=False):
        try:
            y = np.array(x)
        except:
            raise ValueError('invalid value for x.')
        if y.ndim == 1:
            if self.dim == 1:
                y = y[:, np.newaxis]
            else:
                y = y[np.newaxis, :]
        if not (y.ndim == 2 and y.shape[-1] == self.dim):
            raise ValueError('invalid shape for x.')
        log_j = np.zeros(y.shape[0])
        
        try:
            if use_client:
                old_client = self._client
                self._client, _new_client = check_client(self._client)
            for i in range(self.i_iter):
                y = (y - self._m[i]) @ self._A[i].T
                if use_client:
                    foo = self._client.map(self._do_derivative, self._cubic[i],
                                           y.T)
                    map_result = self._client.gather(foo)
                else:
                    map_result = list(starmap(self._do_derivative,
                                              zip(self._cubic[i], y.T)))
                log_j += np.sum(np.log(map_result), axis=0)
                if use_client:
                    foo = self._client.map(self._do_evaluate, self._cubic[i],
                                           y.T)
                    map_result = self._client.gather(foo)
                else:
                    map_result = list(starmap(self._do_evaluate,
                                              zip(self._cubic[i], y.T)))
                y = np.array(map_result).T
            log_j += np.sum(self._logdetA)
            return y, log_j
        finally:
            if use_client and _new_client:
                self._client.cluster.close()
                self._client.close()
                self._client = old_client
    
    def backward_transform(self, y, use_client=False):
        try:
            x = np.array(y)
        except:
            raise ValueError('invalid value for y.')
        if x.ndim == 1:
            if self.dim == 1:
                x = x[:, np.newaxis]
            else:
                x = x[np.newaxis, :]
        if not (x.ndim == 2 and x.shape[-1] == self.dim):
            raise ValueError('invalid shape for y.')
        log_j = np.zeros(x.shape[0])
        
        try:
            if use_client:
                old_client = self._client
                self._client, _new_client = check_client(self._client)
            for i in reversed(range(self.i_iter)):
                if use_client:
                    foo = self._client.map(self._do_solve, self._cubic[i], x.T)
                    map_result = self._client.gather(foo)
                else:
                    map_result = list(starmap(self._do_solve,
                                              zip(self._cubic[i], x.T)))
                x = np.array(map_result).T
                if use_client:
                    foo = self._client.map(self._do_derivative, self._cubic[i],
                                           x.T)
                    map_result = self._client.gather(foo)
                else:
                    map_result = list(starmap(self._do_derivative,
                                              zip(self._cubic[i], x.T)))
                log_j += np.sum(np.log(map_result), axis=0)
                x = x @ self._B[i].T + self._m[i]
            log_j += np.sum(self._logdetA)
            return x, log_j
        finally:
            if use_client and _new_client:
                self._client.cluster.close()
                self._client.close()
                self._client = old_client
    
    def logq(self, x, use_client=False):
        y, log_j = self.forward_transform(x, use_client)
        return np.sum(norm.logpdf(y), axis=-1) + log_j

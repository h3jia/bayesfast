import numpy as np
from .hmc_utils.step_size import DualAverageAdaptation
from .hmc_utils.metrics import QuadMetric, QuadMetricDiag, QuadMetricFull
from .hmc_utils.metrics import QuadMetricDiagAdapt, QuadMetricFullAdapt
from .hmc_utils.stats import NStepStats, NStats
from ..utils.random import check_state, split_state
from copy import deepcopy
import warnings

__all__ = ['NTrace', 'HTrace', 'ETrace', 'TraceTuple']

# TODO: StatsTuple?


class _Trace:
    """Utilities shared by all different Trace classes."""
    def __init__(self, n_chain=4, n_iter=1500, n_warmup=500, x_0=None,
                 random_state=None):
        self._set_n_chain(n_chain)
        self.n_iter = n_iter
        self.n_warmup = n_warmup
        self._set_x_0(x_0)
        self._set_random_state(random_state)
        self._random_state_init = deepcopy(self._random_state)
    
    @property
    def n_chain(self):
        return self._n_chain
    
    def _set_n_chain(self, n):
        try:
            n = int(n)
            assert n > 0
        except:
            raise ValueError('n_chain should be a positive int, instead of '
                             '{}.'.format(n))
        self._n_chain = n
    
    @property
    def n_iter(self):
        try:
            return self._n_iter
        except:
            return 0
    
    @n_iter.setter
    def n_iter(self, n):
        try:
            n = int(n)
            assert n > 0
        except:
            raise ValueError('n_iter should be a positive int, instead of '
                             '{}.'.format(n))
        if n < self.i_iter:
            raise ValueError(
                'you have already run {} iterations, so n_iter should not be '
                'smaller than this number.'.format(self.i_iter))
        if n < self.n_warmup:
            raise ValueError('n_warmup is {}, so n_iter should not be smaller '
                             'than this number.'.format(self.n_warmup))
        self._n_iter = n
    
    @property
    def i_iter(self):
        raise NotImplementedError('Abstract property.')
    
    @property
    def n_warmup(self):
        try:
            return self._n_warmup
        except:
            return 0
    
    @n_warmup.setter
    def n_warmup(self, n):
        try:
            n = int(n)
            assert n > 0
        except:
            raise ValueError('n_warmup should be a positive int, instead of '
                             '{}.'.format(n))
        self._warmup_check(n)
        if n >= self.n_iter:
            raise ValueError('n_iter is {}, so n_warmup should be smaller than '
                             'this number.'.format(self.n_iter))
        self._n_warmup = n
    
    def _warmup_check(self, n):
        pass
    
    def add_iter(self, n):
        self.n_iter = self.n_iter + n
    
    def add_warmup(self, n):
        self.n_warmup = self.n_warmup + n
    
    @property
    def x_0(self):
        return self._x_0
    
    def _set_x_0(self, x):
        if x is None:
            self._x_0 = None
        else:
            try:
                self._x_0 = np.atleast_1d(x).copy()
            except:
                raise ValueError('invalid value for x_0.')
    
    @property
    def samples(self):
        raise NotImplementedError('Abstract property.')
    
    @property
    def input_size(self):
        try:
            return self.x_0.shape[-1]
        except:
            return None
    
    @property
    def random_state(self):
        return self._random_state
    
    def _set_random_state(self, state):
        self._random_state = check_state(state)
    
    @property
    def random_state_init(self):
        return deepcopy(self._random_state_init)


class _HTrace(_Trace):
    """Utilities shared by HTrace and NTrace."""
    def __init__(self, n_chain=4, n_iter=1500, n_warmup=500, x_0=None,
                 random_state=None, step_size=1., adapt_step_size=True,
                 metric='diag', adapt_metric=True, max_change=1000.,
                 target_accept=0.8, gamma=0.05, k=0.75, t_0=10.,
                 initial_mean=None, initial_weight=10., adapt_window=60,
                 update_window=1, doubling=True, transform_x=True):
        super().__init__(n_chain, n_iter, n_warmup, x_0, random_state)
        self._samples = []
        self._chain_id = None
        self._set_max_change(max_change)
        self._transform_x = bool(transform_x)
        self._set_step_size(step_size, adapt_step_size, target_accept, gamma, k,
                            t_0)
        self._set_metric(metric, adapt_metric, initial_mean, initial_weight,
                         adapt_window, update_window, doubling)
    
    def _set_max_change(self, max_change):
        try:
            max_change = float(max_change)
            assert max_change > 0
        except:
            raise ValueError('max_change should be a positive float, instead '
                             'of {}.'.format(max_change))
        self._max_change = max_change
    
    def _set_random_state(self, state):
        if state is None:
            self._random_state = None
        else:
            self._random_state = check_state(state)
    
    @property
    def chain_id(self):
        return self._chain_id
    
    def _init_chain(self, i):
        if self._x_0 is None:
            raise RuntimeError('no valid x_0 is given.')
        if self._chain_id is not None:
            warnings.warn('chain_id is supposed to be set only once, but now '
                          'you are trying to modify it.', RuntimeWarning)
        try:
            i = int(i)
            assert i >= 0
            assert i < self.n_chain
        except:
            raise ValueError(
                'i should satisfy 0 <= i < n_chain, but you give {}.'.format(i))
        self._chain_id = i
        if self.random_state is None:
            self._random_state = check_state(None)
        self._random_state = split_state(self._random_state, self.n_chain)[i]
        self._x_0.reshape((-1, self._x_0.shape[-1]))
        self._x_0 = self._x_0[self._random_state.randint(0, self._x_0.shape[0])]
        self._set_step_size_2()
        self._set_metric_2()
    
    @property
    def step_size(self):
        return self._step_size
    
    @property
    def metric(self):
        return self._metric
    
    @property
    def max_change(self):
        return self._max_change
    
    @property
    def samples(self):
        return np.array(self._samples)
    
    @property
    def samples_original(self):
        try:
            return self._samples_original
        except:
            return self.samples
    
    @property
    def i_iter(self):
        try:
            return len(self._samples)
        except:
            return 0
    
    @property
    def finished(self):
        if self.i_iter < self.n_iter:
            return False
        elif self.i_iter == self.n_iter:
            return True
        else:
            raise RuntimeError('unexpected behavior: i_iter seems larger than '
                               'n_iter.')
    
    @property
    def logp(self):
        return np.array(self.stats._logp)
    
    @property
    def logp_original(self):
        try:
            return self._logp_original
        except:
            return self.logp
    
    @property
    def stats(self):
        try:
            return self._stats
        except:
            raise NotImplementedError('stats is not defined for this trace.')
    
    @property
    def transform_x(self):
        return self._transform_x
    
    def update(self, point, stats):
        self._samples.append(point)
        self._stats.update(stats)
    
    def get(self, since_iter=None, include_warmup=False, return_logp=False):
        if since_iter is None:
            since_iter = 0 if include_warmup else self.n_warmup
        else:
            try:
                since_iter = int(since_iter)
            except:
                raise ValueError('invalid value for since_iter.')
        if return_logp:
            return (self.samples[since_iter:],
                    np.array(self.stats._logp[since_iter:]))
        else:
            return self.samples[since_iter:]
    
    __call__ = get
    
    def _warmup_check(self, n):
        if self.i_iter > 0:
            _adapt_metric = isinstance(self.metric, (QuadMetricDiagAdapt,
                                                     QuadMetricFullAdapt))
            _adapt_step_size = self._step_size._adapt
            if _adapt_metric or _adapt_step_size:
                if self.n_warmup < self.i_iter or n < self.i_iter:
                    warnings.warn(
                        'please be cautious to modify n_warmup for the adaptive'
                        ' HMC/NUTS sampler, when i_iter is larger than '
                        'n_warmup(old) and/or n_warmup(new).', RuntimeWarning)
    
    def _set_step_size(self, step_size, adapt_step_size, target_accept, gamma,
                       k, t_0):
        if isinstance(step_size, DualAverageAdaptation):
            self._step_size = step_size
        else:
            try:
                step_size = float(step_size)
                assert step_size > 0
            except:
                raise ValueError('invalid value for step_size.')
            self._step_size = step_size
            self._adapt_step_size = bool(adapt_step_size)
            
            try:
                target_accept = float(target_accept)
                assert 0 < target_accept < 1
            except:
                raise ValueError('invalid value for target_accept.')
            self._target_accept = target_accept
            
            try:
                gamma = float(gamma)
                assert gamma != 0
            except:
                raise ValueError('invalid value for gamma.')
            self._gamma = gamma
            
            try:
                k = float(k)
            except:
                raise ValueError('invalid value for k.')
            self._k = k
            
            try:
                t_0 = float(t_0)
                assert t_0 >= 0
            except:
                raise ValueError('invalid value for t_0.')
            self._t_0 = t_0
    
    def _set_step_size_2(self):
        if isinstance(self.step_size, DualAverageAdaptation):
            pass
        else:
            self._step_size = DualAverageAdaptation(
                self._step_size / self.input_size**0.25, self._target_accept,
                self._gamma, self._k, self._t_0, self._adapt_step_size)
    
    def _set_metric(self, metric, adapt_metric, initial_mean, initial_weight,
                    adapt_window, update_window, doubling):
        if isinstance(metric, QuadMetric):
            self._metric = metric
        else:
            if metric == 'diag' or metric == 'full':
                pass
            else:
                try:
                    metric = np.asarray(metric)
                    n = metric.shape[0]
                    assert metric.shape == (n,) or metric.shape == (n, n)
                except:
                    raise ValueError('invalid value for metric.')
            self._metric = metric
            self._adapt_metric = bool(adapt_metric)
            
            if initial_mean is None:
                pass
            else:
                try:
                    initial_mean = np.atleast_1d(initial_mean)
                    assert initial_mean.ndim == 1
                except:
                    raise ValueError('invalid value for initial_mean.')
            self._initial_mean = initial_mean
            
            try:
                initial_weight = float(initial_weight)
                assert initial_weight > 0
            except:
                raise ValueError('invalid value for initial_weight.')
            self._initial_weight = initial_weight
            
            try:
                adapt_window = int(adapt_window)
                assert adapt_window > 0
            except:
                raise ValueError('invalid value for adapt_window.')
            self._adapt_window = adapt_window
            
            try:
                update_window = int(update_window)
                assert update_window > 0
            except:
                raise ValueError('invalid value for update_window.')
            self._update_window = update_window
            self._doubling = bool(doubling)
    
    def _set_metric_2(self):
        if isinstance(self.metric, QuadMetric):
            pass
        else:
            if self._metric == 'diag':
                self._metric = np.ones(self.input_size)
            elif self._metric == 'full':
                self._metric = np.eye(self.input_size)
            elif isinstance(self._metric, np.ndarray):
                pass
            else:
                raise RuntimeError('unexpected value for self._metric.')
            
            if self._initial_mean is None:
                self._initial_mean = self.x_0.copy()
            
            if self._metric.ndim == 1 and self._adapt_metric:
                self._metric = QuadMetricDiagAdapt(
                    self.input_size, self._initial_mean, self._metric,
                    self._initial_weight, self._adapt_window,
                    self._update_window, self._doubling)
            elif self._metric.ndim == 2 and self._adapt_metric:
                self._metric = QuadMetricFullAdapt(
                    self.input_size, self._initial_mean, self._metric,
                    self._initial_weight, self._adapt_window,
                    self._update_window, self._doubling)
            elif self._metric.ndim == 1 and not self._adapt_metric:
                self._metric = QuadMetricDiag(self._metric)
            elif self._metric.ndim == 2 and not self._adapt_metric:
                self._metric = QuadMetricFull(self._metric)
            else:
                raise RuntimeError('unexpected value for self._metric.')


class HTrace(_HTrace):
    """Trace class for the (vanilla) HMC sampler."""
    def __init__(*args, **kwargs):
        raise NotImplementedError


class NTrace(_HTrace):
    """Trace class for the NUTS sampler."""
    def __init__(self, n_chain=4, n_iter=1500, n_warmup=500, x_0=None,
                 random_state=None, step_size=1., adapt_step_size=True,
                 metric='diag', adapt_metric=True, max_change=1000.,
                 max_treedepth=10, target_accept=0.8, gamma=0.05, k=0.75,
                 t_0=10., initial_mean=None, initial_weight=10.,
                 adapt_window=60, update_window=1, doubling=True,
                 transform_x=True):
        super().__init__(n_chain, n_iter, n_warmup, x_0, random_state,
                         step_size, adapt_step_size, metric, adapt_metric, 
                         max_change, target_accept, gamma, k, t_0, initial_mean,
                         initial_weight, adapt_window, update_window, doubling,
                         transform_x)
        try:
            max_treedepth = int(max_treedepth)
            assert max_treedepth > 0
        except:
            raise ValueError('max_treedepth should be a postive int, instead '
                             'of {}.'.format(max_treedepth))
        self._max_treedepth = max_treedepth
        self._stats = NStats()
    
    @property
    def max_treedepth(self):
        return self._max_treedepth
    
    @property
    def n_call(self):
        return sum(self._stats._tree_size[1:]) + self.n_iter + 1
        """
        Here we add n_iter because at the beginning of each iteration, 
        We recompute logp_and_grad at the starting point.
        In principle this can be avoided by reusing the old values,
        But the current implementation doesn't do it in this way.
        We add another 1 for the test during initialization.
        """


class ETrace(_Trace):
    """Trace class for the ensemble sampler from emcee."""
    def __init__(*args, **kwargs):
        raise NotImplementedError


class TraceTuple:
    """Collection of multiple NTrace/HTrace from different chains."""
    def __init__(self, traces):
        try:
            traces = tuple(traces)
            if isinstance(traces[0], NTrace):
                self._sampler = 'NUTS'
            elif isinstance(traces[0], HTrace):
                self._sampler = 'HMC'
            else:
                raise ValueError('traces[0] is neither NTrace nor '
                                 'HTrace.')
            _type = type(traces[0])
            for i, t in enumerate(traces):
                assert type(t) == _type
                assert t.chain_id == i
            self._traces = traces
        except:
            raise ValueError('invalid value for traces.')
    
    @property
    def traces(self):
        return self._traces
    
    @property
    def sampler(self):
        return self._sampler
    
    @property
    def n_chain(self):
        return self.traces[0].n_chain
    
    @property
    def n_iter(self):
        return self.traces[0].n_iter
    
    @n_iter.setter
    def n_iter(self, n):
        tmp = self.n_iter
        try:
            for t in self.traces:
                t.n_iter = n
        except:
            for t in self.traces:
                t._n_iter = tmp
            raise
    
    @property
    def i_iter(self):
        return self.traces[0].i_iter
    
    @property
    def n_warmup(self):
        return self.traces[0].n_warmup
    
    @n_warmup.setter
    def n_warmup(self, n):
        tmp = self.n_warmup
        try:
            for t in self.traces:
                t.n_warmup = n
        except:
            for t in self.traces:
                t._n_warmup = tmp
            raise
    
    @property
    def samples(self):
        s = np.array([t.samples for t in self.traces])
        if s.dtype.kind != 'f':
            warnings.warn('the array of samples does not has dtype of float, '
                          'presumably because different chains have run for '
                          'different lengths.', RuntimeWarning)
        return s
    
    @property
    def samples_original(self):
        s = np.array([t.samples_original for t in self.traces])
        if s.dtype.kind != 'f':
            warnings.warn('the array of samples_original does not has dtype of '
                          'float, presumably because different chains have run '
                          'for different lengths.', RuntimeWarning)
        return s
    
    @property
    def logp(self):
        l = np.array([t.logp for t in self.traces])
        if l.dtype.kind != 'f':
            warnings.warn('the array of logp does not has dtype of float, '
                          'presumably because different chains have run for '
                          'different lengths.', RuntimeWarning)
        return l
    
    @property
    def logp_original(self):
        l = np.array([t.logp_original for t in self.traces])
        if l.dtype.kind != 'f':
            warnings.warn('the array of logp_original does not has dtype of '
                          'float, presumably because different chains have run '
                          'for different lengths.', RuntimeWarning)
        return l
    
    @property
    def input_size(self):
        return self.samples.shape[-1]
    
    @property
    def finished(self):
        return self.traces[0].finished
    
    @property
    def stats(self):
        return [t.stats for t in self.traces] # add StatsTuple?
    
    def get(self, since_iter=None, include_warmup=False, original_space=True,
            return_logp=False, flatten=False):
        if since_iter is None:
            since_iter = 0 if include_warmup else self.n_warmup
        else:
            try:
                since_iter = int(since_iter)
            except:
                raise ValueError('invalid value for since_iter.')
        samples = self.samples_original if original_space else self.samples
        samples = samples[:, since_iter:]
        samples = samples.reshape((-1, self.input_size)) if flatten else samples
        if return_logp:
            logp = self.logp_original if original_space else self.logp
            logp = logp[:, since_iter:]
            logp = logp.flatten() if flatten else logp
            return samples, logp
        else:
            return samples
    
    __call__ = get
    
    def __getitem__(self, key):
        return self._traces.__getitem__(key)

    def __len__(self):
        return self._traces.__len__()
    
    def __iter__(self):
        return self._traces.__iter__()
    
    def __next__(self):
        return self._traces.__next__()

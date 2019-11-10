import numpy as np
from .step_size import DualAverageAdaptation
from .metrics import *
from .stats import SamplerStats
from ...utils.random import check_state
from copy import deepcopy

__all__ = ['Trace']


class Trace:
    
    def __init__(self, x_0, logp_0, random_state=None, step_size=None, 
                 adapt_step_size=True, metric=None, adapt_metric=True, 
                 Emax=1000, target_accept=0.8, gamma=0.05, k=0.75, t0=10, 
                 max_treedepth=-1):
        x_0 = np.asarray(x_0).reshape(-1)
        self._input_size = x_0.shape[0]
        self._samples = [x_0]
        self._stats = SamplerStats(logp_0)
        self._n_iter = 0
        self._n_warmup = 0
        # self._n_call = 0
        self._random_state = check_state(random_state)
        self._random_state_init = deepcopy(self._random_state)
        
        self._Emax = float(Emax)
        if self._Emax <= 0:
            raise ValueError(
                'Emax should be a positive float, instead of {}.'.format(Emax))
        self._max_treedepth = int(max_treedepth)
        
        if isinstance(step_size, DualAverageAdaptation):
            pass
        else:
            if step_size is None:
                step_size = 1.
            step_size = DualAverageAdaptation(
                step_size / x_0.shape[0]**0.25, target_accept, gamma, k, t0, 
                bool(adapt_step_size))
        self._step_size = step_size
        
        if isinstance(metric, QuadMetric):
            pass
        else:
            if metric is None:
                metric = np.ones_like(x_0)
            metric = np.atleast_1d(metric)
            if metric.shape[-1] != x_0.shape[0]:
                raise ValueError('dim of metric is incompatible with x_0.')
            if metric.ndim == 1:
                if adapt_metric:
                    metric = QuadMetricDiagAdapt(x_0.shape[0], x_0, metric, 
                                                 10)
                else:
                    metric = QuadMetricDiag(metric)
            elif metric.ndim == 2:
                if adapt_metric:
                    warnings.warn(
                        'You give a full rank metric array and set '
                        'adapt_metric as True, but we haven\'t implemented '
                        'adaptive full rank metric yet, so an adaptive '
                        'diagonal metric will be used.', RuntimeWarning)
                    metric = QuadMetricDiagAdapt(x_0.shape[0], x_0, 
                                                 np.diag(metric), 10)
                else:
                    metric = QuadMetricFull(metric)
            else:
                raise ValueError('metric should be a QuadMetric, or a 1-d '
                                 'or 2-d array.')
        self._metric = metric
        
        
    @property
    def input_size(self):
        return self._input_size
    
    @property
    def n_iter(self):
        return self._n_iter
    
    @property
    def n_warmup(self):
        return self._n_warmup
    
    '''
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
    '''
    
    @property
    def step_size(self):
        return self._step_size
    
    @property
    def metric(self):
        return self._metric
    
    @property
    def random_state(self):
        return self._random_state
    
    @property
    def random_state_init(self):
        return deepcopy(self._random_state_init)
    
    @property
    def Emax(self):
        return self._Emax
    
    @property
    def max_treedepth(self):
        return self._max_treedepth
    
    @property
    def samples(self):
        return np.array(self._samples)
    
    @property
    def i_iter(self):
        return len(self._samples) - 1
    
    @property
    def logp(self):
        return np.array(self._stats._logp)
    
    @property
    def stats(self):
        return self._stats
    
    def update(self, point, stats):
        self._samples.append(np.copy(point))
        self._stats.update(stats)
    
    def get(self, since_iter=None, include_warmup=False, return_logp=False):
        if since_iter is None:
            since_iter = 1 if include_warmup else self.n_warmup + 1
        else:
            try:
                since_iter = int(since_iter)
            except:
                raise ValueError('invalid value for since_iter.')
        if return_logp:
            return (np.array(self._samples[since_iter:]),
                    np.array(self._stats._logp[since_iter:]))
        else:
            return np.array(self._samples[since_iter:])

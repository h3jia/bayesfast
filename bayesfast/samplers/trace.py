import numpy as np
from .pymc3.step_size import DualAverageAdaptation
from .pymc3.metrics import QuadMetric
from .stats import SamplerStats
from copy import deepcopy

__all__ = ['Trace']


class Trace:
    
    def __init__(self, step_size, metric, random_state, Emax, max_treedepth, 
                 x_0, logp_0):
        if isinstance(step_size, DualAverageAdaptation):
            self._step_size = step_size
        else:
            raise ValueError('step_size should be a DualAverageAdaptation.')
        if isinstance(metric, QuadMetric):
            self._metric = metric
        else:
            raise ValueError('metric should be a QuadMetric.')
        if isinstance(random_state, np.random.RandomState):
            self._random_state = random_state
        else:
            raise ValueError('random_state should be a np.random.RandomState.')
        try:
            self._Emax = float(Emax)
            assert self._Emax > 0
        except:
            raise ValueError(
                'Emax should be a positive float, instead of {}.'.format(Emax))
        try:
            self._max_treedepth = int(max_treedepth)
        except:
            raise ValueError('max_treedepth should be an int, instead of '
                             '{}.'.format(max_treedepth))
        x_0 = np.asarray(x_0).reshape(-1)
        self._input_size = x_0.shape[0]
        self._samples = [x_0]
        self._stats = SamplerStats(logp_0)
        self._n_iter = 0
        self._n_warmup = 0
        self._n_call = 0
        self._random_state_init = deepcopy(self._random_state)
        
    @property
    def input_size(self):
        return self._input_size
    
    @property
    def n_iter(self):
        return self._n_iter
    
    @property
    def n_warmup(self):
        return self._n_warmup
    
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
        return self._samples
    
    @property
    def i_iter(self):
        return len(self._samples) - 1
    
    @property
    def stats(self):
        return self._stats
    
    def update(self, point, stats):
        self._samples.append(np.copy(point))
        self._stats.update(stats)
    
    def get(self, since_iter=None, include_warmup=False):
        if since_iter is None:
            since_iter = 1 if include_warmup else self.n_warmup + 1
        else:
            try:
                since_iter = int(since_iter)
            except:
                raise ValueError('invalid value for since_iter.')
        return self._samples[since_iter:]
    
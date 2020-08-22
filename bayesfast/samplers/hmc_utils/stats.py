from collections import namedtuple, OrderedDict

__all__ = ['HStepStats', 'NStepStats', 'THStepStats', 'TNStepStats',
           'HStats', 'NStats', 'THStats', 'TNStats']


hstats_items = ('logp', 'energy', 'n_int_step', 'accept_stat', 'accepted',
                'step_size', 'step_size_bar', 'warmup', 'energy_change',
                'diverging')


nstats_items = ('logp', 'energy', 'tree_depth', 'tree_size', 'mean_tree_accept',
                'step_size', 'step_size_bar', 'warmup', 'energy_change',
                'max_energy_change', 'diverging')


thstats_items = ('u', 'weight', 'logp', 'energy', 'n_int_step', 'accept_stat',
                 'accepted', 'step_size', 'step_size_bar', 'warmup',
                 'energy_change', 'diverging')


tnstats_items = ('u', 'weight', 'logp', 'energy', 'tree_depth', 'tree_size',
                 'mean_tree_accept', 'step_size', 'step_size_bar', 'warmup',
                 'energy_change', 'max_energy_change', 'diverging')


HStepStats = namedtuple('HStepStats', hstats_items)


NStepStats = namedtuple('NStepStats', nstats_items)


THStepStats = namedtuple('THStepStats', thstats_items)


TNStepStats = namedtuple('TNStepStats', tnstats_items)


class _HStats:
    """Utilities shared by HStats and NStats."""
    def __init__(self):
        for si in self.stats_items:
            setattr(self, '_' + si, [])
    
    def update(self, step_stats):
        if not isinstance(step_stats, self._step_stats):
            raise ValueError('invalid value for step_stats.')
        for si in self.stats_items:
            getattr(self, '_' + si).append(getattr(step_stats, si))
    
    def get(self, since_iter=None, include_warmup=False):
        if since_iter is None:
            since_iter = 0 if include_warmup else self.n_warmup
        else:
            try:
                since_iter = int(since_iter)
            except:
                raise ValueError('invalid value for since_iter.')
        return OrderedDict(
            zip(self.stats_items, [getattr(self, '_' + si)[since_iter:] for si
            in self.stats_items]))
    
    __call__ = get
    
    @property
    def stats_items(self):
        raise NotImplementedError('Abstract property.')
    
    @property
    def n_iter(self):
        return len(self._logp)
    
    @property
    def n_warmup(self):
        return self._warmup.index(False)


class HStats(_HStats):
    """Stats class for the (vanilla) HMC sampler."""
    @property
    def stats_items(self):
        return hstats_items
    
    _step_stats = HStepStats


class NStats(_HStats):
    """Stats class for the NUTS sampler."""
    @property
    def stats_items(self):
        return nstats_items
    
    _step_stats = NStepStats


class _TStats:
    """TODO: should we call it weight or weights?"""
    @property
    def _weights(self):
        return self._weight


class THStats(_HStats, _TStats):
    """Stats class for the THMC sampler."""
    @property
    def stats_items(self):
        return thstats_items
    
    _step_stats = THStepStats


class TNStats(_HStats, _TStats):
    """Stats class for the TNUTS sampler."""
    @property
    def stats_items(self):
        return tnstats_items
    
    _step_stats = TNStepStats

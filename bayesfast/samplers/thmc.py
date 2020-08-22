import numpy as np
from .hmc_utils.base_hmc import BaseTHMC
from .hmc_utils.stats import THStepStats
from .sample_trace import THTrace
from .hmc import HMC

__all__ = ['THMC']


class THMC(BaseTHMC, HMC):
    
    _expected_trace = THTrace
    
    _expected_stats = THStepStats
    
    def _stats(self, state, accept_stat, accepted, energy_change):
        stats = {
            'u': state.u,
            'weight': state.weight,
            'logp': state.logp,
            'energy': state.energy,
            'n_int_step': self.sample_trace.n_int_step,
            'accept_stat': accept_stat,
            'accepted': accepted,
            'energy_change': energy_change,
        }
        return stats

import numpy as np
from .hmc_utils.base_hmc import BaseHMC, HMCStepData, DivergenceInfo
from .hmc_utils.integration import IntegrationError
from .hmc_utils.stats import HStepStats
from .sample_trace import HTrace

__all__ = ['HMC']


class HMC(BaseHMC):
    
    _expected_trace = HTrace
    
    _expected_stats = HStepStats
    
    def _hamiltonian_step(self, start, p0, step_size):
        state = start
        try:
            for _ in range(self.sample_trace.n_int_step):
                state = self.integrator.step(step_size, state)
            if np.isfinite(state.energy):
                energy_change = start.energy - state.energy
                if np.abs(energy_change) > self.sample_trace.max_change:
                    divergence_info = DivergenceInfo(
                        'Divergence encountered, large integration error.',
                        None, state)
                else:
                    divergence_info = None
            else:
                energy_change = -np.inf
                divergence_info = DivergenceInfo(
                    'Divergence encountered, bad energy.', None, state)
        except IntegrationError as err:
            energy_change = -np.inf
            divergence_info = DivergenceInfo('Divergence encountered.', err,
                                             state)
        
        accept_stat = min(1, np.exp(energy_change))
        
        if (divergence_info is not None or
            self.sample_trace.random_generator.uniform() >= accept_stat):
            end = start
            accepted = False
        else:
            end = state
            accepted = True
        
        stats = {
            'logp': state.logp,
            'energy': state.energy,
            'n_int_step': self.sample_trace.n_int_step,
            'accept_stat': accept_stat,
            'accepted': accepted,
            'energy_change': energy_change,
        }
        return HMCStepData(end, accept_stat, divergence_info, stats)

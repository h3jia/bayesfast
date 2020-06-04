import numpy as np
from collections import namedtuple
from .hmc_utils.base_hmc import BaseHMC, HMCStepData, DivergenceInfo
from .hmc_utils.integration import IntegrationError
from .trace import HTrace

__all__ = ['HMC']

# TODO: implement the vanilla HMC sampler


class HMC(BaseHMC):
    
    def __init__(self, logp_and_grad, trace, n_steps, dask_key=None):
        super().__init__(logp_and_grad, trace, dask_key=dask_key)
        self.n_steps = n_steps
        
    _expected_trace = HTrace
    
    def logbern(self, log_p):
        if np.isnan(log_p):
            raise FloatingPointError("log_p can't be nan.")
        return np.log(self._trace.random_state.uniform()) < log_p
        
    def _hamiltonian_step(self, start, p0, step_size):
        prev_state = start
        for _ in range(self.n_steps):
            next_state = self.integrator.step(step_size, prev_state)
        init_energy = start.energy
        prop_energy = next_state.energy
        accept = self.logbern(init_energy-prop_energy)
        divergent = np.isnan(prop_energy)
        if accept and not divergent:
            proposal = next_state
        else:
            proposal = start
        stats = {'logp': proposal.logp, 'energy': prop_energy}
        return HMCStepData(proposal, None, divergent, stats)
#proposal is a State (State = namedtuple("State", 'q, p, v, q_grad, energy, logp'))
#accept_stat is the accept probability to converge to; we modify the step_size to achieve this in NUTS
#
import numpy as np
from collections import namedtuple
from .hmc_utils.base_hmc import BaseTHMC
from .hmc_utils.stats import TNStepStats
from .sample_trace import TNTrace
from .nuts import Tree, NUTS

__all__ = ['TNUTS', 'TTree']


# A proposal for the next position
TProposal = namedtuple("Proposal", "q, u, weight, energy, logp, p_accept")


class TTree(Tree):
    
    def _get_proposal(self, point, p_accept):
        return TProposal(point.q, point.u, point.weight, point.energy,
                         point.logp, p_accept)
    
    def stats(self):
        return {
            'u': self.proposal.u,
            'weight': self.proposal.weight,
            'logp': self.proposal.logp,
            'energy': self.proposal.energy,
            'tree_depth': self.depth,
            'tree_size': self.n_proposals,
            'mean_tree_accept': self.accept_sum / self.n_proposals,
            'energy_change': self.proposal.energy - self.start.energy,
            'max_energy_change': self.max_energy_change,
        }


class TNUTS(BaseTHMC, NUTS):
    
    _expected_trace = TNTrace
    
    _expected_stats = TNStepStats
    
    _expected_tree = TTree

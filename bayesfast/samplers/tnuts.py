import numpy as np
from collections import namedtuple
from .nuts import Tree



# A proposal for the next position
TProposal = namedtuple("Proposal", "q, u, weight, energy, logp, p_accept")


class TTree(Tree):
    
    def _get_proposal(self, point, p_accept):
        return TProposal(point.q, point.u, point.weight, point.energy,
                         point.logp, p_accept)

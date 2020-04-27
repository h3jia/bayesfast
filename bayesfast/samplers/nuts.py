import numpy as np
from collections import namedtuple
from .hmc_utils.base_hmc import BaseHMC, HMCStepData, DivergenceInfo
from .hmc_utils.integration import IntegrationError
from .trace import NTrace

__all__ = ['NUTS']

# TODO: review the code


class NUTS(BaseHMC):
    
    _expected_trace = NTrace
    
    def logbern(self, log_p):
        if np.isnan(log_p):
            raise FloatingPointError("log_p can't be nan.")
        return np.log(self._trace.random_state.uniform()) < log_p
        
    def _hamiltonian_step(self, start, p0, step_size):
        tree = _Tree(len(p0), self.integrator, start, step_size, 
                     self._trace.max_change, self.logbern)

        for _ in range(self._trace._max_treedepth):
            direction = self.logbern(np.log(0.5)) * 2 - 1
            divergence_info, turning = tree.extend(direction)
            if divergence_info or turning:
                break

        stats = tree.stats()
        accept_stat = stats['mean_tree_accept']
        return HMCStepData(tree.proposal, accept_stat, divergence_info, stats)


# A proposal for the next position
Proposal = namedtuple("Proposal", "q, q_grad, energy, p_accept, logp")


# A subtree of the binary tree built by nuts.
Subtree = namedtuple("Subtree", "left, right, p_sum, proposal, log_size, "
                     "accept_sum, n_proposals")


class _Tree:
    
    def __init__(self, ndim, integrator, start, step_size, max_change, logbern):
        self.ndim = ndim
        self.integrator = integrator
        self.start = start
        self.step_size = step_size
        self.max_change = max_change
        self.start_energy = np.array(start.energy)

        self.left = self.right = start
        self.proposal = Proposal(
            start.q, start.q_grad, start.energy, 1.0, start.logp)
        self.depth = 0
        self.log_size = 0
        self.accept_sum = 0
        self.n_proposals = 0
        self.p_sum = start.p.copy()
        self.max_energy_change = 0
        self.logbern = logbern

    def extend(self, direction):
        """Double the treesize by extending the tree in the given direction.

        If direction is larger than 0, extend it to the right, otherwise
        extend it to the left.

        Return a tuple `(diverging, turning)` of type (DivergenceInfo, bool).
        `diverging` indicates, that the tree extension was aborted because
        the energy change exceeded `self.max_change`. `turning` indicates that
        the tree extension was stopped because the termination criterior
        was reached (the trajectory is turning back).
        """
        if direction > 0:
            tree, diverging, turning = self._build_subtree(
                self.right, self.depth, np.asarray(self.step_size))
            leftmost_begin, leftmost_end = self.left, self.right
            rightmost_begin, rightmost_end = tree.left, tree.right
            leftmost_p_sum = self.p_sum
            rightmost_p_sum = tree.p_sum
            self.right = tree.right
        else:
            tree, diverging, turning = self._build_subtree(
                self.left, self.depth, np.asarray(-self.step_size))
            leftmost_begin, leftmost_end = tree.right, tree.left
            rightmost_begin, rightmost_end = self.left, self.right
            leftmost_p_sum = tree.p_sum
            rightmost_p_sum = self.p_sum
            self.left = tree.right

        self.depth += 1
        self.accept_sum += tree.accept_sum
        self.n_proposals += tree.n_proposals

        if diverging or turning:
            return diverging, turning

        size1, size2 = self.log_size, tree.log_size
        if self.logbern(size2 - size1):
            self.proposal = tree.proposal

        self.log_size = np.logaddexp(self.log_size, tree.log_size)
        self.p_sum[:] += tree.p_sum

        # Additional turning check only when tree depth > 0
        # to avoid redundant work
        if self.depth > 0:
            left, right = self.left, self.right
            p_sum = self.p_sum
            turning = (p_sum.dot(left.v) <= 0) or (p_sum.dot(right.v) <= 0)
            p_sum1 = leftmost_p_sum + rightmost_begin.p
            turning1 = ((p_sum1.dot(leftmost_begin.v) <= 0) or 
                        (p_sum1.dot(rightmost_begin.v) <= 0))
            p_sum2 = leftmost_end.p + rightmost_p_sum
            turning2 = ((p_sum2.dot(leftmost_end.v) <= 0) or 
                        (p_sum2.dot(rightmost_end.v) <= 0))
            turning = (turning | turning1 | turning2)

        return diverging, turning

    def _single_step(self, left, epsilon):
        """Perform a leapfrog step and handle error cases."""
        try:
            right = self.integrator.step(epsilon, left)
        except IntegrationError as err:
            error_msg = str(err)
            error = err
        else:
            energy_change = right.energy - self.start_energy
            if np.isnan(energy_change):
                energy_change = np.inf

            if np.abs(energy_change) > np.abs(self.max_energy_change):
                self.max_energy_change = energy_change
            if np.abs(energy_change) < self.max_change:
                p_accept = min(1, np.exp(-energy_change))
                log_size = -energy_change
                proposal = Proposal(
                    right.q, right.q_grad, right.energy, p_accept, right.logp)
                tree = Subtree(right, right, right.p,
                               proposal, log_size, p_accept, 1)
                return tree, None, False
            else:
                error_msg = ("Energy change in leapfrog step is too large: %s."
                             % energy_change)
                error = None
        tree = Subtree(None, None, None, None, -np.inf, 0, 1)
        divergance_info = DivergenceInfo(error_msg, error, left)
        return tree, divergance_info, False

    def _build_subtree(self, left, depth, epsilon):
        if depth == 0:
            return self._single_step(left, epsilon)

        tree1, diverging, turning = self._build_subtree(
            left, depth - 1, epsilon)
        if diverging or turning:
            return tree1, diverging, turning

        tree2, diverging, turning = self._build_subtree(
            tree1.right, depth - 1, epsilon)

        left, right = tree1.left, tree2.right

        if not (diverging or turning):
            p_sum = tree1.p_sum + tree2.p_sum
            turning = (p_sum.dot(left.v) <= 0) or (p_sum.dot(right.v) <= 0)
            # Additional U turn check only when depth > 1 
            # to avoid redundant work.
            if depth > 1:
                p_sum1 = tree1.p_sum + tree2.left.p
                turning1 = ((p_sum1.dot(tree1.left.v) <= 0) or 
                            (p_sum1.dot(tree2.left.v) <= 0))
                p_sum2 = tree1.right.p + tree2.p_sum
                turning2 = ((p_sum2.dot(tree1.right.v) <= 0) or 
                            (p_sum2.dot(tree2.right.v) <= 0))
                turning = (turning | turning1 | turning2)

            log_size = np.logaddexp(tree1.log_size, tree2.log_size)
            if self.logbern(tree2.log_size - log_size):
                proposal = tree2.proposal
            else:
                proposal = tree1.proposal
        else:
            p_sum = tree1.p_sum
            log_size = tree1.log_size
            proposal = tree1.proposal

        accept_sum = tree1.accept_sum + tree2.accept_sum
        n_proposals = tree1.n_proposals + tree2.n_proposals

        tree = Subtree(left, right, p_sum, proposal,
                       log_size, accept_sum, n_proposals)
        return tree, diverging, turning

    def stats(self):
        return {
            'logp': self.proposal.logp,
            'energy': self.proposal.energy,
            'tree_depth': self.depth,
            'tree_size': self.n_proposals,
            'mean_tree_accept': self.accept_sum / self.n_proposals,
            'energy_change': self.proposal.energy - self.start.energy,
            'max_energy_change': self.max_energy_change,
        }
    
import numpy as np
from collections import namedtuple
from .hmc_utils.base_hmc import BaseHMC, HMCStepData, DivergenceInfo
from .hmc_utils.integration import IntegrationError, CpuLeapfrogIntegrator, TLeapfrogIntegrator
from .hmc_utils.stats import NStepStats
from .trace import TTrace

__all__ = ['THMC']

# TODO: review the code


class THMC(BaseHMC):
    
    def __init__(self, logp_and_grad, logbase_and_grad, trace, n_steps, dask_key=None):
        super().__init__(logp_and_grad, trace, dask_key=dask_key)
        self._logbase_and_grad = logbase_and_grad
        self.integrator = TLeapfrogIntegrator(self._trace.metric, logp_and_grad, logbase_and_grad)
        self.n_steps = n_steps
    
    _expected_trace = TTrace
    
    def logbern(self, log_p):
        if np.isnan(log_p):
            raise FloatingPointError("log_p can't be nan.")
        return np.log(self._trace.random_state.uniform()) < log_p
        
    def astep(self):
        """Perform a single HMC iteration."""
        try: 
            q0 = self._trace._samples[-1]
            u0 = self._trace._final_u
            Q0 = np.append(u0, q0)
            p0 = self._trace.metric.random(self._trace.random_state)
            v0 = np.random.normal(0,1) # initialize each step
        except:
            q0 = self._trace.x_0
#             u0 = -1.19097569
            u0 = np.random.normal(0,1) # initialize the first time
            Q0 = np.append(u0, q0)
#             p0 = 1.43270697
#             v0 = -0.3126519
            assert Q0.ndim == 1
        p0 = self._trace.metric.random(self._trace.random_state)
        v0 = np.random.normal(0,1) # initialize each step
        P0 = np.append(v0, p0)
        start = self.integrator.compute_state(Q0, P0)

        if not np.isfinite(start.energy):
            self._trace.metric.raise_ok()
            raise RuntimeError(
                "Bad initial energy, please check the Hamiltonian at p = {}, "
                "q = {}."
                "u = {}."
                "v = {}.".format(p0, q0, u0, v0))
            
        step_size = self._trace.step_size.current(self.warmup)
        # see step_size.py
        hmc_step = self._hamiltonian_step(start, P0, step_size)
        self._trace.step_size.update(hmc_step.accept_stat, self.warmup)
        # see step_size.py
        self._trace.metric.update(hmc_step.end.q, self.warmup)
        step_stats = self.trace._stats.make_stats(**hmc_step.stats, 
                                **self._trace.step_size.sizes(), 
                                warmup=self.warmup, 
                                diverging=bool(hmc_step.divergence_info))
        self._trace.update(hmc_step.end.q, hmc_step.end.u, hmc_step.end.pbeta1, step_stats)
    
    def _hamiltonian_step(self, start, p0, step_size):
        if self.n_steps == None:
            # Use NUTS for each HMC iteration
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
        else:
            # Use regular HMC with n_steps steps:
            new_state = start
            for _ in range(self.n_steps):
                prev_state = new_state
                new_state = self.integrator.step(step_size, prev_state)
            init_energy = start.energy
            prop_energy = new_state.energy
            accept = self.logbern(init_energy-prop_energy)
            divergent = np.isnan(prop_energy)
            if accept and not divergent:
                proposal = new_state
            else:
                proposal = start
            stats = {'logp': proposal.logp, 'energy': prop_energy}
            return HMCStepData(proposal, None, divergent, stats)


# A proposal for the next position
Proposal = namedtuple("Proposal", "q, u, pbeta1, energy, p_accept, logp")


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
            start.q, start.u, start.pbeta1, start.energy, 1.0, start.logp)
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
            turning = (p_sum.dot(left.V) <= 0) or (p_sum.dot(right.V) <= 0)
            p_sum1 = leftmost_p_sum + rightmost_begin.p
            turning1 = ((p_sum1.dot(leftmost_begin.V) <= 0) or 
                        (p_sum1.dot(rightmost_begin.V) <= 0))
            p_sum2 = leftmost_end.p + rightmost_p_sum
            turning2 = ((p_sum2.dot(leftmost_end.V) <= 0) or 
                        (p_sum2.dot(rightmost_end.V) <= 0))
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
                    right.q, right.u, right.pbeta1, right.energy, p_accept, right.logp)
                tree = Subtree(right, right, right.p,
                               proposal, log_size, p_accept, 1)
                return tree, None, False
            else:
                error_msg = ("Energy change in leapfrog step is too large: %s."
                             % energy_change)
                error = None
        tree = Subtree(None, None, None, None, -np.inf, 0, 1)
        divergence_info = DivergenceInfo(error_msg, error, left)
        return tree, divergence_info, False

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
            turning = (p_sum.dot(left.V) <= 0) or (p_sum.dot(right.V) <= 0)
            # Additional U turn check only when depth > 1 
            # to avoid redundant work.
            if depth > 1:
                p_sum1 = tree1.p_sum + tree2.left.p
                turning1 = ((p_sum1.dot(tree1.left.V) <= 0) or 
                            (p_sum1.dot(tree2.left.V) <= 0))
                p_sum2 = tree1.right.p + tree2.p_sum
                turning2 = ((p_sum2.dot(tree1.right.V) <= 0) or 
                            (p_sum2.dot(tree2.right.V) <= 0))
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
    
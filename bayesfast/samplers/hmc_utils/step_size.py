import numpy as np
from scipy import stats
import warnings

__all__ = ['DualAverageAdaptation']

# TODO: review the code


class DualAverageAdaptation:
    
    def __init__(self, initial_step, target, gamma, k, t_0, adapt=True):
        self._log_step = np.log(initial_step)
        self._log_bar = self._log_step
        self._target = target
        self._hbar = 0.
        self._k = k
        self._t_0 = t_0
        self._count = 1
        self._mu = np.log(10. * initial_step)
        self._gamma = gamma
        self._adapt = adapt
        self._accept_after_warmup = []

    def current(self, warmup):
        if warmup:
            return np.exp(self._log_step)
        else:
            return np.exp(self._log_bar)

    def update(self, accept_stat, warmup):
        if not warmup:
            self._accept_after_warmup.append(accept_stat)
            return
        if not self._adapt:
            return

        count, k, t_0 = self._count, self._k, self._t_0
        w = 1. / (count + t_0)
        self._hbar = ((1. - w) * self._hbar + w * (self._target - accept_stat))

        self._log_step = self._mu - self._hbar * np.sqrt(count) / self._gamma
        mk = count ** -k
        self._log_bar = mk * self._log_step + (1. - mk) * self._log_bar
        self._count += 1

    def sizes(self):
        return {
            'step_size': np.exp(self._log_step),
            'step_size_bar': np.exp(self._log_bar),
        }

    def check_acceptance(self, i):
        accept = np.array(self._accept_after_warmup)
        mean_accept = np.mean(accept)
        target_accept = self._target
        # Try to find a reasonable interval for acceptable acceptance
        # probabilities. Finding this was mostry trial and error.
        n_bound = min(100, len(accept))
        n_good, n_bad = mean_accept * n_bound, (1. - mean_accept) * n_bound
        lower, upper = stats.beta(n_good + 1, n_bad + 1).interval(0.95)
        if target_accept < lower or target_accept > upper:
            msg_0 = 'for chain #{}, '.format(i) if i is not None else ''
            msg_1 = (
                'the acceptance probability does not match the target. It is '
                '{}, but should be close to {}. Try to increase the number of '
                'tuning steps.'.format(mean_accept, target_accept))
            warnings.warn(msg_0 + msg_1, RuntimeWarning)

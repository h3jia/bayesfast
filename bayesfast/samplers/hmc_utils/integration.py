import numpy as np
from scipy import linalg
from collections import namedtuple

__all__ = ['CpuLeapfrogIntegrator']

# TODO: review the code


State = namedtuple("State", 'q, p, velocity, q_grad, energy, logp')


class IntegrationError(RuntimeError):

    pass


class CpuLeapfrogIntegrator:
    
    def __init__(self, kinetic, logp_and_grad):
        """Leapfrog integrator using CPU."""
        self._kinetic = kinetic
        self._logp_and_grad = logp_and_grad

    def compute_state(self, q, p):
        """Compute Hamiltonian functions using a position and momentum."""
        logp, grad = self._logp_and_grad(q)
        velocity = self._kinetic.velocity(p)
        kinetic = self._kinetic.energy(p, velocity=velocity)
        energy = kinetic - logp
        return State(q, p, velocity, grad, energy, logp)

    def step(self, epsilon, state):
        """Leapfrog integrator step.

        Half a momentum update, full position update, half momentum update.

        Parameters
        ----------
        epsilon: float, > 0
            step scale
        state: State namedtuple,
            current position data
        out: (optional) State namedtuple,
            preallocated arrays to write to in place

        Returns
        -------
        None if `out` is provided, else a State namedtuple
        """
        try:
            return self._step(epsilon, state)
        except linalg.LinAlgError as err:
            msg = "LinAlgError during leapfrog step."
            raise IntegrationError(msg)
        except ValueError as err:
            # Raised by many scipy.linalg functions
            scipy_msg = "array must not contain infs or nans"
            if len(err.args) > 0 and scipy_msg in err.args[0].lower():
                msg = "Infs or nans in scipy.linalg during leapfrog step."
                raise IntegrationError(msg)
            else:
                raise

    def _step(self, epsilon, state):
        axpy = linalg.blas.get_blas_funcs('axpy')
        pot = self._kinetic

        q_new = state.q.copy()
        p_new = state.p.copy()
        velocity_new = np.empty_like(q_new)

        dt = 0.5 * epsilon

        # p is already stored in p_new
        # p_new = p + dt * q_grad
        axpy(state.q_grad, p_new, a=dt)

        pot.velocity(p_new, out=velocity_new)
        # q is already stored in q_new
        # q_new = q + epsilon * v_new
        axpy(velocity_new, q_new, a=epsilon)

        logp, q_new_grad = self._logp_and_grad(q_new)

        # p_new = p_new + dt * q_new_grad
        axpy(q_new_grad, p_new, a=dt)

        kinetic = pot.velocity_energy(p_new, velocity_new)
        energy = kinetic - logp

        return State(q_new, p_new, velocity_new, q_new_grad, energy, logp)

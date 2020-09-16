import numpy as np
from scipy import linalg
from collections import namedtuple

__all__ = ['CpuLeapfrogIntegrator', 'TCpuLeapfrogIntegrator']

# TODO: review the code


State = namedtuple("State", 'q, p, velocity, q_grad, energy, logp')


TState = namedtuple("TState", 'q, u, p, v, velocity, weight, energy, logp')


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


class TCpuLeapfrogIntegrator(CpuLeapfrogIntegrator):

    def __init__(self, kinetic, logp_and_grad, log_and_grad_base):
        """Leapfrog integrator using CPU for THMC/TNUTS."""
        super().__init__(kinetic, logp_and_grad)
        self._log_and_grad_base = log_and_grad_base

    # Functions for the Hamiltonian
    @staticmethod
    def beta_fun(u):
        """Inverse temperature function."""
        return 1 / (1 + np.exp(-u))

    @staticmethod
    def d_beta_fun(u):
        """Derivative of inverse temperature function."""
        expm = np.exp(-u)
        return expm / (1 + expm)**2

    @staticmethod
    def temp_potential(u):
        """
        Temperature term in the potential.
        Minus log of derivative of beta with respect to u.
        """
        return u + 2 * np.log(1 + np.exp(-u))

    @staticmethod
    def d_temp_potential(u):
        """Derivative of temperature term in potential."""
        exp = np.exp(u)
        return (exp - 1) / (exp + 1)

    def compute_state(self, Q, P):
        """Compute the Hamiltonian for THMC at a position and momentum."""
        u = Q[0]
        q = Q[1:]
        v = P[0]
        p = P[1:]
        phi, dphi = [-x for x in self._logp_and_grad(q)]
        psi, dpsi = [-x for x in self._log_and_grad_base(q)]
        velocity = self._kinetic.velocity(p)
        # mass for tempering variable taken to be 1 for now
        kinetic = self._kinetic.energy(p, velocity=velocity) + v * v / 2
        beta = self.beta_fun(u)
        U = self.temp_potential(u)
        potential = beta * phi + (1 - beta) * psi + U
        energy = kinetic + potential
        logp = -phi
        delta = phi - psi
        weight = 1 if delta==0 else delta / np.expm1(delta)
        return TState(q, u, p, v, velocity, weight, energy, logp)

    def _step(self, epsilon, state):
        """
        Perform one step of the leapfrog integration scheme
        on Hamilton's equations for the THMC Hamiltonian.
        """
        axpy = linalg.blas.get_blas_funcs('axpy')
        kin = self._kinetic

        u_new = np.copy(state.u)
        q_new = np.copy(state.q)
        v_new = np.copy(state.v)
        p_new = np.copy(state.p)
        velocity_new = np.copy(state.velocity)

        # half step
        dt = 0.5 * epsilon

        # advance position one half-step
        # q is already stored in q_new
        # u_new = u + dt * v_new
        # (mass for tempering variable taken to be 1 here)
        u_new += v_new * dt
        # axpy(v_new, u_new, a=dt)
        # q_new = q + dt * velocity_new
        axpy(velocity_new, q_new, a=dt)

        # update derivatives of the potential for momentum full-step
        # and second position half-step
        phi, dphi = [-x for x in self._logp_and_grad(q_new)]
        psi, dpsi = [-x for x in self._log_and_grad_base(q_new)]
        beta = self.beta_fun(u_new)
        d_beta = self.d_beta_fun(u_new)
        dU = self.d_temp_potential(u_new)
        d_pot_du_new = d_beta * (phi - psi) + dU
        d_pot_dq_new = beta * dphi + (1 - beta) * dpsi

        # advance momentum one full step
        # v_new = v - epsilon * d_pot_du
        v_new += -d_pot_du_new * epsilon
        # axpy(-d_pot_du_new, v_new, a=epsilon)
        # p_new = p - epsilon * d_pot_dq
        axpy(-d_pot_dq_new, p_new, a=epsilon)

        # advance position another half-step
        # u_new = u + dt * v_new
        # (mass for tempering variable taken to be 1 here)
        u_new += v_new * dt
        # axpy(v_new, u_new, a=dt)
        # q_new = q + epsilon * velocity_new
        kin.velocity(p_new, out=velocity_new)
        axpy(velocity_new, q_new, a=dt)

        # compute new energy
        kinetic = (self._kinetic.energy(p_new, velocity=velocity_new) + v_new *
                   v_new / 2)
        # mass for tempering variable taken to be 1 for now
        phi, dphi = [-x for x in self._logp_and_grad(q_new)]
        psi, dpsi = [-x for x in self._log_and_grad_base(q_new)]
        beta = self.beta_fun(u_new)
        U = self.temp_potential(u_new)
        potential = beta * phi + (1 - beta) * psi + U
        energy = potential + kinetic

        # compute log of the target
        logp = -phi

        # compute P(beta=1 | x)
        delta = phi - psi
        weight = 1 if delta==0 else delta / np.expm1(delta)

        return TState(q_new, u_new, p_new, v_new, velocity_new, weight, energy,
                      logp)

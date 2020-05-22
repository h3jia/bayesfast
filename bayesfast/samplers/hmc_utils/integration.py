import numpy as np
from scipy import linalg
from collections import namedtuple

__all__ = ['CpuLeapfrogIntegrator']

# TODO: review the code


State = namedtuple("State", 'q, p, v, q_grad, energy, logp')

TState = namedtuple("TState", 'u, q, v, p, V, d_pot_du, d_pot_dq, energy, logp')
# u: tempering variable
# q: input parameter values to model
# v: conjugate momentum to u
# p: conjugate momenta to q
# V: velocity corresponding to p
# d_pot_du: derivative of the potential in the Hamiltonian with respect to u, equal to dH/du
# d_pot_dq: derivative of the potential in the Hamiltonian with respect to q, equal to dH/dq
# energy: energy from Hamiltonian
# logp: log likelihood at q


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
        v = self._kinetic.velocity(p)
        kinetic = self._kinetic.energy(p, velocity=v)
        energy = kinetic - logp
        return State(q, p, v, grad, energy, logp)

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
        v_new = np.empty_like(q_new)

        dt = 0.5 * epsilon

        # p is already stored in p_new
        # p_new = p + dt * q_grad
        axpy(state.q_grad, p_new, a=dt)

        pot.velocity(p_new, out=v_new)
        # q is already stored in q_new
        # q_new = q + epsilon * v_new
        axpy(v_new, q_new, a=epsilon)

        logp, q_new_grad = self._logp_and_grad(q_new)

        # p_new = p_new + dt * q_new_grad
        axpy(q_new_grad, p_new, a=dt)

        kinetic = pot.velocity_energy(p_new, v_new)
        energy = kinetic - logp

        return State(q_new, p_new, v_new, q_new_grad, energy, logp)

class TLeapfrogIntegrator(CpuLeapfrogIntegrator):
    
    def __init__(self, kinetic, logp_and_grad, logprior_and_grad):
        """Leapfrog integrator for THMC."""
        super().__init__(kinetic, logp_and_grad)
        self._logprior_and_grad = logprior_and_grad
    
    # Functions for the Hamiltonian
    def beta_fun(self, u):
        """Inverse temperature function."""
        return 1/(1+np.exp(-(u)))
    def d_beta_fun(self, u):
        """Derivative of inverse temperature function."""
        expm = np.exp(-(u))
        return expm/(1+expm)**2
    def temp_potential(self, u):
        """Temperature term in the potential. Minus log of derivative of beta with respect to u."""
        return u + 2*np.log(1+np.exp(-u))
    def d_temp_potential(self, u):
        """Derivative of temperature term in potential."""
        exp = np.exp(u)
        return (exp-1)/(exp+1)
        
    def compute_state(self, Q, P):
        """Compute Hamiltonian functions for THMC using a position and momentum."""
        u = Q[0]
        q = Q[1:]
        v = P[0]
        p = P[1:]
        phi, dphi = [-x for x in self._logp_and_grad(q)]
        psi, dpsi = [-x for x in self._logprior_and_grad(q)]
        V = self._kinetic.velocity(p)
        kinetic = self._kinetic.energy(p, velocity=V) + v*v/2 # mass for tempering variable taken to be 1 for now
        beta = self.beta_fun(u)
        d_beta = self.d_beta_fun(u)
        U = self.temp_potential(u)
        dU = self.d_temp_potential(u)
        potential = beta*phi + (1-beta)*psi + U
        energy = kinetic + potential
        d_pot_du = d_beta*(phi-psi) + dU
        d_pot_dq = beta*dphi + (1-beta)*dpsi
        logp = -phi
        return TState(u, q, v, p, V, d_pot_du, d_pot_dq, energy, logp)
        
    def _step(self, epsilon, state):
        """Perform one step of the leapfrog integration scheme on Hamilton's equations for the THMC Hamiltonian."""
        axpy = linalg.blas.get_blas_funcs('axpy')
        kin = self._kinetic 

        u_new = state.u.copy()
        q_new = state.q.copy()
        v_new = state.v.copy()
        p_new = state.p.copy()
        V_new = np.empty_like(q_new)
        
        # half step
        dt = 0.5 * epsilon

        # advance momentum one half-step
        # p is already stored in p_new
        # v_new = v - dt * d_pot_du8
        axpy(-state.d_pot_du, v_new, a=dt)
        # p_new = p - dt * d_pot_dq
        axpy(-state.d_pot_dq, p_new, a=dt)

        # advance position one full step
        kin.velocity(p_new, out=V_new)
        # q is already stored in q_new
        # u_new = u + epsilon * v_new (mass for tempering variable taken to be 1 here)
        axpy(v_new, u_new, a=epsilon)
        # q_new = q + epsilon * V_new
        axpy(V_new, q_new, a=epsilon)

        # update derivatives of the potential for second momentum half-step
        phi, dphi = [-x for x in self._logp_and_grad(q_new)]
        psi, dpsi = [-x for x in self._logprior_and_grad(q_new)]
        beta = self.beta_fun(u_new)
        d_beta = self.d_beta_fun(u_new)
        dU = self.d_temp_potential(u_new)
        d_pot_du_new = d_beta*(phi-psi) + dU
        d_pot_dq_new = beta*dphi + (1-beta)*dpsi

        # advance momentum another half-step
        # p is already stored in p_new
        # v_new = v - dt * d_pot_du
        axpy(-d_pot_du_new, v_new, a=dt)
        # p_new = p - dt * d_pot_dq
        axpy(-d_pot_dq_new, p_new, a=dt)

        # compute new energy
        kinetic = self._kinetic.energy(p_new, velocity=V_new) + v_new*v_new/2 # mass for tempering variable taken to be 1 for now
        U = self.temp_potential(u_new)
        potential = beta*phi + (1-beta)*psi + U
        energy = kinetic + potential
        
        # compute log likelihood
        logp = -phi

        return TState(u_new, q_new, v_new, p_new, V_new, d_pot_du_new, d_pot_dq_new, energy, logp)
# import numpyro
import multiprocess
n_dim = 4
n_chain = min(4, multiprocess.cpu_count())
# numpyro.set_host_device_count(n_chain)
import bayesfast as bf
import jax.numpy as jnp
from numpyro.infer import MCMC, NUTS


def f(x):
    return ((x - jnp.arange(n_dim))**2 / 2).sum()


def test_sample():
    bf.random.set_key(0)
    kernel = NUTS(potential_fn=f)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000, num_chains=n_chain,
                progress_bar=False)
    mcmc.run(bf.random.get_key(), init_params=jnp.ones((n_chain, n_dim)))
    posterior_samples = mcmc.get_samples()
    assert jnp.all(jnp.abs(jnp.std(posterior_samples, axis=0) - 1) < 0.05)
    # mcmc.print_summary()

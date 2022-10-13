import jax
import jax.numpy as jnp
import bayesfast as bf


def test_random():
    bf.random.set_key(0)
    a = jax.random.normal(bf.random.get_key(False), (1,))
    b = jax.random.normal(bf.random.get_key(True), (1,))
    c = jax.random.normal(bf.random.get_key(True), (1,))
    d = jax.random.normal(bf.random.get_key(True), (10000,))

    assert jnp.array_equal(a, b)
    assert not jnp.array_equal(b, c)
    assert jnp.isclose(jnp.mean(d), 0, 1, 2e-2)
    assert jnp.isclose(jnp.std(d), 1, 1, 2e-2)

import jax
import jax.numpy as jnp
import bayesfast as bf


def f_poly(x):
    return (x[..., 0]**3 - 2 * x[..., 1]**3 + 3 * x[..., 1] * x[..., 2] * x[..., 3] -
            4 * x[..., 2]**2 * x[..., 3] + 5 * x[..., 0]**2 - 6 * x[..., 0] * x[..., 2] +
            7 * x[..., 1] - 8 )[..., jnp.newaxis]


def test_poly():
    bf.random.set_key(0)
    x = jax.random.normal(bf.random.get_key(), (50, 4))
    y = f_poly(x)
    foo = bf.PolyModel(bf.PolyConfig('<=cubic-3', 4, 1), use_trust=False)
    foo.fit([bf.ModuleCache((x[_],), {}, y[_]) for _ in range(x.shape[0])])
    assert jnp.all(jnp.isclose(y, jnp.array([foo(_) for _ in x])))

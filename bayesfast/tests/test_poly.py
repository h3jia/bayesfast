import numpy as np
import bayesfast as bf
import numdifftools as nd

bf.utils.random.set_generator(0)
rng = bf.utils.random.get_generator()
x = rng.normal(size=(50, 4))


def poly_f(x):
    return (
        x[..., 0]**3 - 2 * x[..., 1]**3 + 3 * x[..., 1] * x[..., 2] * x[..., 3]
        - 4 * x[..., 2]**2 * x[..., 3] + 5 * x[..., 0]**2
        - 6 * x[..., 0] * x[..., 2] + 7 * x[..., 1] - 8
    )[..., np.newaxis]


def test_poly():
    s = bf.modules.PolyModel('cubic-3', input_size=4, output_size=1)
    y = poly_f(x)
    s.fit(x, y)
    y_s = np.concatenate([s(x_i) for x_i in x])
    assert np.isclose(y_s, y).all()
    j = nd.Gradient(lambda x: poly_f(x).flatten())(x[0])[np.newaxis]
    j_s = s.jac(x[0])[0]
    assert np.isclose(j_s, j).all()

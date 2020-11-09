import numpy as np
import bayesfast as bf


def test_sobol_uniform():
    a = np.array([0.5, 0.75, 0.25, 0.375])
    b = bf.utils.sobol.uniform(0, 1, 4).flatten()
    assert np.isclose(a, b).all()

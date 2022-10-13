import numpy as np
import bayesfast as bf
from numdifftools import Gradient, Hessdiag

p = bf.Pipeline(input_scales=[[-10, 10], [-5, 8], [-4, 6], [-8, 6]],
                hard_bounds=[[0, 0], [0, 1], [1, 0], [1, 1]])
x = np.ones(4) * 0.6


def test_constraint_tg():
    a = p.to_original_grad(x)
    b = np.diag(Gradient(p.to_original)(x))
    assert np.isclose(a, b).all()


def test_constraint_tg2():
    a = p.to_original_grad2(x)
    b = np.diag(Hessdiag(p.to_original)(x))
    assert np.isclose(a, b).all()


def test_constraint_fg():
    a = p.from_original_grad(x)
    b = np.diag(Gradient(p.from_original)(x))
    assert np.isclose(a, b).all()


def test_constraint_fg2():
    a = p.from_original_grad2(x)
    b = np.diag(Hessdiag(p.from_original)(x))
    assert np.isclose(a, b).all()

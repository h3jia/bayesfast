import numpy as np
import bayesfast as bf
from scipy.stats import multivariate_normal
import numdifftools as nd


def test_uni_gaussian():
    gaussian = bf.modules.Gaussian(0, 1, lower=None, upper=0)
    truth = multivariate_normal.logpdf(-2, 0, 1) + np.log(2)
    assert np.isclose(gaussian(-2), truth).all()
    assert np.isclose(gaussian.jac(-2),
                      nd.Gradient(lambda x: gaussian(x)[0])(-2)).all()


def test_diag_gaussian():
    gaussian = bf.modules.Gaussian(np.zeros(2), np.ones(2), lower=np.zeros(2),
                                   upper=None)
    truth = (multivariate_normal.logpdf(np.ones(2), np.zeros(2), np.eye(2)) +
             np.log(4))
    assert np.isclose(gaussian(np.ones(2)), truth)
    assert np.isclose(gaussian.jac(np.ones(2)),
                      nd.Gradient(lambda x: gaussian(x)[0])(np.ones(2))).all()


def test_multi_gaussian():
    gaussian = bf.modules.Gaussian(np.zeros(2), np.array(([1, 0.1], [0.1, 1])),
                                   lower=None, upper=None)
    truth = multivariate_normal.logpdf(np.ones(2), np.zeros(2),
                                       np.array(([1, 0.1], [0.1, 1])))
    assert np.isclose(gaussian(np.ones(2)), truth).all()
    assert np.isclose(gaussian.jac(np.ones(2)),
                      nd.Gradient(lambda x: gaussian(x)[0])(np.ones(2))).all()

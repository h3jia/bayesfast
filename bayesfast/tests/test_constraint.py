import bayesfast as bf
import numpy as np
from numdifftools import Gradient, Hessdiag

# TODO: finish comparison


p = bf.Pipeline(input_scales=[[-10, 10], [-5, 8], [-4, 6], [-8, 6]],
                hard_bounds=[[0, 0], [0, 1], [1, 0], [1, 1]])
x = np.ones(4) * 0.6

p.to_original_grad(x), np.diag(Gradient(p.to_original)(x))
p.to_original_grad2(x), np.diag(Hessdiag(p.to_original)(x))
p.from_original_grad(x), np.diag(Gradient(p.from_original)(x))
p.from_original_grad2(x), np.diag(Hessdiag(p.from_original)(x))

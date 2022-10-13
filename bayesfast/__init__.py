# this is a temporary workaround, which enables you to run (up to # of CPU cores) parallel chains
# if you find undesired side effects, you may want to fix it to the default value with
#     import jax.numpy as jnp
#     jnp.ones(2) # or any jax numpy evaluations
# before importing bayesfast, although then the MCMC chains will be drawn sequentially
# you can similarly set it to a larger value if you want to run even more chains in parallel
# we will need to revise this later for GPU support

import multiprocess
import numpyro
numpyro.set_host_device_count(multiprocess.cpu_count())

from .core import *
from .modules import *
from .utils import *

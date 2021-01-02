from .module import Surrogate
from .density import Density, DensityLite
from .sample import sample
from ..modules.poly import PolyConfig, PolyModel
from ..samplers import SampleTrace, NTrace, _HTrace, TraceTuple
from ..samplers import _get_step_size, _get_metric
from ..utils import all_isinstance, Laplace, untemper_laplace_samples
from ..utils.parallel import ParallelBackend, get_backend
from ..utils.random import get_generator
from ..utils.sobol import multivariate_normal
from ..utils import SystematicResampler, integrated_time
from ..utils.collections import VariableDict, PropertyList
from ..evidence import GBS, GIS, GHM
import numpy as np
from collections import namedtuple, OrderedDict
import warnings
from copy import deepcopy
from scipy.special import logsumexp

__all__ = ['BaseStep', 'OptimizeStep', 'SampleStep', 'PostStep', 'Recipe']

# TODO: RecipeTrace.n_call
# TODO: early stop in pipeline evaluation
# TODO: early stop by comparing KL
# TODO: use tqdm to add progress bar for map
# TODO: better control when we don't have enough points before resampling
# TODO: {} as default value
# TODO: monitor the progress of IS
# TODO: improve optimization with trust region?
#       https://arxiv.org/pdf/1804.00154.pdf
# TODO: add checkpoint facility
# TODO: review Recipe.__getstate__


class BaseStep:
    """Utilities shared by `OptimizeStep` and `SampleStep`."""
    def __init__(self, surrogate_list=(), alpha_n=2, fitted=False,
                 sample_trace=None, x_0=None, random_generator=None,
                 reuse_metric=True):
        self.surrogate_list = surrogate_list
        self.alpha_n = alpha_n
        self.fitted = fitted
        self.sample_trace = sample_trace
        self.x_0 = x_0
        self.random_generator = random_generator
        self.reuse_metric = reuse_metric

    @property
    def surrogate_list(self):
        return self._surrogate_list

    @surrogate_list.setter
    def surrogate_list(self, sl):
        if isinstance(sl, Surrogate):
            sl = [sl]
        self._surrogate_list = PropertyList(sl, self._sl_check)

    def _sl_check(self, sl):
        for i, s in enumerate(sl):
            if not isinstance(s, Surrogate):
                raise ValueError('element #{} of surrogate_list is not a '
                                 'Surrogate'.format(i))
        return sl

    @property
    def n_surrogate(self):
        return len(self._surrogate_list)

    @property
    def has_surrogate(self):
        return self.n_surrogate > 0

    @property
    def alpha_n(self):
        return self._alpha_n

    @alpha_n.setter
    def alpha_n(self, a):
        try:
            a = float(a)
        except Exception:
            raise ValueError('alpha_n should be a float.')
        self._alpha_n = a

    @property
    def n_eval(self):
        return int(self._alpha_n *
                   max(su.n_param for su in self._surrogate_list))

    @property
    def x_0(self):
        return self._x_0

    @x_0.setter
    def x_0(self, x):
        if x is None:
            self._x_0 = None
        else:
            try:
                self._x_0 = np.atleast_2d(x).copy()
            except Exception:
                raise ValueError('invalid value for x_0.')

    @property
    def fitted(self):
        return self._fitted

    @fitted.setter
    def fitted(self, f):
        self._fitted = bool(f)

    @property
    def sample_trace(self):
        return self._sample_trace

    @sample_trace.setter
    def sample_trace(self, t):
        if t is None:
            t = {}
        if isinstance(t, dict):
            t = NTrace(**t)
        elif isinstance(t, (SampleTrace, TraceTuple)):
            pass
        else:
            raise ValueError('invalid value for sample_trace.')
        self._sample_trace = t

    @property
    def random_generator(self):
        if self._random_generator is None:
            return get_generator()
        else:
            return self._random_generator

    @random_generator.setter
    def random_generator(self, generator):
        if generator is None:
            self._random_generator = None
        else:
            self._random_generator = np.random.default_rng(generator)

    @property
    def reuse_metric(self):
        return self._reuse_metric

    @reuse_metric.setter
    def reuse_metric(self, rm):
        self._reuse_metric = bool(rm)


class OptimizeStep(BaseStep):
    """Configuring a step for optimization."""
    def __init__(self, surrogate_list=(), alpha_n=2., laplace=None, eps_pp=0.1,
                 eps_pq=0.1, max_iter=5, x_0=None, random_generator=None,
                 fitted=False, run_sampling=True, sample_trace=None,
                 reuse_metric=True):
        super().__init__(surrogate_list, alpha_n, fitted, sample_trace, x_0,
                         random_generator, reuse_metric)
        self.laplace = laplace
        self.eps_pp = eps_pp
        self.eps_pq = eps_pq
        self.max_iter = max_iter
        self.run_sampling = run_sampling

    @property
    def laplace(self):
        return self._laplace

    @laplace.setter
    def laplace(self, lap):
        if lap is None:
            lap = {'beta': 100.}
        if isinstance(lap, dict):
            lap = Laplace(**lap)
        elif isinstance(lap, Laplace):
            pass
        else:
            raise ValueError('laplace should be a Laplace')
        self._laplace = lap

    @property
    def eps_pp(self):
        return self._eps_pp

    @eps_pp.setter
    def eps_pp(self, eps):
        try:
            eps = float(eps)
            assert eps > 0
        except Exception:
            raise ValueError('eps_pp should be a positive float.')
        self._eps_pp = eps

    @property
    def eps_pq(self):
        return self._eps_pq

    @eps_pq.setter
    def eps_pq(self, eps):
        try:
            eps = float(eps)
            assert eps > 0
        except Exception:
            raise ValueError('eps_pq should be a positive float.')
        self._eps_pq = eps

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, mi):
        try:
            mi = int(mi)
            assert mi > 0
        except Exception:
            raise ValueError('max_iter should be a positive int.')
        self._max_iter = mi

    @property
    def run_sampling(self):
        return self._run_sampling

    @run_sampling.setter
    def run_sampling(self, run):
        self._run_sampling = bool(run)


class SampleStep(BaseStep):
    """Configuring a step for sampling."""
    def __init__(self, surrogate_list=(), alpha_n=2., sample_trace=None,
                 resampler={}, reuse_samples=0, reuse_step_size=True,
                 reuse_metric=True, random_generator=None, logp_cutoff=True,
                 alpha_min=0.75, alpha_supp=1.25, x_0=None, fitted=False):
        super().__init__(surrogate_list, alpha_n, fitted, sample_trace, x_0,
                         random_generator, reuse_metric)
        self.resampler = resampler
        self.reuse_samples = reuse_samples
        self.reuse_step_size = reuse_step_size
        self.logp_cutoff = logp_cutoff
        self.alpha_min = alpha_min
        self.alpha_supp = alpha_supp

    @property
    def resampler(self):
        return self._resampler

    @resampler.setter
    def resampler(self, rs):
        if isinstance(rs, dict):
            rs = SystematicResampler(**rs)
        elif rs is None or callable(rs):
            pass
        else:
            raise ValueError('invalid value for resampler.')
        self._resampler = rs

    @property
    def reuse_samples(self):
        return self._reuse_samples

    @reuse_samples.setter
    def reuse_samples(self, rs):
        try:
            self._reuse_samples = int(rs)
        except Exception:
            raise ValueError('invalid value for reuse_samples.')

    @property
    def reuse_step_size(self):
        return self._reuse_step_size

    @reuse_step_size.setter
    def reuse_step_size(self, rss):
        self._reuse_step_size = bool(rss)

    @property
    def logp_cutoff(self):
        return self._logp_cutoff

    @logp_cutoff.setter
    def logp_cutoff(self, lc):
        self._logp_cutoff = bool(lc)

    @property
    def alpha_min(self):
        return self._alpha_min

    @alpha_min.setter
    def alpha_min(self, am):
        try:
            am = float(am)
            assert 0. < am <= 1.
        except Exception:
            raise ValueError('invalid value for alpha_min.')
        self._alpha_min = am

    @property
    def alpha_supp(self):
        return self._alpha_supp

    @alpha_supp.setter
    def alpha_supp(self, asu):
        try:
            asu = float(asu)
            assert asu > 0.
        except Exception:
            raise ValueError('invalid value for alpha_supp.')
        self._alpha_supp = asu

    @property
    def n_eval_min(self):
        return int(self.alpha_min * self.n_eval)


class PostStep:
    """Configuring a step for post-processing."""
    def __init__(self, n_is=0, k_trunc=0.25, evidence_method=None,
                 random_generator=None):
        self.n_is = n_is
        self.k_trunc = k_trunc
        self.evidence_method = evidence_method
        self.random_generator = random_generator

    @property
    def n_is(self):
        return self._n_is

    @n_is.setter
    def n_is(self, n):
        try:
            self._n_is = int(n)
        except Exception:
            raise ValueError('invalid value for n_is.')

    @property
    def k_trunc(self):
        return self._k_trunc

    @k_trunc.setter
    def k_trunc(self, k):
        try:
            self._k_trunc = float(k)
        except Exception:
            raise ValueError('invalid value for k_trunc.')

    @property
    def evidence_method(self):
        return self._evidence_method

    @evidence_method.setter
    def evidence_method(self, em):
        if em is None:
            pass
        elif em == 'GBS':
            em = GBS()
        elif em == 'GIS':
            em = GIS()
        elif em == 'GHM':
            em = GHM()
        elif isinstance(em, dict):
            em = GBS(**em)
        elif hasattr(em, 'run'):
            pass
        else:
            raise ValueError('invalid value for evidence_method.')
        self._evidence_method = em

    @property
    def random_generator(self):
        if self._random_generator is None:
            return get_generator()
        else:
            return self._random_generator

    @random_generator.setter
    def random_generator(self, generator):
        if generator is None:
            self._random_generator = None
        else:
            self._random_generator = np.random.default_rng(generator)


class SampleStrategy:
    """Configuring a multi-step sample strategy."""
    def __init__(self):
        self._i = 0

    def update(self, sample_results):
        raise NotImplementedError('abstract method.')

    @property
    def n_step(self):
        raise NotImplementedError('abstract property.')


class StaticSample(SampleStrategy):
    """Configuring a static multi-step sample strategy."""
    def __init__(self, sample_steps, multiplicity=None, verbose=True):
        super().__init__()
        if multiplicity is not None:
            if not hasattr(sample_steps, '__iter__'):
                warnings.warn('multiplicity is ignored since sample_steps is '
                              'not iterable.', RuntimeWarning)
            else:
                try:
                    sample_steps = [x for i, x in enumerate(sample_steps) for j
                                    in range(multiplicity[i])]
                except Exception:
                    warnings.warn('multiplicity is ignored since I failed to'
                                  'interpret it.', RuntimeWarning)
        self.sample_steps = sample_steps
        self.verbose = verbose

    @property
    def sample_steps(self):
        return self._sample_steps

    @sample_steps.setter
    def sample_steps(self, steps):
        if isinstance(steps, SampleStep):
            self._sample_steps = (deepcopy(steps),)
        elif isinstance(steps, dict):
            self._sample_steps = (SampleStep(**deepcopy(steps)),)
        elif all_isinstance(steps, (SampleStep, dict)) and len(steps) > 0:
            self._sample_steps = list(steps)
            for i, s in enumerate(self._sample_steps):
                s = deepcopy(s)
                if isinstance(s, dict):
                    self._sample_steps[i] = SampleStep(**s)
                else:
                    self._sample_steps[i] = s
            self._sample_steps = tuple(self._sample_steps)
        else:
            raise ValueError('invalid value for sample_steps.')

    @property
    def n_step(self):
        return len(self.sample_steps)

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, v):
        self._verbose = bool(v)

    def update(self, sample_results):
        i_step = len(sample_results)
        if i_step < self.n_step:
            if self.verbose:
                print('\n *** StaticSample: returning the #{} SampleStep. *** '
                      '\n'.format(i_step))
            return deepcopy(self.sample_steps[i_step])
        else:
            if self.verbose:
                print('\n *** StaticSample: iter #{}, no more SampleStep. *** '
                      '\n'.format(i_step))
            return None


class DynamicSample(SampleStrategy):
    """Configuring a static multi-step sample strategy."""
    def __init__(self, *args):
        raise NotImplementedError


RecipePhases = namedtuple('RecipePhases', 'optimize, sample, post')


class RecipeTrace:
    """
    Recording the process of running a Recipe.

    Notes
    -----
    The default behavior of SampleStrategy initialization may change later.
    """
    def __init__(self, optimize=None, sample=None, post=None,
                 sample_multiplicity=None):
        if isinstance(optimize, OptimizeStep) or optimize is None:
            self._s_optimize = deepcopy(optimize)
        elif isinstance(optimize, dict):
            self._s_optimize = OptimizeStep(**deepcopy(optimize))
        else:
            raise ValueError('invalid value for optimize.')

        if isinstance(sample, SampleStrategy):
            self._strategy = sample
        else:
            try:
                # TODO: update this when DynamicSample is ready
                self._strategy = StaticSample(sample, sample_multiplicity)
            except:
                raise ValueError('failed to initialize a StaticSample.')

        self._s_sample = []

        if post is None:
            post = {}
        if isinstance(post, PostStep):
            self._s_post = deepcopy(post)
        elif isinstance(post, dict):
            self._s_post = PostStep(**deepcopy(post))
        else:
            raise ValueError('invalid value for post.')

        self._r_optimize = []
        self._r_sample = []
        self._r_post = None

        self._n_optimize = 0 if self._s_optimize is None else 1
        self._n_sample = self._strategy.n_step
        self._n_post = 0 if self._s_post is None else 1

        self._i_optimize = 0
        self._i_sample = 0
        self._i_post = 0

    @property
    def results(self):
        return RecipePhases(tuple(self._r_optimize), tuple(self._r_sample),
                            self._r_post)

    @property
    def steps(self):
        return RecipePhases(self._s_optimize, self._s_sample, self._s_post)

    @property
    def sample_strategy(self):
        return self._strategy

    @property
    def i(self):
        return RecipePhases(self._i_optimize, self._i_sample, self._i_post)

    @property
    def n(self):
        return RecipePhases(self._n_optimize, self._n_sample, self._n_post)

    # TODO: finish this
    @property
    def n_call(self):
        if self._r_post is None:
            _n_call = 0
            for _opt in self._r_optimize:
                if len(_opt.surrogate_list) > 0:
                    _n_call += len(_opt.var_dicts)
                else:
                    raise NotImplementedError
            for _sam in self._r_sample:
                if len(_sam.surrogate_list) > 0:
                    _n_call += len(_sam.var_dicts)
                else:
                    raise NotImplementedError
            return _n_call
        else:
            return self._r_post.n_call

    # TODO: update this when DynamicSample is ready
    @property
    def finished(self):
        if self._n_sample is not None:
            return RecipePhases(self._i_optimize == self._n_optimize,
                                self._i_sample == self._n_sample,
                                self._i_post == self._n_post)
        else:
            raise NotImplementedError


# I'm not good at naming things... :)
PointDoublet = namedtuple('PointDoublet', 'x, x_trans')


DensityQuartet = namedtuple('DensityQuartet',
                            'logp, logq, logp_trans, logq_trans')


OptimizeResult = namedtuple('OptimizeResult', 'x_max, f_max, surrogate_list, '
                            'var_dicts, laplace_samples, laplace_result, '
                            'samples, sample_trace')


SampleResult = namedtuple('SampleResult', 'samples, surrogate_list, '
                          'var_dicts, sample_trace')


PostResult = namedtuple('PostResult', 'samples, weights, weights_trunc, logp, '
                        'logq, logz, logz_err, x_p, x_q, logp_p, logq_q, '
                        'trace_p, trace_q, n_call, x_max, f_max')


class Recipe:

    def __init__(self, density, parallel_backend=None, recipe_trace=None,
                 optimize=None, sample=None, post=None,
                 sample_multiplicity=None, copy_density=True):
        if isinstance(density, (Density, DensityLite)):
            self._density = deepcopy(density) if copy_density else density
        else:
            raise ValueError('density should be a Density or DensityLite.')

        self.parallel_backend = parallel_backend

        if recipe_trace is None:
            recipe_trace = RecipeTrace(optimize, sample, post,
                                       sample_multiplicity)
        elif isinstance(recipe_trace, RecipeTrace):
            pass
        elif isinstance(recipe_trace, dict):
            recipe_trace = RecipeTrace(**recipe_trace)
        else:
            raise ValueError('recipe_trace should be a RecipeTrace or None.')
        self._recipe_trace = recipe_trace

    def __getstate__(self):
        """We need this to make self._parallel_backend work correctly."""
        self_dict = self.__dict__.copy()
        del self_dict['_parallel_backend'], self_dict['_recipe_trace']
        # TODO: review this
        #       we remove recipe_trace because it contains a PropertyList
        #       that is not pickle-able
        #       however, this may lead to unexpected bahaviors
        #       if one wants to pickle this Recipe object
        return self_dict

    @property
    def density(self):
        return self._density

    @property
    def parallel_backend(self):
        if self._parallel_backend is None:
            return get_backend()
        else:
            return self._parallel_backend

    @parallel_backend.setter
    def parallel_backend(self, backend):
        if backend is None:
            self._parallel_backend = None
        else:
            self._parallel_backend = ParallelBackend(backend)

    @property
    def recipe_trace(self):
        return self._recipe_trace

    def _opt_surro(self, x_0, var_dicts):
        step = self.recipe_trace._s_optimize
        result = self.recipe_trace._r_optimize

        _logp = lambda x: self.density.logp(x, original_space=False,
                                            use_surrogate=True)
        _grad = lambda x: self.density.grad(x, original_space=False,
                                            use_surrogate=True)
        x_0 = self.density.from_original(x_0[0])
        laplace_result = step.laplace.run(logp=_logp, x_0=x_0, grad=_grad)

        x_trans = laplace_result.x_max
        x = self.density.to_original(x_trans)
        x_max = PointDoublet(x, x_trans)

        logp = self.density.logp(x, original_space=True, use_surrogate=False)
        logp_trans = self.density.from_original_density(density=logp, x=x)
        logq_trans = laplace_result.f_max
        logq = self.density.to_original_density(density=logq_trans, x=x)
        f_max = DensityQuartet(float(logp), float(logq), float(logp_trans),
                               float(logq_trans))

        laplace_samples = self.density.to_original(laplace_result.samples)
        surrogate_list = deepcopy(self.density._surrogate_list)
        result.append(
            OptimizeResult(x_max=x_max, f_max=f_max,
            surrogate_list=surrogate_list, var_dicts=var_dicts,
            laplace_samples=laplace_samples, laplace_result=laplace_result,
            samples=None, sample_trace=None))

    def _opt_step(self):
        # DEVELOPMENT NOTES
        # if has surrogate, iterate until convergence
        # if no surrogate, just run on true model
        # in the end, optionally run sampling
        step = self.recipe_trace._s_optimize
        result = self.recipe_trace._r_optimize
        recipe_trace = self.recipe_trace

        if step.has_surrogate:
            if isinstance(self._density, DensityLite):
                raise RuntimeError('self.density should be a Density, instead '
                                   'of DensityLite, for surrogate modeling.')
            self._density.surrogate_list = step._surrogate_list

            if step.fitted:
                if step.x_0 is None:
                    x_0 = np.zeros(self.density.input_size)
                else:
                    x_0 = step.x_0.copy()
                var_dicts = None
            else:
                if step.x_0 is None:
                    dim = self.density.input_size
                    x_0 = multivariate_normal(np.zeros(dim), np.eye(dim),
                                              step.n_eval)
                else:
                    if step.n_eval > 0:
                        if step.x_0.shape[0] < step.n_eval:
                            raise RuntimeError(
                                'I need {} points to fit the surrogate model, '
                                'but you only gave me enough {} points in '
                                'x_0.'.format(step.n_eval, step.x_0.shape[0]))
                        x_0 = step.x_0[:step.n_eval].copy()
                    else:
                        x_0 = step.x_0.copy()
                self.density.use_surrogate = False
                self.density.original_space = True
                with self.parallel_backend:
                    var_dicts = self.parallel_backend.map(self.density.fun, x_0)
                self.density.fit(var_dicts)
            self._opt_surro(x_0, var_dicts)
            _a = result[-1].f_max
            _pq = _a.logp_trans - _a.logq_trans
            print(' OptimizeStep proceeding: iter #0 finished, while current '
                  'logp = {:.3f}, logp_trans = {:.3f}, delta_pq = '
                  '{:.3f}.'.format(_a.logp, _a.logp_trans, _pq))

            for i in range(1, step.max_iter):
                if step.n_eval <= 0:
                    raise RuntimeError('alpha_n should be positive if max_iter '
                                       'is larger than 1.')
                x_0 = result[-1].laplace_samples
                if x_0.shape[0] < step.n_eval:
                    raise RuntimeError(
                        'I need {} points to fit the surrogate model, but I '
                        'can only get {} points from the previous '
                        'iteration.'.format(step.n_eval, x_0.shape[0]))
                x_0 = x_0[:step.n_eval].copy()
                self.density.use_surrogate = False
                self.density.original_space = True
                with self.parallel_backend:
                    var_dicts = self.parallel_backend.map(self.density.fun, x_0)
                self.density.fit(var_dicts)
                self._opt_surro(x_0, var_dicts)
                _a = result[-1].f_max
                _b = result[-2].f_max
                _pp = _a.logp_trans - _b.logp_trans
                _pq = _a.logp_trans - _a.logq_trans
                print(' OptimizeStep proceeding: iter #{} finished, while '
                      'current logp = {:.3f}, logp_trans = {:.3f}, delta_pp = '
                      '{:.3f}, delta_pq = {:.3f}.'.format(i, _a.logp,
                      _a.logp_trans, _pp, _pq))
                if i == step.max_iter - 1:
                    warnings.warn('Optimization did not converge within the max'
                                  ' number of iterations.', RuntimeWarning)
                if (abs(_pp) < step._eps_pp) and (abs(_pq) < step._eps_pq):
                    break

            logp_trans_all = np.asarray([r.f_max.logp_trans for r in result])
            is_max = np.where(logp_trans_all == np.max(logp_trans_all))[0]
            if is_max.size == 1:
                i_max = is_max[0]
            else:
                logq_trans_all = np.asarray(
                    [r.f_max.logq_trans for r in result])
                diff_all = np.abs(logp_trans_all - logq_trans_all)
                i_max = is_max[np.argmin(diff_all[is_max])]

            result.append(result[i_max])
            print(' OptimizeStep proceeding: we will use iter #{} as it has '
                  'the highest logp_trans.\n'.format(i_max))

        else:
            if step.x_0 is None:
                dim = self.density.input_size
                if dim is None:
                    raise RuntimeError('Neither OptimizeStep.x_0 nor Density'
                                       '/DensityLite.input_size is defined.')
                x_0 = np.zeros(dim)
            else:
                x_0 = self.density.from_original(step.x_0[0])
            _logp = lambda x: self.density.logp(x, original_space=False)
            # if self.density.grad is well-defined, we will use it
            # otherwise, we will use finite-difference gradient
            try:
                _grad_0 = self.density.grad(x_0, original_space=False)
                assert np.all(np.isfinite(_grad_0))
                _grad = lambda x: self.density.grad(x, original_space=False)
            except Exception:
                _grad = None
            # TODO: allow user-defined hessian for optimizer?
            # TODO: if generating pseudo random numbers in Laplace
            #       use the random generator of the OptimizeStep?
            laplace_result = step.laplace.run(logp=_logp, x_0=x_0, grad=_grad)

            x_trans = laplace_result.x_max
            x = self.density.to_original(x_trans)
            x_max = PointDoublet(x, x_trans)

            logp_trans = laplace_result.f_max
            logp = self.density.to_original_density(density=logp_trans, x=x_max)
            f_max = DensityQuartet(float(logp), None, float(logp_trans), None)

            laplace_samples = self.density.to_original(laplace_result.samples)
            result.append(
                OptimizeResult(x_max=x_max, f_max=f_max, surrogate_list=(),
                var_dicts=None, laplace_samples=laplace_samples,
                laplace_result=laplace_result, samples=None, sample_trace=None))

        if step.has_surrogate and step.run_sampling:
            self._opt_sample()
        recipe_trace._i_optimize = 1
        print('\n ***** OptimizeStep finished. ***** \n')

    def _opt_sample(self):
        step = self.recipe_trace._s_optimize
        result = self.recipe_trace._r_optimize
        sample_trace = step.sample_trace

        if sample_trace.x_0 is None:
            sample_trace.x_0 = result[-1].laplace_samples
            sample_trace._x_0_transformed = False
        if step.reuse_metric:
            cov = result[-1].laplace_result.cov.copy()
            if sample_trace._metric == 'diag':
                sample_trace._metric = np.diag(cov)
            elif sample_trace._metric == 'full':
                sample_trace._metric = cov

        self._density.surrogate_list = result[-1].surrogate_list
        self._density.use_surrogate = True
        t = sample(self.density, sample_trace=sample_trace,
                   parallel_backend=self.parallel_backend)
        x = t.get(flatten=True)
        result[-1] = result[-1]._replace(samples=x, sample_trace=t)
        print('\n *** Finished sampling the surrogate density defined by the '
              'selected OptimizeStep. *** \n')

    def _sam_step(self):
        steps = self.recipe_trace._s_sample
        results = self.recipe_trace._r_sample
        recipe_trace = self.recipe_trace

        i = recipe_trace._i_sample
        this_step = recipe_trace._strategy.update(results)

        while this_step is not None:
            sample_trace = this_step.sample_trace
            get_prev_step = not (i == 0 and not recipe_trace._i_optimize)
            get_prev_samples = get_prev_step or (this_step.x_0 is not None)

            if get_prev_step:
                if i == 0:
                    prev_result = recipe_trace._r_optimize[-1]
                    prev_step = recipe_trace._s_optimize
                else:
                    prev_result = results[i - 1]
                    prev_step = steps[i - 1]

            get_prev_density = (get_prev_step and this_step.x_0 is None and
                                prev_step.sample_trace is not None)

            if get_prev_samples:
                if this_step.x_0 is None:
                    if prev_result.samples is None:
                        prev_samples = untemper_laplace_samples(
                            prev_result.laplace_result)
                        prev_transformed = True
                    else:
                        prev_samples = prev_result.samples
                        prev_transformed = False
                else:
                    prev_samples = this_step.x_0
                    prev_transformed = False

            if get_prev_density:
                prev_density = prev_result.sample_trace.get(return_type='logp',
                                                            flatten=True)

            if isinstance(sample_trace, _HTrace):
                if sample_trace.x_0 is None and get_prev_samples:
                    sample_trace.x_0 = prev_samples
                    sample_trace._x_0_transformed = prev_transformed

                if get_prev_step:
                    if sample_trace._step_size is None:
                        if (this_step.reuse_step_size and
                            prev_result.sample_trace is not None):
                            sample_trace._step_size = _get_step_size(
                                prev_result.sample_trace)

                    if (sample_trace._metric == 'diag' or
                        sample_trace._metric == 'full'):
                        if (this_step.reuse_metric and
                            prev_result.sample_trace is not None):
                            sample_trace._metric = _get_metric(
                                prev_result.sample_trace, sample_trace._metric)

            if this_step.has_surrogate:
                if not isinstance(self._density, Density):
                    raise RuntimeError('self.density should be a Density for '
                                       'surrogate modeling.')
                self._density.surrogate_list = this_step._surrogate_list

                if this_step._fitted:
                    var_dicts = None

                else:
                    if not get_prev_samples:
                        raise RuntimeError('You did not give me samples to fit '
                                           'the surrogate model.')

                    if (this_step.n_eval > 0 and
                        prev_samples.shape[0] < this_step.n_eval):
                        raise RuntimeError(
                            'I need {} points to fit the surrogate model, but I'
                            ' can find at most {} points.'.format(
                            this_step.n_eval, prev_samples.shape[0]))

                    if i > 0 and not prev_step.has_surrogate:
                        warnings.warn(
                            'you are doing surrogate modeling after sampling '
                            'the true density. Please make sure this is what '
                            'you want.', RuntimeWarning)

                    if get_prev_density:
                        if this_step.resampler is None:
                            i_resample = np.arange(this_step.n_eval)
                        else:
                            i_resample = this_step.resampler(prev_density,
                                                             this_step.n_eval)

                    else:
                        if (this_step.resampler is not None or
                            this_step.logp_cutoff):
                            warnings.warn('resampler and logp_cutoff will be '
                                          'ignored, when get_prev_density is '
                                          'False.', RuntimeWarning)
                        if this_step.n_eval > 0:
                            i_resample = np.arange(this_step.n_eval)
                        else:
                            i_resample = np.arange(prev_samples.shape[0])

                    x_fit = prev_samples[i_resample]
                    self.density.use_surrogate = False
                    self.density.original_space = True
                    with self.parallel_backend:
                        var_dicts = np.asarray(
                            self.parallel_backend.map(self.density.fun, x_fit))
                    var_dicts_fit = var_dicts.copy()

                    if this_step.reuse_samples:
                        for j in range(i):
                            if (j + this_step.reuse_samples >= i or
                                this_step.reuse_samples < 0):
                                var_dicts_fit = np.concatenate(
                                    (var_dicts_fit, results[j].var_dicts))

                    if this_step.logp_cutoff and get_prev_density:
                        logp_fit = np.concatenate(
                            [vd.fun[self.density.density_name] for vd in
                            var_dicts_fit])
                        logq_fit = prev_density[i_resample]
                        logq_min = np.min(logq_fit)
                        np.delete(prev_samples, i_resample, axis=0)
                        np.delete(prev_density, i_resample, axis=0)

                        is_good = logp_fit > logq_min
                        n_good = np.sum(is_good)
                        f_good = n_good / logp_fit.size
                        if f_good < 0.5:
                            warnings.warn('more than half of the samples are '
                                          'abandoned because their logp < '
                                          'logq_min.', RuntimeWarning)
                        if f_good == 0.:
                            raise RuntimeError(
                                'f_good is 0, indicating that the samples seem '
                                'very bad. Please check your recipe setup. You '
                                'may also want to try logp_cutoff=False for the'
                                ' SampleStep.')

                        var_dicts_fit = var_dicts_fit[is_good]
                        while len(var_dicts_fit) < this_step.n_eval_min:
                            n_eval_supp = ((this_step.n_eval_min -
                                           len(var_dicts_fit)) / f_good *
                                           this_step.alpha_supp)
                            n_eval_supp = max(int(n_eval_supp), 4)
                            if prev_samples.shape[0] < n_eval_supp:
                                raise RuntimeError('I do not have enough '
                                                   'supplementary points.')
                            if this_step.resampler is None:
                                i_resample = np.arange(n_eval_supp)
                            else:
                                i_resample = this_step.resampler(
                                    prev_density, n_eval_supp)

                            x_fit = prev_samples[i_resample]
                            self.density.use_surrogate = False
                            self.density.original_space = True
                            with self.parallel_backend:
                                var_dicts_supp = np.asarray(
                                    self.parallel_backend.map(self.density.fun,
                                    x_fit))
                            logp_supp = np.concatenate(
                                [vd.fun[self.density.density_name] for vd in
                                var_dicts_supp])
                            np.delete(prev_samples, i_resample, axis=0)
                            np.delete(prev_density, i_resample, axis=0)

                            is_good = logp_supp > logq_min
                            n_good = np.sum(is_good)
                            if n_good < logp_supp.size / 2:
                                warnings.warn(
                                    'more than half of the samples are '
                                    'abandoned because their logp < logq_min.',
                                    RuntimeWarning)
                            var_dicts = np.concatenate((var_dicts,
                                                        var_dicts_supp))
                            var_dicts_fit = np.concatenate(
                                (var_dicts_fit, var_dicts_supp[is_good]))

                    self.density.fit(var_dicts_fit)

                self.density.use_surrogate = True
                t = sample(self.density, sample_trace=sample_trace,
                           parallel_backend=self.parallel_backend)
                x = t.get(flatten=True)
                surrogate_list = deepcopy(self._density._surrogate_list)
                results.append(SampleResult(
                    samples=x, surrogate_list=surrogate_list,
                    var_dicts=var_dicts, sample_trace=t))

            else:
                if isinstance(self._density, Density):
                    self.density.use_surrogate = False
                t = sample(self.density, sample_trace=sample_trace,
                           parallel_backend=self.parallel_backend)
                x = t.get(flatten=True)
                results.append(SampleResult(samples=x, surrogate_list=(),
                                            var_dicts=None, sample_trace=t))

            steps.append(this_step)
            print('\n *** SampleStep proceeding: iter #{} finished. *** '
                  '\n'.format(i))

            recipe_trace._i_sample += 1
            i = recipe_trace._i_sample
            this_step = recipe_trace._strategy.update(results)

        print('\n ***** SampleStep finished. ***** \n')

    def _pos_step(self):
        step = self.recipe_trace._s_post
        recipe_trace = self.recipe_trace

        x_p = None
        x_q = None

        f_logp = None
        f_logq = None

        logp_p = None
        logq_q = None

        x_max = None
        f_max = None

        samples = None
        weights = None
        weights_trunc = None
        logp = None
        logq = None

        trace_p = None
        trace_q = None

        logz = None
        logz_err = None

        if recipe_trace._i_optimize:
            opt_result = recipe_trace._r_optimize[-1]
            x_max = opt_result.x_max
            f_max = opt_result.f_max

        if recipe_trace._i_sample:
            prev_step = recipe_trace._s_sample[-1]
            prev_result = recipe_trace._r_sample[-1]

            if prev_step.has_surrogate:
                trace_q = prev_result.sample_trace
                x_q = trace_q.get(return_type='samples', flatten=False)
                logq_q = trace_q.get(return_type='logp', flatten=False)
                self.density._surrogate_list = prev_step.surrogate_list

            else:
                trace_p = prev_result.sample_trace
                x_p = trace_p.get(return_type='samples', flatten=False)
                logp_p = trace_p.get(return_type='logp', flatten=False)

        elif recipe_trace._i_optimize:
            prev_step = recipe_trace._s_optimize
            prev_result = recipe_trace._r_optimize[-1]

            if prev_step.has_surrogate and prev_result.sample_trace is not None:
                # sample_trace in OptimizeStep will be ignored
                # if has_surrogate is False
                trace_q = prev_result.sample_trace
                x_q = trace_q.get(return_type='samples', flatten=False)
                logq_q = trace_q.get(return_type='logp', flatten=False)
                self.density._surrogate_list = prev_step.surrogate_list

            else:
                warnings.warn('no existing samples found.', RuntimeWarning)

        else:
            raise RuntimeError('you have run neither OptimizeStep nor '
                               'SampleStep before the PostStep.')

        if x_p is not None:
            samples = x_p.reshape((-1, x_p.shape[-1]))
            weights = np.ones(samples.shape[0])
            weights_trunc = weights
            logp = logp_p.reshape(-1)

            if step.evidence_method is not None:
                logz, logz_err = step.evidence_method(
                    x_p=trace_p, logp=self._f_logp, logp_p=logp_p)
            if step.n_is > 0:
                warnings.warn('n_is will not be used when we already have exact'
                              ' samples from logp.', RuntimeWarning)

        elif x_q is not None:
            samples = x_q.reshape((-1, x_q.shape[-1]))
            logq = logq_q.reshape(-1)

            if step.n_is != 0:
                if step.n_is < 0 or step.n_is > samples.shape[0]:
                    if step.n_is > 0:
                        warnings.warn(
                            'you set n_is as {}, but I can only get {} samples '
                            'from the previous step, so I will use all these '
                            'samples to do IS for now.'.format(step.n_is,
                            samples.shape[0]), RuntimeWarning)
                    n_is = samples.shape[0]
                else:
                    n_is = step.n_is
                    foo = int(samples.shape[0] / n_is)
                    samples = samples[::foo][:n_is]
                    logq = logq[::foo][:n_is]

                self.density.use_surrogate = False
                self.density.original_space = True
                with self.parallel_backend:
                    logp = np.asarray(
                        self.parallel_backend.map(self.density.logp,
                        samples)).reshape(-1)
                weights = np.exp(logp - logq)
                if step.k_trunc < 0:
                    weights_trunc = weights.copy()
                else:
                    weights_trunc = np.clip(weights, 0, np.mean(weights) *
                                            n_is**step.k_trunc)

                if step.evidence_method is not None:
                    logz_q, logz_err_q = step.evidence_method(
                        x_p=trace_q, logp=self._f_logq, logp_p=logq_q)
                    logz_pq = logsumexp(logp - logq, b=1 / logp.size)
                    foo = np.exp(logp - logq - logz_pq)
                    tau = float(integrated_time(foo))
                    logz_err_pq = (
                        np.var(foo) / np.mean(foo)**2 / logp.size * tau)**0.5
                    logz = logz_q + logz_pq
                    logz_err = (logz_err_q**2 + logz_err_pq**2)**0.5

            else:
                weights = np.ones(samples.shape[0])
                weights_trunc = weights

                if step.evidence_method is not None:
                    warnings.warn('since n_is is 0, we are computing the '
                                  'evidence of logq, which may differ from the '
                                  'evidence of logp.', RuntimeWarning)
                    logz, logz_err = step.evidence_method(
                        x_p=trace_q, logp=self._f_logq, logp_p=logq_q)

        else:
            if (step.n_is is not None) or (step.evidence_method is not None):
                warnings.warn('n_is and evidence_method will not be used when '
                              'we only have Laplace samples.', RuntimeWarning)

        try:
            n_call = recipe_trace.n_call + step.n_is
            warnings.warn('as of now, n_call does not take the possible logp '
                          'calls during evidence evaluation into account.',
                          RuntimeWarning)
        except Exception:
            n_call = None
        recipe_trace._r_post = PostResult(
            samples, weights, weights_trunc, logp, logq, logz, logz_err, x_p,
            x_q, logp_p, logq_q, trace_p, trace_q, n_call, x_max, f_max)
        recipe_trace._i_post = 1
        print('\n ***** PostStep finished. ***** \n')

    def _f_logp(self, x):
        return self.density.logp(x, original_space=True, use_surrogate=False)

    def _f_logq(self, x):
        return self.density.logp(x, original_space=True, use_surrogate=True)

    def run(self):
        f_opt, f_sam, f_pos = self.recipe_trace.finished
        if not f_opt:
            self._opt_step()
        if not f_sam:
            self._sam_step()
        if not f_pos:
            self._pos_step()

    def get(self):
        try:
            return self.recipe_trace._r_post
        except Exception:
            raise RuntimeError('you have not run a PostStep.')

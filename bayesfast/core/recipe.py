from .module import Surrogate
from .density import Density, DensityLite
from .sample import sample
from ..modules.poly import PolyConfig, PolyModel
from ..samplers import SampleTrace, NTrace, _HTrace, TraceTuple
from ..samplers import _get_step_size, _get_metric
from ..utils import threadpool_limits, check_client, all_isinstance
from ..utils import Laplace, untemper_laplace_samples
from ..utils.random import check_state, split_state, multivariate_normal
from ..utils import SystematicResampler
from ..utils.collections import VariableDict, PropertyList
import numpy as np
from distributed import Client
from collections import namedtuple, OrderedDict
import warnings
from copy import deepcopy

__all__ = ['BaseStep', 'OptimizeStep', 'SampleStep', 'PostStep', 'Recipe']


# TODO: property.setter?
# TODO: RecipeTrace.n_call
# TODO: early stop in pipeline evaluation
# TODO: early stop by comparing KL
# TODO: use tqdm to add progress bar for _map_fun
# TODO: better control when we don't have enough points before resampling
# TODO: allow IS over hmc_samples in OptimizeStep
# TODO: review the choice of x_0 for SampleStep
# TODO: monitor the progress of IS
# TODO: improve optimization with trust region?
#       https://arxiv.org/pdf/1804.00154.pdf
# TODO: add checkpoint facility

class BaseStep:
    """Utilities shared by `OptimizeStep` and `SampleStep`."""
    def __init__(self, surrogate_list=[], alpha_n=2, fitted=False,
                 sample_trace=None, x_0=None, random_state=None,
                 reuse_metric=True):
        self._set_surrogate_list(surrogate_list)
        self._set_alpha_n(alpha_n)
        self._set_fitted(fitted)
        self._set_sample_trace(sample_trace)
        self._set_random_state(random_state)
        self._set_reuse_metric(reuse_metric)
    
    @property
    def surrogate_list(self):
        return self._surrogate_list
    
    def _set_surrogate_list(self, sl):
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
    
    def _set_alpha_n(self, a):
        try:
            a = float(a)
            assert a <= 0
        except:
            raise ValueError('alpha_n should be a positive float.')
        self._alpha_n = a
    
    @property
    def n_eval(self):
        return int(self._alpha_n * 
                   max(su.n_param for su in self._surrogate_list))
    
    @property
    def x_0(self):
        return self._x_0
    
    def _set_x_0(self, x):
        if x is None:
            self._x_0 = None
        else:
            try:
                self._x_0 = np.atleast_2d(x).copy()
            except:
                raise ValueError('invalid value for x_0.')
    
    @property
    def fitted(self):
        return self._fitted
    
    def _set_fitted(self, f):
        self._fitted = bool(f)
    
    @property
    def sample_trace(self):
        return self._sample_trace
    
    def _set_sample_trace(self, t):
        if t is None:
            t = NTrace()
        elif isinstance(t, (SampleTrace, TraceTuple)):
            pass
        else:
            raise ValueError('invalid value for sample_trace.')
        self._sample_trace = t
    
    @property
    def random_state(self):
        return self._random_state
    
    def _set_random_state(self, state):
        if state is None:
            self._random_state = None
        else:
            self._random_state = check_state(state)
    
    @property
    def reuse_metric(self):
        return self._reuse_metric
    
    def _set_reuse_metric(self, rm):
        self._reuse_metric = bool(rm)


class OptimizeStep(BaseStep):
    """Configuring a step for optimization."""
    def __init__(self, surrogate_list=[], alpha_n=2., laplace=None, eps_pp=0.1,
                 eps_pq=0.1, max_iter=10, x_0=None, random_state=None,
                 fitted=False, run_sampling=True, sample_trace=None,
                 reuse_metric=True):
        super().__init__(surrogate_list, alpha_n, fitted, sample_trace, x_0,
                         random_state, reuse_metric)
        self._set_laplace(laplace)
        self._set_eps_pp(eps_pp)
        self._set_eps_pq(eps_pq)
        self._set_max_iter(max_iter)
        self._set_x_0(x_0)
        self._set_run_sampling(run_sampling)
    
    @property
    def laplace(self):
        return self._laplace
    
    def _set_laplace(self, laplace):
        if laplace is None:
            laplace = Laplace(beta=100.)
        elif isinstance(laplace, Laplace):
            pass
        else:
            raise ValueError('laplace should be a Laplace')
        self._laplace = laplace
    
    @property
    def eps_pp(self):
        return self._eps_pp
    
    def _set_eps_pp(self, eps):
        try:
            eps = float(eps)
            assert eps > 0
        except:
            raise ValueError('eps_pp should be a positive float.')
        self._eps_pp = eps
    
    @property
    def eps_pq(self):
        return self._eps_pq
    
    def _set_eps_pq(self, eps):
        try:
            eps = float(eps)
            assert eps > 0
        except:
            raise ValueError('eps_pq should be a positive float.')
        self._eps_pq = eps
    
    @property
    def max_iter(self):
        return self._max_iter
    
    def _set_max_iter(self, mi):
        try:
            mi = int(mi)
            assert mi > 0
        except:
            raise ValueError('max_iter should be a positive int.')
        self._max_iter = mi
    
    @property
    def run_sampling(self):
        return self._run_sampling
    
    def _set_run_sampling(self, run):
        self._run_sampling = bool(run)


class SampleStep(BaseStep):
    """Configuring a step for sampling."""
    def __init__(self, surrogate_list=[], alpha_n=2., sample_trace=None,
                 resampler={}, reuse_samples=0, reuse_step_size=True,
                 reuse_metric=True, random_state=None, logp_cutoff=True,
                 alpha_min=1.5, alpha_supp=0.1, x_0=None, fitted=False):
        super().__init__(surrogate_list, alpha_n, fitted, sample_trace, x_0,
                         random_state, reuse_metric)
        self._set_resampler(resampler)
        self._set_reuse_samples(reuse_samples)
        self._set_reuse_step_size(reuse_step_size)
        self._set_logp_cutoff(logp_cutoff)
        self._set_alpha_min(alpha_min)
        self._set_alpha_supp(alpha_supp)
    
    @property
    def resampler(self):
        return self._resampler
    
    def _set_resampler(self, rs):
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
    
    def _set_reuse_samples(self, rs):
        try:
            self._reuse_samples = int(rs)
        except:
            raise ValueError('invalid value for reuse_samples.')
    
    @property
    def reuse_step_size(self):
        return self._reuse_step_size
    
    def _set_reuse_step_size(self, rss):
        self._reuse_step_size = bool(rss)
    
    @property
    def logp_cutoff(self):
        return self._logp_cutoff
    
    def _set_logp_cutoff(self, lc):
        self._logp_cutoff = bool(lc)
    
    @property
    def alpha_min(self):
        return self._alpha_min
    
    def _set_alpha_min(self, am):
        try:
            am = float(am)
            assert am > 0 and am < self._alpha_n
        except:
            raise ValueError('invalid value for alpha_min.')
        self._alpha_min = am
    
    @property
    def alpha_supp(self):
        return self._alpha_supp
    
    def _set_alpha_supp(self, asu):
        try:
            asu = float(asu)
            assert asu > 0
        except:
            raise ValueError('invalid value for alpha_supp.')
        self._alpha_supp = asu
    
    @property
    def n_eval_min(self):
        return int(self._alpha_min * 
                   max(su.n_param for su in self._surrogate_list))
    
    @property
    def n_eval_supp(self):
        return int(self._alpha_supp * 
                   max(su.n_param for su in self._surrogate_list))


class PostStep:
    """Configuring a step for post-processing."""
    def __init__(self, n_is=0, k_trunc=0.25):
        self._set_n_is(n_is)
        self._set_k_trunc(k_trunc)
    
    @property
    def n_is(self):
        return self._n_is
    
    def _set_n_is(self, n):
        try:
            self._n_is = int(n)
        except:
            raise ValueError('invalid value for n_is.')
    
    @property
    def k_trunc(self):
        return self._k_trunc
    
    def _set_k_trunc(self, k):
        try:
            self._k_trunc = float(k)
        except:
            raise ValueError('invalid value for k_trunc.')


RecipePhases = namedtuple('RecipePhases', 'optimize, sample, post')


class RecipeTrace:
    """Recording the process of running a Recipe."""
    def __init__(self, optimize=None, sample=[], post=None):
        if isinstance(optimize, OptimizeStep) or optimize is None:
            self._s_optimize = optimize
        else:
            raise ValueError('optimize should be an OptimizeStep or None.')
        
        if isinstance(sample, SampleStep):
            self._s_sample = [sample]
        elif all_isinstance(sam, SampleStep):
            self._s_sample = tuple(sample)
        else:
            raise ValueError('sample should be a SampleStep, or consists of '
                             'SampleStep(s).')
        
        if isinstance(post, PostStep) or post is None:
            self._s_post = post
        else:
            raise ValueError('post should be a PostStep or None.')
        
        self._r_optimize = []
        self._r_sample = []
        self._r_post = None
        
        self._n_optimize = 0 if self._s_optimize is None else 1
        self._n_sample = len(self._s_sample)
        self._n_post = 0 if self._s_post is None else 1
        
        self._i_optimize = 0
        self._i_sample = 0
        self._i_post = 0
    
    @property
    def result(self):
        return RecipePhases(tuple(self._r_optimize), tuple(self._r_sample),
                            self._r_post)
    
    @property
    def steps(self):
        return RecipePhases(self._s_optimize, self._s_sample, self._s_post)
    
    @property
    def i(self):
        return RecipePhases(self._i_optimize, self._i_sample, self._i_post)
    
    @property
    def n(self):
        return RecipePhases(self._n_optimize, self._n_sample, self._n_post)
    
    # TODO: finish this
    @property
    def n_call(self):
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
        _pos = self._r_post
        if _pos is not None and _pos.weights is not None:
            _n_call += len(_pos.weights)
        return _n_call
    
    @property
    def finished(self):
        return RecipePhases(self._i_optimize == self._n_optimize, 
                            self._i_sample == self._n_sample,
                            self._i_post == self._n_post)


# I'm not good at naming things...
PointDoublet = namedtuple('PointDoublet', 'x, x_trans')


DensityQuartet = namedtuple('DensityQuartet', 
                            'logp, logq, logp_trans, logq_trans')


OptimizeResult = namedtuple('OptimizeResult', 'x_max, f_max, surrogate_list, '
                            'var_dicts, laplace_samples, laplace_result, '
                            'samples, sample_trace')


SampleResult = namedtuple('SampleResult', 'samples, surrogate_list, '
                          'var_dicts, sample_trace')


PostResult = namedtuple('PostResult', 'samples, weights, logp, logq, '
                        'samples_raw, weights_raw')


class Recipe:
    
    def __init__(self, density, client=None, recipe_trace=None, optimize=None, 
                 sample=[], post=None, copy_density=True):
        if isinstance(density, (Density, DensityLite)):
            self._density = deepcopy(density) if copy_density else density
        else:
            raise ValueError('density should be a Density or DensityLite.')
        
        self.client = client
        
        if recipe_trace is None:
            recipe_trace = RecipeTrace(optimize, sample, post)
        elif isinstance(recipe_trace, RecipeTrace):
            pass
        else:
            raise ValueError('recipe_trace should be a RecipeTrace or None.')
        self._recipe_trace = recipe_trace
    
    @property
    def density(self):
        return self._density
    
    @property
    def client(self):
        return self._client
    
    @client.setter
    def client(self, clt):
        if isinstance(clt, (int, Client)) or clt is None:
            self._client = clt
        else:
            raise ValueError('invalid value for client.')
    
    @property
    def recipe_trace(self):
        return self._recipe_trace
    
    @classmethod
    def _map_fun(cls, client, density, x):
        foo = client.map(density.fun, x)
        return client.gather(foo)
    
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
        if step._random_state is None:
            step._set_random_state(check_state(None))
        
        if step.has_surrogate:
            if isinstance(self._density, DensityLite):
                raise RuntimeError('self.density should be a Density, instead '
                                   'of DensityLite, for surrogate modeling.')
            self._density.surrogate_list = step._surrogate_list
            
            if step.fitted:
                x_0 = None
                var_dicts = None
            else:
                if step.x_0 is None:
                    dim = self.density.input_size
                    x_0 = multivariate_normal(np.zeros(dim), np.eye(dim),
                                              step.n_eval)
                else:
                    if step.x_0.shape[0] < step.n_eval:
                        raise RuntimeError(
                            'I need {} points to fit the surrogate model, but '
                            'you only gave me enough {} points in x_0.'.format(
                            step.n_eval, step.x_0.shape[0]))
                    x_0 = step.x_0[:step.n_eval].copy()
                var_dicts = self._map_fun(self._client, self._density, x_0)
                self._density.fit(var_dicts)
            self._opt_surro(x_0, var_dicts)
            print(' OptimizeStep proceeding: iter #0 finished.')
            
            for i in range(1, step._max_iter):
                x_0 = result[-1].laplace_samples
                if x_0.shape[0] < step.n_eval:
                    raise RuntimeError(
                        'I need {} points to fit the surrogate model, but I '
                        'can only get {} points from the previous '
                        'iteration.'.format(step.n_eval, x_0.shape[0]))
                x_0 = x_0[:step.n_eval].copy()
                var_dicts = self._map_fun(self._client, self._density, x_0)
                self._density.fit(var_dicts)
                self._opt_surro(x_0, var_dicts)
                _a = result[-1].f_max
                _b = result[-2].f_max
                _pp = _a.logp_trans - _b.logp_trans
                _pq = _a.logp_trans - _a.logq_trans
                print(' OptimizeStep proceeding: iter #{} finished, while '
                      'delta_pp = {:.3f}, delta_pq = {:.3f}.'.format(i, _pp, 
                      _pq))
                if (abs(_pp) < step._eps_pp) and (abs(_pq) < step._eps_pq):
                    break
            if i == step._max_iter - 1:
                warnings.warn('Optimization did not converge within the max '
                              'number of iterations.', RuntimeWarning)

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
            except:
                _grad = None
            # TODO: allow user-defined hessian for optimizer?
            # TODO: if generating pseudo random numbers in Laplace
            #       use the RandomState of the OptimizeStep?
            laplace_result = step.laplace.run(logp=_logp, x_0=x_0, grad=_grad)
            
            x_trans = laplace_result.x_max
            x = self.density.to_original(x_trans)
            x_max = PointDoublet(x, x_trans)
            
            logp_trans = laplace_result.f_max
            logp = self.density.to_original_density(density=logp_trans, x=x_max)
            f_max = DensityQuartet(float(logp), None, float(logp_trans), None)
            
            laplace_samples = self.density.to_original(laplace_result.samples)
            result.append(
                OptimizeResult(x_max=x_max, f_max=f_max, surrogate_list=[],
                var_dicts=None, laplace_samples=laplace_samples,
                laplace_result=laplace_result, samples=None, sample_trace=None))
        
        if step.has_surrogate and step.sample_trace is not None:
            self._opt_sample()
        result._i_optimize = 1
        print('\n ***** OptimizeStep finished. ***** \n')
    
    def _opt_sample(self):
        step = self.recipe_trace._s_optimize
        result = self.recipe_trace._r_optimize
        sample_trace = step.sample_trace
        
        if sample_trace._x_0 is None:
            sample_trace._set_x_0(result[-1].laplace_samples)
            sample_trace._x_0_transformed = False
        if sample_trace._random_state is None:
            sample_trace._set_random_state(step._random_state)
        if step.reuse_metric:
            cov = result[-1].laplace_result.cov.copy()
            if sample_trace._metric == 'diag':
                sample_trace._metric = np.diag(cov)
            elif sample_trace._metric == 'full':
                sample_trace._metric = cov
        
        old_list = self._density.surrogate_list
        self._density.surrogate_list = result[-1].surrogate_list
        t = sample(self.density, sample_trace=sample_trace, client=self.client)
        x = t.get(flatten=True)
        step._random_state = deepcopy(t[0]._random_state)
        result[-1] = result[-1]._replace(samples=x, sample_trace=t)
        self._density.surrogate_list = old_list
        print('\n *** Finished sampling the surrogate density defined by the '
              'last OptimizeStep. *** \n')
    
    def _sam_step(self):
        steps = self.recipe_trace._s_sample
        result = self.recipe_trace._r_sample
        recipe_trace = self.recipe_trace
        
        for i in range(recipe_trace._i_sample, recipe_trace._n_sample):
            this_step = steps[i]
            sample_trace = this_step.sample_trace
            has_prev_step = not (i == 0 and not recipe_trace._i_optimize)
            has_prev_samples = has_prev_step or this_step.x_0 is not None
            
            if has_prev_step:
                if i == 0:
                    prev_result = recipe_trace._r_optimize
                    prev_step = recipe_trace._s_optimize
                else:
                    prev_result = result[-1]
                    prev_step = steps[-1]
            
            has_prev_density = (has_prev_step and this_step.x_0 is None and
                                prev_step.sample_trace is not None)
            
            if has_prev_samples:
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
            
            if has_prev_density:
                prev_density = prev_result.sample_trace.get(return_logp=True,
                                                            flatten=True)
            
            if this_step._random_state is None:
                if has_prev_step:
                    this_step._set_random_state(
                        deepcopy(prev_step._random_state))
                else:
                    this_step._set_random_state(check_state(None))
            
            if isinstance(sample_trace, _HTrace):
                if sample_trace._x_0 is None and has_prev_samples:
                    sample_trace._set_x_0(prev_samples)
                    sample_trace._x_0_transformed = prev_transformed
                
                if sample_trace._random_state is None:
                    _1, _2 = split_state(this_step._random_state)
                    this_step._random_state = _1
                    sample_trace._set_random_state(_2)
                
                if has_prev_step:
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
                if isinstance(self._density, Density):
                    self._density.use_surrogate = True
                else:
                    raise RuntimeError('self.density should be a Density for '
                                       'surrogate modeling.')
                self._density.surrogate_list = this_step._surrogate_list
                
                if this_step._fitted:
                    var_dicts = None
                
                else:
                    if not has_prev_samples:
                        raise RuntimeError('You did not give me samples to fit '
                                           'the surrogate model.')
                    
                    if prev_samples.shape[0] < this_step.n_eval:
                        raise RuntimeError(
                            'I need {} points to fit the surrogate model, but I'
                            ' can find at most {} points.'.format(
                            this_step.n_eval, prev_samples.shape[0]))
                    
                    if i > 0 and not prev_step.has_surrogate:
                        warnings.warn(
                            'you are doing surrogate modeling after sampling '
                            'the true density. Please make sure this is what '
                            'you want.', RuntimeWarning)
                    
                    if has_prev_density:
                        if this_step.resampler is None:
                            i_resample = np.arange(this_step.n_eval)
                        else:
                            i_resample = this_step.resampler(prev_density,
                                                             this_step.n_eval)
                    
                    else:
                        if (this_step.resampler is not None or
                            this_step.logp_cutoff):
                            warnings.warn('resampler and logp_cutoff will be '
                                          'ignored, when has_prev_density is '
                                          'False.', RuntimeWarning)
                        i_resample = np.arange(this_step.n_eval)
                    
                    x_fit = prev_samples[i_resample]
                    var_dicts = self._map_fun(self.client, self.density, x_fit)
                    var_dicts_all = var_dicts.copy()
                    
                    if this_step.reuse_samples:
                        for j in range(i):
                            if (j + this_step.reuse_samples >= i or
                                this_step.reuse_samples < 0):
                                var_dicts_all.extend(result[j].var_dicts)
                    
                    if this_step.logp_cutoff and has_prev_density:
                        logp_all = np.concatenate(
                            [vd.fun[self.density.extract_name] for vd in
                            var_dicts_all])
                        logq_fit = prev_density[i_resample]
                        logq_min = np.min(logq_fit)
                        np.delete(prev_samples, i_resample, axis=0)
                        np.delete(prev_density, i_resample, axis=0)
                        
                        is_good = logp_all > logq_min
                        n_good = np.sum(is_good)
                        if n_good < logp_all.size / 2:
                            warnings.warn('more than half of the samples are '
                                          'abandoned because their logp < '
                                          'logq_min.', RuntimeWarning)
                        
                        var_dicts_all = var_dicts[is_good]
                        while len(var_dicts_all) < this_step.n_eval_min:
                            if prev_samples.shape[0] < this_step.n_eval_supp:
                                raise RuntimeError('I do not have enough '
                                                   'supplementary points.')
                            if this_step.resampler is None:
                                i_resample = np.arange(this_step.n_eval_supp)
                            else:
                                i_resample = this_step.resampler(
                                    prev_density, this_step.n_eval_supp)
                            
                            x_fit = prev_samples[i_resample]
                            var_dicts_supp = self._map_fun(self.client, 
                                                           self.density, x_fit)
                            logp_supp = np.concatenate(
                                [vd.fun[self.density.extract_name] for vd in
                                var_dicts_supp])
                            np.delete(prev_samples, i_resample, axis=0)
                            np.delete(prev_density, i_resample, axis=0)
                            var_dicts.extend(var_dicts_supp)
                            
                            is_good = logp_supp > logq_min
                            n_good = np.sum(is_good)
                            if n_good < logp_supp.size / 2:
                                warnings.warn(
                                    'more than half of the samples are '
                                    'abandoned because their logp < logq_min.',
                                    RuntimeWarning)
                            var_dicts_all.extend(var_dicts_supp[is_good])
                    
                    self.density.fit(var_dicts_all)
                
                t = sample(self._density, sample_trace=sample_trace,
                           client=self._client)
                x = t.get(flatten=True)
                surrogate_list = deepcopy(self._density._surrogate_list)
                result.append(SampleResult(
                    samples=x, surrogate_list=surrogate_list, 
                    var_dicts=var_dicts, sample_trace=t))
            
            else:
                if isinstance(self._density, Density):
                    self._density.use_surrogate = False
                t = sample(self._density, sample_trace=sample_trace,
                           client=self._client)
                x = t.get(flatten=True)
                result.append(SampleResult(samples=x, surrogate_list=[], 
                                           var_dicts=None, sample_trace=t))
            
            result._i_sample += 1
            step._random_state = deepcopy(t[0]._random_state)
            print('\n *** SampleStep proceeding: iter #{} finished. *** '
                  '\n'.format(i))
        print(' ***** SampleStep finished. ***** \n')
    
    def _pos_step(self):
        step = self.recipe_trace._s_post
        result = self.recipe_trace._r_post
        
        if result.n.sample:
            _samples = result.result.sample[-1].samples
            samples = _samples.reshape((-1, _samples.shape[-1]))
            _logq = np.array([self.density.to_original_density(
                             *t.get(return_logp=True)[::-1]) for t in 
                             result.result.sample[-1].trace])
            logq = _logq.reshape(-1)
            if steps.n_is == 0:
                result.append(
                    PostResult(samples, None, None, logq, _samples, None))
            else:
                if steps.n_is < 0:
                    n_is = samples.shape[0]
                elif steps.n_is > 0:
                    if not steps.n_is <= samples.shape[0]:
                        warnings.warn(
                            'we do not have enough samples to do IS as you '
                            'requested. We will only do IS for the existing '
                            'samples.', RuntimeWarning)
                        n_is = samples.shape[0]
                    else:
                        n_is = steps.n_is
                    foo = int(samples.shape[0] / n_is)
                    samples = samples[::foo][:n_is]
                    logq = logq[::foo][:n_is]
                else:
                    raise RuntimeError('unexpected value for steps.n_is.')
                var_dicts = self._map_fun(
                    self._client, self._density, samples)
                logp = np.concatenate(
                    [vd._fun[self._density._density_name] for vd in 
                    var_dicts])
                weights_raw = np.exp(logp - logq)
                weights_raw = np.where(np.isfinite(weights_raw), weights_raw, 0)
                if steps.k_trunc < 0:
                    weights = weights_raw.copy()
                else:
                    weights = np.clip(weights_raw, 0, np.mean(weights_raw) * 
                                      n_is**steps.k_trunc)
                result.append(PostResult(
                    samples, weights, logp, logq, _samples, weights_raw))

        elif result.n.optimize:
            raise NotImplementedError

        else:
            raise RuntimeError(
                'the recipe has neither OptimizeStep nor SampleStep.')
        
        print(' ***** PostStep finished. ***** \n')
    
    def run(self):
        try:
            old_client = self._client
            self._client, _new_client = check_client(client)
            f_opt, f_sam, f_pos = self._result.finished
            if not f_opt:
                self._opt_step()
            if not f_sam:
                self._sam_step()
            if not f_pos:
                self._pos_step()
            
        finally:
            if _new_client:
                self._client.cluster.close()
                self._client.close()
                self._client = old_client
    
    def get(self):
        try:
            return self._result.result.post[0]
        except:
            raise RuntimeError('you have not run a PostStep.')

from .module import Surrogate
from .density import Density, DensityLite
from .sample import sample
from ..modules.poly import PolyConfig, PolyModel
from ..utils import Laplace, threadpool_limits, check_client, all_isinstance
from ..utils.random import check_state, resample, multivariate_normal
from ..utils.collections import VariableDict, PropertyList
import numpy as np
from distributed import Client
from collections import namedtuple, OrderedDict
import warnings
from copy import deepcopy

__all__ = ['BaseStep', 'OptimizeStep', 'SampleStep', 'PostStep', 'Recipe']


# TODO: early stop in pipeline evaluation
# TODO: use tqdm to add progress bar for _map_fun
# TODO: better control when we don't have enough points before resampling
# TODO: allow IS over hmc_samples in OptimizeStep
# TODO: review the choice of x_0 for SampleStep
# TODO: monitor the progress of IS
# TODO: improve optimization with trust region? 
#       https://arxiv.org/pdf/1804.00154.pdf

class BaseStep:
    """Utilities shared by `OptimizeStep` and `SampleStep`."""
    def __init__(self, surrogate_list=[], fit_options={}, alpha_n=2, 
                 sample_options={}, prefit=False):
        self.surrogate_list = surrogate_list
        
        if isinstance(fit_options, dict):
            self._fit_options = [
                fit_options for i in range(max(1, self.n_surrogate))] ########################################################### ?
        elif (hasattr(fit_options, '__iter__') and
              all_isinstance(fit_options, dict)):
            self._fit_options = list(fit_options)
            if len(self._fit_options) < self.n_surrogate:
                self._fit_options.extend([{} for i in range(
                    self.n_surrogate - len(self._fit_options))])
        else:
            raise ValueError(
                'fit_options should be a dict or consist of dict(s).')
        
        self.alpha_n = alpha_n
        
        if not isinstance(sample_options, dict):
            raise ValueError('sample_options should be a dict.')
        self._sample_options = sample_options
        
        self._prefit = bool(prefit)
    
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
    def fit_options(self):
        return tuple(self._fit_options)
    
    @property
    def alpha_n(self):
        return self._alpha_n
    
    @alpha_n.setter
    def alpha_n(self, a):
        a = float(a)
        if a <= 0:
            raise ValueError('alpha_n should be positive.')
        self._alpha_n = a
    
    @property
    def n_eval(self):
        return int(self._alpha_n * 
                   max(su.n_param for su in self._surrogate_list))
    
    @property
    def sample_options(self):
        return self._sample_options
    
    @property
    def prefit(self):
        return self._prefit


class OptimizeStep(BaseStep):
    
    def __init__(self, surrogate_list=[], fit_options={}, alpha_n=2.,
                 sample_options={'beta': 0.01}, prefit=False, eps_pp=0.1,
                 eps_pq=0.1, max_iter=10, run_hmc=False, hmc_options={}):
        super().__init__(surrogate_list, fit_options, alpha_n, sample_options,
                         prefit)
        self.eps_pp = eps_pp
        self.eps_pq = eps_pq
        self.max_iter = max_iter
        self.run_hmc = run_hmc
        
    @property
    def eps_pp(self):
        return self._eps_pp
    
    @eps_pp.setter
    def eps_pp(self, eps):
        eps = float(eps)
        if eps <= 0:
            raise ValueError('eps_pp should be a positive float.')
        self._eps_pp = eps
    
    @property
    def eps_pq(self):
        return self._eps_pq
    
    @eps_pq.setter
    def eps_pq(self, eps):
        eps = float(eps)
        if eps <= 0:
            raise ValueError('eps_pq should be a positive float.')
        self._eps_pq = eps
    
    @property
    def max_iter(self):
        return self._max_iter
    
    @max_iter.setter
    def max_iter(self, mi):
        mi = int(mi)
        if mi < 2:
            raise ValueError('max_iter should be larger than 1.')
        self._max_iter = mi
    
    @property
    def run_hmc(self):
        return self._run_hmc
    
    @run_hmc.setter
    def run_hmc(self, run):
        self._run_hmc = bool(run)
    
    @property
    def hmc_options(self):
        return self._hmc_options
    
    @hmc_options.setter
    def hmc_options(self, ho):
        if not isinstance(ho, dict):
            raise ValueError('hmc_options should be a dict.')
        self._hmc_options = ho


class SampleStep(BaseStep):
    
    def __init__(self, surrogate_list=[], fit_options={}, alpha_n=2.,
                 sample_options={}, prefit=False, resample_options={},
                 reuse_steps=0, logp_cutoff=True, alpha_min=1.5,
                 alpha_supp=0.1, adapt_metric=False):
        super().__init__(surrogate_list, fit_options, alpha_n, sample_options,
                         prefit)
        
        if not isinstance(resample_options, dict):
            raise ValueError('resample_options should be a dict.')
        self._resample_options = resample_options
        
        self._reuse_steps = int(reuse_steps)
        self._logp_cutoff = bool(logp_cutoff)
        
        alpha_min = float(alpha_min)
        if alpha_min <= 0:
            raise ValueError('alpha_min should be a positive float.')
        self._alpha_min = alpha_min
        
        alpha_supp = float(alpha_supp)
        if alpha_supp <= 0:
            raise ValueError('alpha_supp should be a positive float.')
        self._alpha_supp = alpha_supp
        
        self._adapt_metric = bool(adapt_metric)
        
    @property
    def resample_options(self):
        return self._resample_options
    
    @property
    def reuse_steps(self):
        return self._reuse_steps
    
    @property
    def logp_cutoff(self):
        return self._logp_cutoff
    
    @property
    def alpha_min(self):
        return self._alpha_min
    
    @property
    def alpha_supp(self):
        return self._alpha_supp
    
    @property
    def adapt_metric(self):
        return self._adapt_metric
    
    @property
    def n_eval_min(self):
        return int(self._alpha_min * 
                   max(su.n_param for su in self._surrogate_list))
    
    @property
    def n_eval_supp(self):
        return int(self._alpha_supp * 
                   max(su.n_param for su in self._surrogate_list))


class PostStep:
    
    def __init__(self, n_is=None, k_trunc=None):
        self.n_is = n_is
        self.k_trunc = k_trunc
    
    @property
    def n_is(self):
        return self._n_is
    
    @n_is.setter
    def n_is(self, n):
        self._n_is = 0 if (n is None) else int(n)
    
    @property
    def k_trunc(self):
        return self._k_trunc
    
    @k_trunc.setter
    def k_trunc(self, k):
        self._k_trunc = 0.25 if (k_trunc is None) else float(k_trunc)


RecipePhases = namedtuple('RecipePhases', 'optimize, sample, post')


class RecipeResult:
        
    def __init__(self, optimize=None, sample=[], post=None, x_0=None, 
                 random_state=None, input_size=None):
        if isinstance(optimize, OptimizeStep) or optimize is None:
            self._s_optimize = optimize
        else:
            raise ValueError('optimize should be a OptimizeStep or None.')
        
        if isinstance(sample, SampleStep):
            self._s_sample = [sample]
        elif (hasattr(sample, '__iter__') and 
              all(isinstance(sam, SampleStep) for sam in sample)):
            self._s_sample = list(sample)
        else:
            raise ValueError('sample should be a SampleStep, or consists of '
                             'SampleStep(s).')
        
        if isinstance(post, PostStep):
            pass
        elif post is None:
            if len(self._s_sample):
                post = PostStep()
        else:
            raise ValueError('post should be a PostStep or None.')
        self._s_post = post
        
        self._d_optimize = []
        self._d_sample = []
        self._d_post = []
        
        self._n_optimize = 0 if self._s_optimize is None else 1
        self._n_sample = len(self._s_sample)
        self._n_post = 0 if self._s_post is None else 1
        
        self._i_optimize = 0
        self._i_sample = 0
        self._i_post = 0
        
        self._random_state = check_state(random_state)
        self._random_state_init = deepcopy(self._random_state)
        
        input_size = int(input_size)
        if input_size < 1:
            raise ValueError('input_size should be larger than 1.')
        self._input_size = input_size
        
        if x_0 is None:
            warnings.warn('you did not give me x_0, so I will use standard '
                          'Gaussian when needed. If this fails, you should '
                          'consider give me some better x_0.', RuntimeWarning)
            self._x_0 = x_0
        else:
            x_0 = np.atleast_2d(x_0)
            if x_0.shape[-1] != self._input_size:
                raise ValueError(
                    'the shape of x_0 is not consistent with the density.')
            self._x_0 = x_0.reshape((-1, self._input_size))
    
    @property
    def data(self):
        return RecipePhases(self._d_optimize, self._d_sample, self._d_post)
    
    @property
    def steps(self):
        return RecipePhases(self._s_optimize, self._s_sample, self._s_post)
    
    @property
    def i(self):
        return RecipePhases(self._i_optimize, self._i_sample, self._i_post)
    
    @property
    def n(self):
        return RecipePhases(self._n_optimize, self._n_sample, self._n_post)
    
    @property
    def n_call(self):
        _n_call = 0
        for _opt in self.data.optimize:
            if len(_opt.surrogate_list) > 0:
                _n_call += len(_opt.var_dicts)
            else:
                raise NotImplementedError
        for _sam in self.data.sample:
            if len(_sam.surrogate_list) > 0:
                _n_call += len(_sam.var_dicts)
            else:
                raise NotImplementedError
        for _pos in self.data.post:
            if _pos.weights is None:
                pass
            else:
                _n_call += len(_pos.weights)
        return _n_call
        
    
    @property
    def x_0(self):
        return self._x_0
    
    @property
    def random_state(self):
        return self._random_state
    
    @property
    def random_state_init(self):
        return deepcopy(self._random_state_init)
    
    @property
    def input_size(self):
        return self._input_size
    
    @property
    def finished(self):
        return RecipePhases(self._i_optimize == self._n_optimize, 
                            self._i_sample == self._n_sample,
                            self._i_post == self._n_post)


DensityQuartet = namedtuple('DensityQuartet', 
                            'logp, logq, logp_trans, logq_trans')
    

OptimizeResult = namedtuple('OptimizeResult', 'x_max, f_max, samples, '
                            'surrogate_list, hmc_samples, var_dicts, '
                            'Laplace, trace')


SampleResult = namedtuple('SampleResult', 'samples, surrogate_list, '
                          'var_dicts, trace')


PostResult = namedtuple('PostResult', 'samples, weights, logp, logq, '
                        'samples_raw, weights_raw')


class Recipe:
    
    def __init__(self, density, client=None, result=None, optimize=None, 
                 sample=[], post=None, x_0=None, random_state=None):
        if isinstance(density, (Density, DensityLite)):
            self._density = density
        else:
            raise ValueError('density should be a Density or DensityLite.')
        
        self.client = client
        
        if result is None:
            self._result = RecipeResult(optimize, sample, post, x_0, 
                                        random_state, self._density.input_size)
        elif isinstance(result, RecipeResult):
            if result.input_size == self._density.input_size:
                self._result = result
            else:
                raise ValueError('the input_size of the result is inconsistent '
                                 'with the Recipe.')
        else:
            raise ValueError('result should be a RecipeResult or None.')
    
    @property
    def density(self):
        return self._density
    
    @property
    def client(self):
        return self._client
    
    @client.setter
    def client(self, clt):
        self._client = clt
    
    @property
    def input_size(self):
        return self._density.input_size
    
    @property
    def result(self):
        return self._result
    
    @property
    def n_call(self):
        return self._result.n_call
    
    @classmethod
    def _map_fun(cls, client, density, x):
        foo = client.map(density.fun, x)
        return client.gather(foo)
    
    @property
    def logp(self):
        return self._density.logp
    
    @property
    def grad(self):
        return self._density.grad
    
    @property
    def logp_and_grad(self):
        return self._density.logp_and_grad
    
    @property
    def to_original(self):
        return self._density.to_original
    
    @property
    def to_original_grad(self):
        return self._density.to_original_grad
    
    @property
    def to_original_grad2(self):
        return self._density.to_original_grad2
    
    @property
    def to_original_density(self):
        return self._density.to_original_density
    
    @property
    def from_original(self):
        return self._density.from_original
    
    @property
    def from_original_grad(self):
        return self._density.from_original_grad
    
    @property
    def from_original_grad2(self):
        return self._density.from_original_grad2
        
    @property
    def from_original_density(self):
        return self._density.from_original_density
    
    def _opt_surro(self, x_0, var_dicts):
        steps = self.result.steps.optimize
        data = self.result.data.optimize
        
        _logp = lambda x: self.logp(x, use_surrogate=True, original_space=False)
        _grad = lambda x: self.grad(x, use_surrogate=True, original_space=False)
        x_0 = self.from_original(x_0[0])
        laplace = Laplace(_logp, x_0, grad=_grad)
        lap_res = laplace.run(**steps._sample_options)
        
        x_max = self.to_original(lap_res.x_max)
        logp = self.logp(x_max, use_surrogate=False, original_space=True)
        logq_trans = lap_res.f_max
        logp_trans = self.from_original_density(density=logp, x=x_max)
        logq = self.to_original_density(density=logq_trans, x=x_max)
        f_max = DensityQuartet(float(logp), float(logq), float(logp_trans), 
                               float(logq_trans))
        samples = self.to_original(lap_res.samples)
        
        surrogate_list = deepcopy(self._density._surrogate_list)
        data.append(
            OptimizeResult(x_max=x_max, f_max=f_max, samples=samples, 
            surrogate_list=surrogate_list, hmc_samples=None, 
            var_dicts=var_dicts, Laplace=lap_res, trace=None))
    
    def _opt_step(self):
        # DEVELOPMENT NOTES
        # if has surrogate, iterate until convergence
        # if no surrogate, just run on true model
        # in the end, optionally run hmc
        result = self.result
        steps = self.result.steps.optimize
        data = self.result.data.optimize
        data.clear()
        
        if steps.has_surrogate:
            if isinstance(self._density, DensityLite):
                raise RuntimeError('self.density should be a Density, instead '
                                   'of DensityLite, for surrogate modeling.')
            self._density.surrogate_list = steps._surrogate_list
            
            if steps._prefit:
                x_0 = None
                var_dicts = None
            else:
                if result._x_0 is None:
                    x_0 = multivariate_normal(np.zeros(self.input_size), 
                                              np.eye(self.input_size), 
                                              steps.n_eval)
                else:
                    if result._x_0.shape[0] < steps.n_eval:
                        raise RuntimeError(
                            'I need {} points to fit the surrogate model, but '
                            'you only gave me enough {} points in x_0.'.format(
                            steps.n_eval, result._x_0.shape[0]))
                    x_0 = result._x_0[:steps.n_eval].copy()
                var_dicts = self._map_fun(self._client, self._density, x_0)
                self._density.fit(var_dicts, **steps._fit_options[0])
            self._opt_surro(x_0, var_dicts)
            print(' OptimizeStep proceeding: iter #0 finished.')
            
            for i in range(1, steps._max_iter):
                x_0 = data[-1].samples
                if x_0.shape[0] < steps.n_eval:
                    raise RuntimeError(
                        'I need {} points to fit the surrogate model, but I '
                        'can only get {} points from the previous '
                        'iteration.'.format(steps.n_eval, x_0.shape[0]))
                x_0 = x_0[:steps.n_eval].copy()
                var_dicts = self._map_fun(self._client, self._density, x_0)
                self._density.fit(var_dicts, **steps._fit_options[0])
                self._opt_surro(x_0, var_dicts)
                _a = data[-1].f_max
                _b = data[-2].f_max
                _pp = _a[2] - _b[2]
                _pq = _a[2] - _a[3]
                print(' OptimizeStep proceeding: iter #{} finished, while '
                      'delta_pp = {:.3f}, delta_pq = {:.3f}.'.format(i, _pp, 
                      _pq))
                if (abs(_pp) < steps._eps_pp) and (abs(_pq) < steps._eps_pq):
                    break
            if i == steps._max_iter - 1:
                warnings.warn('Optimization did not converge within the max '
                              'number of iterations.', RuntimeWarning)

        else:
            if self._x_0 is None:
                x_0 = np.zeros(self.input_size)
            else:
                x_0 = self.from_original(self._x_0[0])
            logp = lambda x: self.logp(x, original_space=False)
            if self._density.has_true_jac:
                grad = lambda x: self.grad(x, original_space=False)
            else:
                grad = None
            laplace = Laplace(logp, x_0, grad=grad)
            lap_res = laplace.run(**steps._sample_options)
            
            x_max = self.to_original(lap_res.x_max)
            logp_trans = lap_res.f_max
            logp = self.to_original_density(density=logp_trans, x_trans=x_max)
            f_max = DensityQuartet(float(logp), None, float(logp_trans), None)
            samples = self.to_original(lap_res.samples)
            
            data.append(
                OptimizeResult(x_max=x_max, f_max=f_max, samples=samples,
                surrogate_list=[], hmc_samples=None, var_dicts=None, 
                Laplace=lap_res, trace=None))
        
        if steps._run_hmc:
            self._opt_hmc()
        result._i_optimize = 1
        print('\n ***** OptimizeStep finished. ***** \n')
    
    def _opt_hmc(self):
        result = self.result
        steps = self.result.steps.optimize
        data = self.result.data.optimize
        
        old_list = self._density.surrogate_list
        self._density.surrogate_list = data[-1].surrogate_list
        ho = {'sampler_options': {'metric': np.diag(data[-1].Laplace.cov)}}
        ho.update(steps.hmc_options)
        x, t = sample(
            self._density, self._client, random_state=result._random_state, 
            x_0=data[-1].samples, return_trace=True, **ho)
        data[-1] = data[-1]._replace(hmc_samples=x, trace=t)
        self._density.surrogate_list = old_list
        print('\n *** Finished sampling the density defined by the last '
              'OptimizeStep. *** \n')
    
    def _sam_step(self):
        result = self.result
        steps = self.result.steps.sample
        data = self.result.data.sample
        
        for i in range(result.i.sample, result.n.sample):
            soi = {}
            if steps[i].has_surrogate:
                self._density.surrogate_list = steps[i]._surrogate_list
                if i == 0 and steps[i]._prefit:
                    var_dicts = None
                    x_0 = result._x_0
                else:
                    if i == 0 and not result.n.optimize:
                        warnings.warn(
                            'we find neither prefit surrogate models nor '
                            'optimize steps, so we have to fit the surrogate '
                            'model directly from x_0, which is however '
                            'depreciated.', RuntimeWarning)
                        if result._x_0 is None:
                            x_0 = multivariate_normal(
                                np.zeros(self.input_size), 
                                np.eye(self.input_size), steps.n_eval)
                        else:
                            if result._x_0.shape[0] < steps.n_eval:
                                raise RuntimeError(
                                    'I need {} points to fit the surrogate '
                                    'model, but you only gave me enough {} '
                                    'points in x_0.'.format(steps.n_eval, 
                                    result._x_0.shape[0]))
                            x_0 = result._x_0[:steps.n_eval].copy()
                        var_dicts = self._map_fun(
                            self._client, self._density, x_0)
                        self._density.fit(var_dicts, **steps[i]._fit_options[0])
                        
                    else:
                        if i == 0 and result.n.optimize:
                            if result.data.optimize[-1].hmc_samples is None:
                                self._opt_hmc()
                            _x_0 = result.data.optimize[-1].hmc_samples
                            _logq = np.array([self.to_original_density(
                                *t.get(return_logp=True)[::-1]) for t in 
                                result.data.optimize[-1].trace])
                        else:
                            _x_0 = data[-1].samples
                            _logq = np.array([self.to_original_density(
                                *t.get(return_logp=True)[::-1]) for t in 
                                result.data.sample[-1].trace])
                        _x_0 = _x_0.reshape((-1, _x_0.shape[-1]))
                        _logq = _logq.reshape(-1)
                        _logq_min = np.min(_logq)
                        
                        if not steps[i].adapt_metric:
                            cov = np.cov(self.from_original(_x_0), rowvar=False)
                            soi.update({'sampler_options': 
                                        {'metric': cov, 'adapt_metric': False}})
                            #soi['sampler_options'] = {'metric': np.diag(cov)}
                        
                        if i != len(steps) - 1 and not steps[i].adapt_metric:
                            soi.update({'n_iter': 2500, 'n_warmup': 500})
                        elif i == len(steps) - 1 and not steps[i].adapt_metric:
                            soi.update({'n_iter': 4500, 'n_warmup': 500})
                        elif i == len(steps) - 1 and steps[i].adapt_metric:
                            soi.update({'n_iter': 5000, 'n_warmup': 1000})
                        
                        resample_options = steps[i].resample_options
                        """if resample_options == {}:
                            if i > 0:
                                resample_options = {'nodes': [1, 25, 100], 
                                                    'weights': [0.5, 0.5]}"""
                        i_r = resample(_logq, n=steps[i].n_eval, 
                                       **resample_options)
                        x_0 = _x_0[i_r]
                        np.delete(_x_0, i_r, axis=0)
                        np.delete(_logq, i_r, axis=0)
                        
                        var_dicts = self._map_fun(
                            self._client, self._density, x_0)
                        var_dictsall = var_dicts.copy()
                        _logp = np.concatenate(
                            [vd._fun[self._density._density_name] for vd in 
                            var_dicts])
                        _logp_all = _logp.copy()
                        if steps[i].reuse_steps:
                            for j in range(i):
                                if (j + steps[i].reuse_steps >= i or 
                                    steps[i].reuse_steps < 0):
                                    var_dictsall.extend(data[j].var_dicts)
                                    _logp_s = np.concatenate(
                                        [vd._fun[self._density._density_name] 
                                        for vd in data[j].var_dicts])
                                    _logp_all = np.concatenate(
                                        (_logp_all, _logp_s))
                        if steps[i].logp_cutoff:
                            i_p = (_logp_all > _logq_min)
                            while np.sum(i_p) < steps[i].n_eval_min:
                                if steps[i].n_eval_supp > _logq.size:
                                    raise ValueError(
                                        'we do not have enough samples that '
                                        'meet the logp_cutoff condition.')
                                i_r = resample(_logq, n=steps[i].n_eval_supp,
                                               **resample_options)
                                x_0 = _x_0[i_r]
                                np.delete(_x_0, i_r, axis=0)
                                np.delete(_logq, i_r, axis=0)
                                var_dictss = self._map_fun(
                                    self._client, self._density, x_0)
                                var_dicts.extend(var_dictss)
                                var_dictsall.extend(var_dictss)
                                _logp_s = np.concatenate(
                                    [vd._fun[self._density._density_name] for vd 
                                    in var_dictss])
                                _logp_all = np.concatenate(
                                    (_logp_all, _logp_s))
                                i_p = (_logp_all > _logq_min)
                        else:
                            i_p = np.arange(len(var_dictsall))
                        self._density.fit(np.asarray(var_dictsall)[i_p], 
                                          **steps[i]._fit_options[0])
                
                soi.update(steps[i].sample_options)
                x, t = sample(self._density, self._client, 
                              random_state=result._random_state, 
                              x_0=x_0, return_trace=True, **soi)
                surrogate_list = deepcopy(self._density._surrogate_list)
                data.append(SampleResult(
                    samples=x, surrogate_list=surrogate_list, 
                    var_dicts=var_dicts, trace=t))
            
            else:
                if i == 0:
                    if result.n.optimize:
                        if result.data.optimize[-1].hmc_samples is not None:
                            x_0 = result.data.optimize[-1].hmc_samples
                        else:
                            x_0 = result.data.optimize[-1].samples
                    else:
                        x_0 = result.x_0
                else:
                    x_0 = data[-1].samples
                self._density.surrogate_list = []
                soi.update(steps[i].sample_options)
                x, t = sample(self._density, self._client, 
                              random_state=result._random_state, 
                              x_0=x_0, return_trace=True, **soi)
                data.append(SampleResult(samples=x, surrogate_list=[], 
                                         var_dicts=None, trace=t))
            
            result._i_sample += 1
            print('\n *** SampleStep proceeding: iter #{} finished. *** '
                  '\n'.format(i))
        print(' ***** SampleStep finished. ***** \n')
    
    def _pos_step(self):
        result = self.result
        steps = self.result.steps.post
        data = self.result.data.post
        
        if result.n.sample:
            _samples = result.data.sample[-1].samples
            samples = _samples.reshape((-1, _samples.shape[-1]))
            _logq = np.array(
                [self.to_original_density(*t.get(return_logp=True)[::-1]) for t 
                in result.data.sample[-1].trace])
            logq = _logq.reshape(-1)
            if steps.n_is == 0:
                data.append(
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
                data.append(PostResult(
                    samples, weights, logp, logq, _samples, weights_raw))

        elif result.n.optimize:
            raise NotImplementedError

        else:
            raise RuntimeError(
                'the recipe has neither OptimizeStep nor SampleStep.')
        
        print(' ***** PostStep finished. ***** \n')
    
    def run(self, client=None, steps=-1):
        try:
            if client is not None:
                self._client = client
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
            return self._result.data.post[0]
        except:
            raise RuntimeError('you have not run a PostStep.')

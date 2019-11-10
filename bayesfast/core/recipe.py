from .density import Density
from ..modules.poly import PolyConfig, PolyModel
from ..utils import Laplace
from ..utils.random import resample


class RecipeStep:
    def __init__(self, surrogate_list, fit_options={}, resample_options={}):
        pass


class SurrogateRecipe:
    def __init__(self, density, step_list):
        self._set_density(density)
        self._set_step_list(step_list)
    
    @property
    def density(self):
        return self._density
    
    def _set_density(self, density):
        if isinstance(density, Density):
            self._density = density
        else:
            raise ValueError('density should be a Density.')
    
    @property
    def step_list(self):
        return tuple(self._step_list)
    
    def _set_step_list(self, step_list):
        if isinstance(step_list, RecipeStep):
            sl = [step_list]
        else:
            sl = []
            try:
                for step in step_list:
                    assert isinstance(step, RecipeStep)
                    sl.append(step)
            except:
                raise ValueError('step_list should be a RecipeStep, or a list '
                                 'of RecipeStep(s).')
        self._step_list = sl


class OptimizeRecipe(SurrogateRecipe):
    pass


class SampleRecipe(SurrogateRecipe):
    pass

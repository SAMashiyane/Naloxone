# Author: Salio(Mohtarami)
# https://github.com/SAMashiyane
# Date: 2023-2024
# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/_random.pyx

from typing import Any, Callable
from unittest.mock import patch
import numpy as np
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.utils import check_random_state
from naloxlib.inward.pipeline import pipeline_predict_inverse_only

def _mp_sample_without_replacement(
    n_population: int, n_samples: int, method=None, random_state=None
) -> Any:

    if n_population < 0:
        raise ValueError(
            "n_population should be greater than 0, got %s." % n_population
        )

    if n_samples > n_population:
        raise ValueError(
            "n_population should be greater or equal than "
            "n_samples, got n_samples > n_population (%s > %s)"
            % (n_samples, n_population)
        )

    rng = check_random_state(random_state)
    rng_randint = rng.randint

    selected = set()
    for i in range(n_samples):
        j = rng_randint(n_population, dtype=np.uint64)
        while j in selected:
            j = rng_randint(n_population, dtype=np.uint64)
        selected.add(j)
    return [int(x) for x in selected]


def _mp_ParameterGrid_getitem(self, ind):

    # This is used to make discrete sampling without replacement memory
    # efficient.
    ind = int(ind)
    for sub_grid in self.param_grid:
        # XXX: could memoize information used here
        if not sub_grid:
            if ind == 0:
                return {}
            else:
                ind -= 1
                continue


        keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
        sizes = [len(v_list) for v_list in values_lists]
        total = int(np.product(sizes, dtype=np.uint64))

        if ind >= total:
            # Try the next grid
            ind -= total
        else:
            out = {}
            for key, v_list, n in zip(keys, values_lists, sizes):
                ind, offset = divmod(int(ind), n)
                out[key] = v_list[offset]
            return out

    raise IndexError("ParameterGrid index out of range")


class MultimetricScorerPatched(_MultimetricScorer):

    def _use_cache(self, estimator):
        try:
            return super()._use_cache(estimator)
        except AttributeError:
            return True


def fit_and_score(*args, **kwargs) -> dict:


    def wrapper(*args, **kwargs) -> dict:
        with patch(
            "sklearn.model_selection._validation._MultimetricScorer",
            MultimetricScorerPatched,
        ), patch("sklearn.model_selection._validation._score", score(_score)):
            return _fit_and_score(*args, **kwargs)

    return wrapper(*args, **kwargs)


def score(f: Callable) -> Callable:


    def wrapper(*args, **kwargs):
        args = list(args)  # Convert to list for item assignment
        if len(args[0]) > 1:  # Has transformers
            args[1], y_transformed = args[0]._memory_full_transform(
                args[0], args[1], args[2], with_final=False
            )
            args[2] = args[2][args[2].index.isin(y_transformed.index)]

        with pipeline_predict_inverse_only():
            return f(args[0], *tuple(args[1:]), **kwargs)

    return wrapper

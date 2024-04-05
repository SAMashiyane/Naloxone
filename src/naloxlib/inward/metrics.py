import traceback
import warnings
from typing import Callable

import numpy as np
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics._scorer import _Scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _safe_indexing

from naloxlib.efficacy.depend import get_label_encoder

_fit_failed_message_warning = (
    "Metric '{0}' failed and error score {1} has been returned instead. "
    "If this is a custom metric, this usually means that the error is "
    "in the metric code. "
    "Full exception below:\n{2}"
)


def get_pos_label(globals_dict: dict):
    if globals_dict.get("pipeline"):
        le = get_label_encoder(globals_dict["pipeline"])
        if le:
            return le.classes_
    elif globals_dict.get("y") is not None:
        known_classes = np.unique(globals_dict["y"].values)
        return known_classes
    return None


def _get_response_method(response_method, needs_threshold, needs_proba):

    needs_threshold_provided = needs_threshold != "deprecated"
    needs_proba_provided = needs_proba != "deprecated"
    response_method_provided = response_method is not None

    needs_threshold = False if needs_threshold == "deprecated" else needs_threshold
    needs_proba = False if needs_proba == "deprecated" else needs_proba

    if response_method_provided and (needs_proba_provided or needs_threshold_provided):
        raise ValueError(
            "You cannot set both `response_method` and `needs_proba` or "
            "`needs_threshold` at the same time. Only use `response_method` since "
            "the other two are deprecated in version 1.4 and will be removed in 1.6."
        )

    if needs_proba_provided or needs_threshold_provided:
        warnings.warn(
            (
                "The `needs_threshold` and `needs_proba` parameter are deprecated in "
                "version 1.4 and will be removed in 1.6. You can either let "
                "`response_method` be `None` or set it to `predict` to preserve the "
                "same behaviour."
            ),
            FutureWarning,
        )

    if response_method_provided:
        return response_method

    if needs_proba is True and needs_threshold is True:
        raise ValueError(
            "You cannot set both `needs_proba` and `needs_threshold` at the same "
            "time. Use `response_method` instead since the other two are deprecated "
            "in version 1.4 and will be removed in 1.6."
        )

    if needs_proba is True:
        response_method = "predict_proba"
    elif needs_threshold is True:
        response_method = ("decision_function", "predict_proba")
    else:
        response_method = "predict"

    return response_method


class EncodedDecodedLabelsScoreFunc:
    """Wrapper to handle both encoded and decoded labels."""

    def __init__(self, score_func: Callable, labels: list):
        self.score_func = score_func
        self.labels = tuple(labels) if labels is not None else None
        self.__name__ = score_func.__name__

    def __call__(self, y_true, y_pred, **kwargs):
        if self.labels and _safe_indexing(y_true, 0) in self.labels:
            kwargs["labels"] = self.labels
            kwargs["pos_label"] = self.labels[-1]
        return self.score_func(y_true, y_pred, **kwargs)


class EncodedDecodedLabelsReplaceScoreFunc:
    """Wrapper to encode y_true and y_pred if necessary."""

    def __init__(self, score_func: Callable, labels: list):
        self.score_func = score_func
        self.labels = np.array(labels) if labels is not None else None
        self.__name__ = score_func.__name__

    def __call__(self, y_true, y_pred, **kwargs):
        try:
            return self.score_func(y_true, y_pred, **kwargs)
        except ValueError as e:
            if self.labels is not None and "is not a valid label" in str(e):
                encoder = LabelEncoder()
                encoder.classes_ = self.labels
                return self.score_func(
                    encoder.transform(y_true), encoder.transform(y_pred), **kwargs
                )
            else:
                raise


class BinaryMulticlassScoreFunc:
    """Wrapper to replace call kwargs with preset values if target is binary."""

    def __init__(self, score_func: Callable, kwargs_if_binary: dict):
        self.score_func = score_func
        self.kwargs_if_binary = kwargs_if_binary
        self.__name__ = score_func.__name__

    def __call__(self, y_true, y_pred, **kwargs):
        if self.kwargs_if_binary:
            labels = kwargs.get("labels", None)
            is_binary = (
                len(labels) <= 2
                if labels is not None
                else ((y_true == 0) | (y_true == 1)).all()
            )
            if is_binary:
                kwargs = {**kwargs, **self.kwargs_if_binary}
        return self.score_func(y_true, y_pred, **kwargs)


class ScorerWithErrorScore(_Scorer):
    def __init__(
        self, score_func, sign, kwargs, error_score=np.nan, response_method="predict"
    ):
        super().__init__(
            score_func=score_func,
            sign=sign,
            kwargs=kwargs,
            response_method=response_method,
        )
        self.error_score = error_score

    def _score(
        self,
        method_caller,
        estimator,
        X,
        y_true,
        sample_weight=None,
    ):


        try:
            return super()._score(
                method_caller=method_caller,
                estimator=estimator,
                X=X,
                y_true=y_true,
                sample_weight=sample_weight,
            )
        except Exception:
            warnings.warn(
                _fit_failed_message_warning.format(
                    repr(self), self.error_score, traceback.format_exc()
                ),
                FitFailedWarning,
            )
            return self.error_score

    def _factory_args(self):
        return (
            f", response_method={self._response_method}, error_score={self.error_score}"
        )


def make_scorer_with_error_score(
    score_func,
    *,
    response_method=None,
    greater_is_better=True,
    needs_proba="deprecated",
    needs_threshold="deprecated",
    error_score=np.nan,
    **kwargs,
):


    response_method = _get_response_method(
        response_method, needs_threshold, needs_proba
    )

    sign = 1 if greater_is_better else -1

    # Create an instance of ScorerWithErrorScore
    scorer = ScorerWithErrorScore(
        score_func, sign, kwargs, error_score, response_method
    )

    return scorer

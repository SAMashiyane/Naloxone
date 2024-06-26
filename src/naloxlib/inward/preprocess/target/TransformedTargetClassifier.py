
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import (
    _deprecate_positional_args,
    check_is_fitted,
    check_X_y,
)

from .utils import TargetTransformerMixin


class TransformedTargetClassifier(
    TargetTransformerMixin, ClassifierMixin, BaseEstimator
):


    @_deprecate_positional_args
    def __init__(self, classifier=None, *, transformer=None, check_inverse=True):
        self.classifier = classifier
        self.transformer = transformer
        self.check_inverse = check_inverse

    def _fit_transformer(self, y):

        if self.transformer is not None:
            self.transformer_ = clone(self.transformer)
        else:  # Just return the identity transformer
            self.transformer_ = LabelEncoder()

        self.transformer_.fit(y)
        if self.check_inverse:
            idx_selected = slice(None, None, max(1, y.shape[0] // 10))
            y_sel = _safe_indexing(y, idx_selected)
            y_sel_t = self.transformer_.transform(y_sel)

            if (np.ravel(y_sel) != self.transformer_.inverse_transform(y_sel_t)).any():
                warnings.warn(
                    "The provided functions or transformer are"
                    
                    ", set 'check_inverse=False'",
                    UserWarning,
                )

    def fit(self, X, y, **fit_params):


        X, y = check_X_y(X, y)

        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values




        self.classes_ = np.unique(y)

        self._training_dim = y.ndim

        if y.ndim == 1:
            y_2d = y.reshape(-1, 1)
        else:
            y_2d = y
        self._fit_transformer(y_2d)


        y_trans = self.transformer_.transform(y_2d)

        if y_trans.ndim == 2 and y_trans.shape[1] == 1:
            y_trans = y_trans.squeeze(axis=1)

        if self.classifier is None:
            from sklearn.linear_model import LogisticRegression

            self.classifier_ = LogisticRegression()
        else:
            self.classifier_ = clone(self.classifier)

        self.classifier_.fit(X, y_trans, **fit_params)

        self._carry_over_estimator_fit_vars(
            self.classifier_, ignore=["classes_", "transformer_", "classifier_"]
        )

        return self

    def predict(self, X):

        check_is_fitted(self)
        pred = self.classifier_.predict(X)
        if pred.ndim == 1:
            pred_trans = self.transformer_.inverse_transform(pred.reshape(-1, 1))
        else:
            pred_trans = self.transformer_.inverse_transform(pred)
        if (
            self._training_dim == 1
            and pred_trans.ndim == 2
            and pred_trans.shape[1] == 1
        ):
            pred_trans = pred_trans.squeeze(axis=1)

        return pred_trans

    def _more_tags(self):
        return {"poor_score": True, "no_validation": True}

    @property
    def n_features_in_(self):

        try:
            check_is_fitted(self)
        except NotFittedError as nfe:
            raise AttributeError(
                "{} object has no n_features_in_ attribute.".format(
                    self.__class__.__name__
                )
            ) from nfe

        return self.classifier_.n_features_in_

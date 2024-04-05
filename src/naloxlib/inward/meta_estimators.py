import sklearn.compose
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import PowerTransformer
from sklearn.utils.validation import check_is_fitted


class PowerTransformedTargetRegressor(sklearn.compose.TransformedTargetRegressor):
    def __init__(
        self,
        regressor=None,
        *,
        power_transformer_method="box-cox",
        power_transformer_standardize=True,
        **kwargs,
    ):
        self.regressor = regressor
        self.power_transformer_method = power_transformer_method
        self.power_transformer_standardize = power_transformer_standardize
        self.transformer = PowerTransformer(
            method=self.power_transformer_method,
            standardize=self.power_transformer_standardize,
        )
        self.func = None
        self.inverse_func = None
        self.check_inverse = False
        self._fit_vars = set()
        self.set_params(**kwargs)

    def __getattr__(self, name: str):
        # override getattr to allow grabbing of regressor attrs
        if name not in ("regressor", "regressor_"):
            if hasattr(self, "regressor_"):
                return getattr(self.regressor_, name)
            return getattr(self.regressor, name)

    def fit(self, X, y, **fit_params):



        y = y.astype("float64")

        super().fit(X, y, **fit_params)
        return self

    def set_params(self, **params):

        if "regressor" in params:
            self.regressor = params.pop("regressor")
        if "power_transformer_method" in params:
            self.power_transformer_method = params.pop("power_transformer_method")
            self.transformer.set_params(**{"method": self.power_transformer_method})
        if "power_transformer_standardize" in params:
            self.power_transformer_standardize = params.pop(
                "power_transformer_standardize"
            )
            self.transformer.set_params(
                **{"standardize": self.power_transformer_standardize}
            )
        return self.regressor.set_params(**params)

    def get_params(self, deep=True):

        r = self.regressor.get_params(deep=deep)
        r["power_transformer_method"] = self.power_transformer_method
        r["power_transformer_standardize"] = self.power_transformer_standardize
        r["regressor"] = self.regressor
        return r


class CustomProbabilityThresholdClassifier(ClassifierMixin, BaseEstimator):


    def __init__(
        self,
        classifier=None,
        *,
        probability_threshold=0.5,
        **kwargs,
    ):
        self.classifier = classifier
        self.probability_threshold = probability_threshold
        self.set_params(**kwargs)

    def fit(self, X, y, **fit_params):
        if (
            not isinstance(self.probability_threshold, (int, float))
            or self.probability_threshold > 1
            or self.probability_threshold < 0
        ):
            raise TypeError(
                "probability_threshold parameter only accepts value between 0 to 1."
            )

        if self.classifier is None:
            from sklearn.linear_model import LogisticRegression

            self.classifier_ = LogisticRegression()
        else:
            self.classifier_ = clone(self.classifier)

        self.classifier_.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        check_is_fitted(self)
        if not hasattr(self.classifier_, "predict_proba"):
            return self.classifier_.predict(X, **predict_params)
        pred = self.classifier_.predict_proba(X, **predict_params)
        if pred.shape[1] > 2:
            raise ValueError(
                f"{self.__class__.__name__} can only be used for binary classification."
            )
        return (pred[:, 1] >= self.probability_threshold).astype("int")

    def __getattr__(self, name: str):
        # override getattr to allow grabbing of regressor attrs
        if name not in ("classifier", "classifier_"):
            if hasattr(self, "classifier_"):
                return getattr(self.classifier_, name)
            return getattr(self.classifier, name)

    def set_params(self, **params):

        if "classifier" in params:
            self.classifier = params.pop("classifier")
        if "probability_threshold" in params:
            self.probability_threshold = params.pop("probability_threshold")
        return self.classifier.set_params(**params)

    def get_params(self, deep=True):

        r = self.classifier.get_params(deep=deep)
        r["classifier"] = self.classifier
        r["probability_threshold"] = self.probability_threshold
        return r


def get_estimator_from_meta_estimator(estimator):

    
    if not isinstance(
        estimator, (TransformedTargetRegressor, CustomProbabilityThresholdClassifier)
    ):
        return estimator


    if hasattr(estimator, "classifier_"):
        return get_estimator_from_meta_estimator(estimator.classifier_)

    if hasattr(estimator, "classifier"):
        return get_estimator_from_meta_estimator(estimator.classifier)
    return estimator

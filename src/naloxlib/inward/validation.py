
from sklearn.utils.validation import check_is_fitted

def is_sklearn_pipeline(object):
    from sklearn.pipeline import Pipeline

    return isinstance(object, Pipeline)

def is_sklearn_cv_generator(object):
    return not isinstance(object, str) and hasattr(object, "split")

def is_fitted(estimator) -> bool:
    try:
        check_is_fitted(estimator)
        return True
    except Exception:
        return False




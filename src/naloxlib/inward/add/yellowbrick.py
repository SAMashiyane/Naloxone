# Author: Salio(Mohtarami)
# https://github.com/SAMashiyane
# Date: 2023-2024

from yellowbrick.utils.helpers import get_model_name as get_model_name_original
from naloxlib.inward.meta_estimators import get_estimator_from_meta_estimator
def is_estimator(model):
    try:
        return callable(getattr(model, "fit"))
    except Exception:
        return False
def get_model_name(model):
    return get_model_name_original(get_estimator_from_meta_estimator(model))

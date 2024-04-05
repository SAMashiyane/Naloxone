# Author: Salio(Mohtarami)
# https://github.com/SAMashiyane
# Date: 2023-2024


import numpy as np
import pandas as pd
from scipy import sparse
import functools
import inspect
import traceback
import pandas.io.formats.style
import warnings
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, List, Set, Tuple
from sklearn.metrics import get_scorer
from sklearn.metrics._scorer import _Scorer
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold
from sklearn.model_selection._split import _BaseKFold

from typing import Dict, Optional
from typing import Union
from naloxlib.inward.logging import get_logger
import naloxlib.box
from naloxlib.inward.validation import (
    is_sklearn_cv_generator,
    is_sklearn_pipeline,

)


if TYPE_CHECKING:
    from naloxlib.inward.naloxlib_experiment.naloxlib_experiment import (
        _naloxlibExperiment,
    )

logger = get_logger()
#---------------------------------start depend by Salio--------------------------------

def _check_soft_dependencies(
    package: str,
    severity: str = "error",
    extra: Optional[str] = "all_extras",
    install_name: Optional[str] = None,
) -> bool:
    """
    """

SEQUENCE = (list, tuple, np.ndarray, pd.Series)
SEQUENCE_LIKE = Union[SEQUENCE]
DATAFRAME_LIKE = Union[dict, list, tuple, np.ndarray, sparse.spmatrix, pd.DataFrame]
TARGET_LIKE = Union[int, str, list, tuple, np.ndarray, pd.Series]
LABEL_COLUMN = "prediction_label"
SCORE_COLUMN = "prediction_score"


class MLUsecase(Enum):
    CLASSIFICATION = auto()
    TIME_SERIES = auto()

def get_ml_task(y):
    c1 = y.dtype == "int64"
    c2 = y.nunique() <= 20
    c3 = y.dtype.name in ["object", "bool", "category"]
    if (c1 & c2) | c3:
        ml_usecase = MLUsecase.CLASSIFICATION
    else:
        ml_usecase = None
    return ml_usecase


def get_classification_task(y):
    return "Binary" if y.nunique() == 2 else "Multiclass"

def to_df(data, index=None, columns=None, dtypes=None):
    n_cols = lambda data: data.shape[1] if hasattr(data, "shape") else len(data[0])

    if data is not None:
        if not isinstance(data, pd.DataFrame):
            # Assign default column names (dict already has column names)
            if not isinstance(data, dict) and columns is None:
                columns = [f"feature_{str(i)}" for i in range(1, n_cols(data) + 1)]

            # Create dataframe from sparse matrix or directly from data
            if sparse.issparse(data):
                data = pd.DataFrame.sparse.from_spmatrix(data, index, columns)
            else:
                data = pd.DataFrame(data, index, columns)

            if dtypes is not None:
                data = data.astype(dtypes)

        # Convert all column names to str
        data = data.rename(columns=lambda col: str(col))

    return data


def to_series(data, index=None, name=None):

    name = name or "target"
    if data is not None and not isinstance(data, pd.Series):
        if isinstance(data, pd.DataFrame):
            try:
                data = data[name]
            except Exception:
                data = data.squeeze()
        elif isinstance(data, np.ndarray) and data.ndim > 1:
            data = data.flatten()
        data = pd.Series(data, index=index, name=name)

    return data


def check_features_exist(features: List[str], X: pd.DataFrame):
    
    missing_features = []
    for fx in features:
        if fx not in X.columns:
            missing_features.append(fx)

    if len(missing_features) != 0:
        raise ValueError(
            f"\n\nColumn(s): {missing_features} not found in the feature dataset by salio"
            "\nThey are either missing from the features or you have specified "
            "a target column as a feature. Available feature columns are :"
            f"\n{X.columns.to_list()}"
        )


def id_or_display_name(metric, input_ml_usecase, target_ml_usecase):
    if input_ml_usecase == target_ml_usecase:
        output = metric.id
    else:
        output = metric.display_name

    return output


def variable_return(X, y):
    
    if y is None:
        return X
    elif X is None:
        return y
    else:
        return X, y


def get_config(variable: str, globals_d: dict):


    function_params_str = ", ".join(
        [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
    )

    logger = get_logger()

    logger.info("Initializing get_config() by Mohtarami:)")
    logger.info(f"get_config({function_params_str})")

    if variable not in globals_d["naloxlib_globals"]:
        raise ValueError(
            f"Variable {variable} not found. Possible variables are: {globals_d['naloxlib_globals']}"
        )

    global_var = globals_d[variable]

    logger.info(f"Global variable: {variable} returned as {global_var}")
    logger.info(
        "get_config() successfully completed ...........Mohtarami :) .............."
    )

    return global_var


def set_config(variable: str, value, globals_d: dict):
    function_params_str = ", ".join(
        [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
    )

    logger = get_logger()

    logger.info("Initializing set_config() by Mohtarami")
    logger.info(f"set_config({function_params_str})")

    if variable.startswith("_"):
        raise ValueError(f"Variable {variable} is read only ('_' prefix).")

    if variable not in globals_d["naloxlib_globals"] or variable == "naloxlib_globals":
        raise ValueError(
            f"Variable {variable} not found. Possible variables are: {globals_d['naloxlib_globals']}"
        )

    globals_d[variable] = value


    if not globals_d["gpu_param"] and variable == "n_jobs_param":
        globals_d["gpu_n_jobs_param"] = value

    logger.info(f"Global variable: {variable} updated to {value}")
    logger.info(
        "set_config() successfully completed........Mohtarami :) .................."
    )


def save_config(file_name: str, globals_d: dict):

    function_params_str = ", ".join(
        [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
    )

    logger = get_logger()

    logger.info("Initializing save_config() by Mohtarami")
    logger.info(f"save_config({function_params_str})")

    globals_to_ignore = {
        "_all_models",
        "_all_models_internal",
        "_all_metrics",
        "_master_model_container",
        "_display_container",
    }

    globals_to_dump = {
        k: v
        for k, v in globals_d.items()
        if k in globals_d["naloxlib_globals"] and k not in globals_to_ignore
    }

    import joblib

    joblib.dump(globals_to_dump, file_name)

    logger.info(f"Global variables dumped to {file_name}")
    logger.info(
        "save_config() successfully completed..........Mohtarami ;) ..................."
    )


def load_config(file_name: str, globals_d: dict):

    function_params_str = ", ".join(
        [f"{k}={v}" for k, v in locals().items() if not k == "globals_d"]
    )

    logger = get_logger()

    logger.info("Initializing load_config()")
    logger.info(f"load_config({function_params_str})")

    import joblib

    loaded_globals = joblib.load(file_name)

    logger.info(f"Global variables loaded from {file_name}")

    for k, v in loaded_globals.items():
        globals_d[k] = v

    globals_d["logger"] = get_logger()

    logger.info(f"Global variables set to match those in {file_name}")

    logger.info(
        "load_config() successfully completed.....Mohtarami:)) ........................."
    )


def color_df(
    df: pd.DataFrame, color: str, names: list, axis: int = 1
) -> pandas.io.formats.style.Styler:
    return df.style.apply(
        lambda x: [f"background: {color}" if (x.name in names) else "" for _ in x],
        axis=axis,
    )


def get_model_id(
    e, all_models: Dict[str, "naloxlib.box.models.ModelContainer"]
) -> str:
    from naloxlib.inward.meta_estimators import get_estimator_from_meta_estimator

    return next(
        (
            k
            for k, v in all_models.items()
            if v.is_estimator_equal(get_estimator_from_meta_estimator(e))
        ),
        None,
    )


def get_model_name(
    e,
    all_models: Dict[str, "naloxlib.box.models.ModelContainer"],
    deep: bool = True,
) -> str:
    all_models = all_models or {}
    old_e = e
    if isinstance(e, str) and e in all_models:
        model_id = e
    else:
        if deep:
            while hasattr(e, "get_params"):
                old_e = e
                params = e.get_params()
                if "steps" in params:
                    e = params["steps"][-1][1]
                elif "base_estimator" in params:
                    e = params["base_estimator"]
                elif "estimator" in params:
                    e = params["estimator"]
                else:
                    break
        if e is None or isinstance(e, str):
            e = old_e
        model_id = get_model_id(e, all_models)

    if model_id is not None:
        name = all_models[model_id].name
    else:
        try:
            name = type(e).__name__
        except Exception:
            name = str(e).split("(")[0]

    return name


def is_special_model(
    e, all_models: Dict[str, "naloxlib.box.models.ModelContainer"]
) -> bool:
    try:
        return all_models[get_model_id(e, all_models)].is_special
    except Exception:
        return False


def get_class_name(class_var: Any) -> str:
    return str(class_var)[8:-2]


def get_package_name(class_var: Any) -> str:
    if not isinstance(str, class_var):
        class_var = get_class_name(class_var)
    return class_var.split(".")[0]


def param_grid_to_lists(param_grid: dict) -> dict:
    if param_grid:
        for k, v in param_grid.items():
            if not isinstance(v, np.ndarray):
                v = list(v)
            param_grid[k] = v
    return param_grid


def np_list_arange(
    start: float, stop: float, step: float, inclusive: bool = False
) -> List[float]:

    convert_to_float = (
        isinstance(start, float) or isinstance(stop, float) or isinstance(step, float)
    )
    if convert_to_float:
        stop = float(stop)
        start = float(start)
        step = float(step)
    stop = stop + (step if inclusive else 0)
    range_ = list(np.arange(start, stop, step))
    range_ = [
        start
        if x < start
        else stop
        if x > stop
        else float(round(x, 15))
        if isinstance(x, float)
        else x
        for x in range_
    ]
    range_[0] = start
    range_[-1] = stop - step
    return range_


def get_function_params(function: Callable) -> Set[str]:
    return inspect.signature(function).parameters


def calculate_metrics(
    metrics: Dict[str, "naloxlib.box.metrics.MetricContainer"],
    y_test,
    pred,
    pred_proba: Optional[float] = None,
    score_dict: Optional[Dict[str, np.array]] = None,
    weights: Optional[list] = None,
    **additional_kwargs,
) -> Dict[str, np.array]:
    score_dict = []

    for k, v in metrics.items():
        score_dict.append(
            _calculate_metric(
                v,
                v.score_func,
                v.display_name,
                y_test,
                pred,
                pred_proba,
                weights,
                **additional_kwargs,
            )
        )

    score_dict = dict([x for x in score_dict if x is not None])
    return score_dict


def _calculate_metric(
    container, score_func, display_name, y_test, pred_, pred_proba, weights, **kwargs
):
    if not score_func:
        return None
    kwargs = {
        **{k: v for k, v in kwargs.items() if k in get_function_params(score_func)},
        **container.args,
    }
    target = pred_proba if container.target == "pred_proba" else pred_
    try:
        calculated_metric = score_func(y_test, target, sample_weight=weights, **kwargs)
    except Exception:
        try:
            calculated_metric = score_func(y_test, target, **kwargs)
        except Exception:
            warnings.warn(traceback.format_exc())
            calculated_metric = 0

    return display_name, calculated_metric


def normalize_custom_transformers(
    transformers: Union[Any, Tuple[str, Any], List[Any], List[Tuple[str, Any]]]
) -> list:
    if isinstance(transformers, dict):
        transformers = list(transformers.items())
    if isinstance(transformers, list):
        for i, x in enumerate(transformers):
            _check_custom_transformer(x)
            if not isinstance(x, tuple):
                transformers[i] = (f"custom_step_{i}", x)
    else:
        _check_custom_transformer(transformers)
        if not isinstance(transformers, tuple):
            transformers = ("custom_step", transformers)
        if is_sklearn_pipeline(transformers[0]):
            return transformers.steps
        transformers = [transformers]
    return transformers


def _check_custom_transformer(transformer):
    actual_transformer = transformer
    if isinstance(transformer, tuple):
        if len(transformer) != 2:
            raise ValueError("Transformer tuple must have a size of 2.")
        if not isinstance(transformer[0], str):
            raise TypeError("First element of transformer tuple must be a str.")
        actual_transformer = transformer[1]
    if not (
        (
            hasattr(actual_transformer, "fit")
            and hasattr(actual_transformer, "transform")
            and hasattr(actual_transformer, "fit_transform")
        )
        or (
            hasattr(actual_transformer, "fit")
            and hasattr(actual_transformer, "fit_resample")
        )
    ):
        raise TypeError(
            "Transformer must be an object implementing methods 'fit', 'transform' and 'fit_transform'/'fit_resample'."
        )


def get_cv_splitter(
    fold: Optional[Union[int, BaseCrossValidator]],
    default: BaseCrossValidator,
    seed: int,
    shuffle: bool,
    int_default: str = "kfold",
) -> BaseCrossValidator:
    
    if not fold:
        return default
    if is_sklearn_cv_generator(fold):
        return fold
    if type(fold) is int:
        if default is not None:
            if isinstance(default, _BaseKFold) and fold <= 1:
                raise ValueError(
                    "k-fold cross-validation requires at least one"
                    " train/test split by setting n_splits=2 or more,"
                    f" got n_splits={fold}."
                )
            try:
                default_copy = deepcopy(default)
                default_copy.n_splits = fold
                return default_copy
            except Exception:
                raise ValueError(f"Couldn't set 'n_splits' to {fold} for {default}.")
        else:
            fold_seed = seed if shuffle else None
            if int_default == "kfold":
                return KFold(fold, random_state=fold_seed, shuffle=shuffle)
            elif int_default == "stratifiedkfold":
                return StratifiedKFold(fold, random_state=fold_seed, shuffle=shuffle)
            else:
                raise ValueError(
                    "Wrong value for int_default param. Needs to be either 'kfold' or 'stratifiedkfold'."
                )
    raise TypeError(
        f"{fold} is of type {type(fold)} while it needs to be either a CV generator or int."
    )


def get_cv_n_folds(
    fold: Optional[Union[int, BaseCrossValidator]], default, X, y=None, groups=None
) -> int:
    """
    """
    if not fold:
        fold = default
    if isinstance(fold, int):
        return fold
    else:
        return fold.get_n_splits(X, y=y, groups=groups)


class set_n_jobs(object):

    def __init__(self, model, n_jobs=None):
        self.params = {}
        self.model = model
        self.n_jobs = n_jobs
        try:
            self.params = {
                k: v
                for k, v in self.model.get_params().items()
                if k.endswith("n_jobs") or k.endswith("thread_count")
            }
        except Exception:
            pass

    def __enter__(self):
        if self.params:
            self.model.set_params(**{k: self.n_jobs for k, v in self.params.items()})

    def __exit__(self, type, value, traceback):
        if self.params:
            self.model.set_params(**self.params)


class true_warm_start(object):


    def __init__(self, model):
        self.params = {}
        self.model = model
        try:
            self.params = {
                k: v
                for k, v in self.model.get_params().items()
                if k.endswith("warm_start")
            }
        except Exception:
            pass

    def __enter__(self):
        if self.params:
            self.model.set_params(**{k: True for k, v in self.params.items()})

    def __exit__(self, type, value, traceback):
        if self.params:
            self.model.set_params(**self.params)


class nullcontext(object):
    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


def get_groups(
    groups: Union[str, pd.DataFrame],
    X_train: pd.DataFrame,
    default: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    if groups is None:
        if default is None:
            return default
        else:

            return default.loc[X_train.index]
    elif isinstance(groups, str):
        if groups not in X_train.columns:
            raise ValueError(
                f"Column {groups} used for groups is not present in the dataset."
            )
        groups = X_train[groups]
    else:
        groups = groups.loc[X_train.index]
        if groups.shape[0] != X_train.shape[0]:
            raise ValueError(
                f"groups has length {groups.shape[0]} which doesn't match X_train "
                f"length of {len(X_train)}."
            )

    return groups


def get_all_object_vars_and_properties(object):

    d = {}
    for k in object.__dir__():
        try:
            if k[:2] != "__" and type(getattr(object, k, "")).__name__ != "method":
                d[k] = getattr(object, k, "")
        except Exception:
            pass
    return d


def is_fit_var(key):
    return key and (
        (key.endswith("_") and not key.startswith("_")) or (key in ["n_clusters"])
    )


def can_early_stop(
    estimator,
    consider_partial_fit,
    consider_warm_start,
    consider_xgboost,
    params,
):


    logger = get_logger()

    from sklearn.ensemble import BaseEnsemble
    from sklearn.tree import BaseDecisionTree

    try:
        base_estimator = estimator.steps[-1][1]
    except Exception:
        base_estimator = estimator

    if consider_partial_fit:
        pass
        # can_partial_fit = supports_partial_fit(base_estimator, params=params)
    else:
        can_partial_fit = False

    if consider_warm_start:
        is_not_tree_subclass = not issubclass(type(base_estimator), BaseDecisionTree)
        is_ensemble_subclass = issubclass(type(base_estimator), BaseEnsemble)
        can_warm_start = hasattr(base_estimator, "warm_start") and (
            (
                hasattr(base_estimator, "max_iter")
                and is_not_tree_subclass
                and not is_ensemble_subclass
            )
            or (is_ensemble_subclass and hasattr(base_estimator, "n_estimators"))
        )
    else:
        can_warm_start = False

    is_xgboost = False

    if _check_soft_dependencies("xgboost", extra="models", severity="warning"):
        if consider_xgboost:
            from xgboost.sklearn import XGBModel

            is_xgboost = isinstance(base_estimator, XGBModel)

    logger.info(
        f"can_partial_fit: {can_partial_fit}, can_warm_start: {can_warm_start}, is_xgboost: {is_xgboost}"
    )

    return can_partial_fit or can_warm_start or is_xgboost


def infer_ml_usecase(y: pd.Series) -> Tuple[str, str]:
    c1 = "int" in y.dtype.name
    c2 = y.nunique() <= 20
    c3 = y.dtype.name in ["object", "bool", "category"]

    if (c1 and c2) or c3:
        ml_usecase = "classification"
    else:
        pass
    if y.nunique() > 2:
        subcase = "multi"
    else:
        subcase = "binary"
    return ml_usecase, subcase


def get_columns_to_stratify_by(
    X: pd.DataFrame, y: pd.DataFrame, stratify: Union[bool, List[str]]
) -> pd.DataFrame:
    if not stratify:
        stratify = None
    else:
        if isinstance(stratify, list):
            data = pd.concat([X, y], axis=1)
            if not all(col in data.columns for col in stratify):
                raise ValueError("Column to stratify by does not exist in the dataset.")
            stratify = data[stratify]
        else:
            stratify = y
    return stratify


def check_if_global_is_not_none(globals_d: dict, global_names: dict):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for name, message in global_names.items():
                if globals_d[name] is None:
                    raise ValueError(message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def df_shrink_dtypes(df, skip=[], obj2cat=True, int2uint=False):

    excl_types, skip = {"category", "datetime64[ns]", "bool"}, set(skip)

    typemap = {
        "int": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.int8, np.int32, np.int64)
        ],
        "uint": [
            (np.dtype(x), np.iinfo(x).min, np.iinfo(x).max)
            for x in (np.uint8, np.uint32, np.uint64)
        ],
        "float": [
            (np.dtype(x), np.finfo(x).min, np.finfo(x).max)
            for x in (np.float32, np.float64, np.longdouble)
        ],
    }

    if obj2cat:

        typemap["object"] = "category"
    else:
        excl_types.add("object")

    new_dtypes = {}
    exclude = lambda dt: dt[1].name not in excl_types and dt[0] not in skip

    for c, old_t in filter(exclude, df.dtypes.items()):
        t = next((v for k, v in typemap.items() if old_t.name.startswith(k)), None)

        if isinstance(t, list):  # Find the smallest type that fits
            if int2uint and t == typemap["int"] and df[c].min() >= 0:
                t = typemap["uint"]
            new_t = next(
                (r[0] for r in t if r[1] <= df[c].min() and r[2] >= df[c].max()), None
            )
            if new_t and new_t == old_t:
                new_t = None
        else:
            new_t = t if isinstance(t, str) else None

        if new_t:
            new_dtypes[c] = new_t

    return df.astype(new_dtypes)


def get_label_encoder(pipeline):

    try:
        encoder = next(
            step[1] for step in pipeline.steps if step[0] == "label_encoding"
        )
        return encoder.transformer
    except StopIteration:
        return





def deep_clone(estimator: Any) -> Any:

    estimator_ = deepcopy(estimator)
    return estimator_


def check_metric(
    actual: pd.Series,
    prediction: pd.Series,
    metric: str,
    round: int = 4,
    train: Optional[pd.Series] = None,
):

    from naloxlib.box.metrics.base_metric import (
        get_all_metric_containers as get_all_class_metric_containers,
    )



    globals_dict = {"y": prediction}
    metric_containers = {
        **get_all_class_metric_containers(globals_dict),
    }
    metrics = {
        v.name: functools.partial(v.score_func, **(v.args or {}))
        for k, v in metric_containers.items()
    }

    if isinstance(train, pd.Series):
        input_params = [actual, prediction, train]
    else:
        input_params = [actual, prediction]

    # metric calculation starts here

    if metric in metrics:
        try:
            result = metrics[metric](*input_params)
        except Exception:
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            actual = le.fit_transform(actual)
            prediction = le.transform(prediction)
            result = metrics[metric](actual, prediction)
        result = np.around(result, round)
        return float(result)
    else:
        raise ValueError(
            f"Couldn't find metric '{metric}' Possible metrics are: {', '.join(metrics.keys())}."
        )


def _get_metrics_dict(
    metrics_dict: Dict[str, Union[str, _Scorer]]
) -> Dict[str, _Scorer]:
    """
    """
    return_metrics_dict = {}
    for k, v in metrics_dict.items():
        if isinstance(v, str):
            return_metrics_dict[k] = get_scorer(v)
        else:
            return_metrics_dict[k] = v
    return return_metrics_dict





def get_system_logs():

    with open("logs.log", "r") as file:
        lines = file.read().splitlines()

    for line in lines:
        if not line:
            continue

        columns = [col.strip() for col in line.split(":") if col]
        print(columns)


def _coerce_empty_dataframe_to_none(
    data: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:

    if isinstance(data, pd.DataFrame) and data.empty:
        return None
    else:
        return data


def _resolve_dict_keys(
    dict_: Dict[str, Any], key: str, defaults: Dict[str, Any]
) -> Any:

    if key not in defaults:
        raise KeyError(f"Key '{key}' not present in Defaults dictionary.")
    return dict_.get(key, defaults[key])


def get_allowed_engines(
    estimator: str, all_allowed_engines: Dict[str, List[str]]
) -> Optional[List[str]]:

    allowed_engines = all_allowed_engines.get(estimator, None)
    return allowed_engines


class LazyExperimentMapping(Mapping):

    def __init__(self, experiment: "_naloxlibExperiment"):
        self._experiment = experiment
        self._keys = self._experiment._variable_keys.union(
            self._experiment._property_keys
        )
        if "variables" in self._keys:
            self._keys.remove("variables")
        self._cache = {}

    def __getitem__(self, key):
        if key in self._cache:
            return self._cache[key]
        if key in self._keys:
            item = getattr(self._experiment, key, None)
            self._cache[key] = item
            return item
        else:
            raise KeyError(key)

    def __len__(self):
        return len(self._keys)

    def __iter__(self):
        return iter(self._keys)

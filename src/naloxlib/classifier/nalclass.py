# Author: Salio(Mohtarami)
# https://github.com/SAMashiyane
# Date: 2023-2024

import re
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from naloxlib.efficacy.depend import check_if_global_is_not_none
from naloxlib.box.metrics.base_metric import get_all_metric_containers
from naloxlib.box.models.base_model import (
    ALL_ALLOWED_ENGINES,
    get_all_model_containers,
    get_container_default_engines,
)
from naloxlib.inward.display import CommonDisplay
from naloxlib.inward.pipeline import Pipeline as InternalPipeline
from naloxlib.inward.preprocess.preprocessor import Preprocessor
from naloxlib.inward.naloxlib_experiment.non_ts_supervised_experiment import (
    _NonTSSupervisedExperiment,
)

from naloxlib.efficacy.depend import DATAFRAME_LIKE, SEQUENCE_LIKE, TARGET_LIKE
from naloxlib.efficacy.depend import (
    MLUsecase,
    get_label_encoder,
)

class ClassificationExperiment(_NonTSSupervisedExperiment, Preprocessor):
    _create_app_predict_kwargs = {"raw_score": True}

    def __init__(self) -> None:
        super().__init__()
        self._ml_usecase = MLUsecase.CLASSIFICATION
        self.exp_name_log = "clf-default-name"
        self._variable_keys = self._variable_keys.union(
            {"fix_imbalance", "is_multiclass"}
        )
        self._available_plots = {
            "pipeline": "Pipeline Plot",
            "parameter": "Hyperparameters",
            "auc": "AUC",
            "confusion_matrix": "Confusion Matrix",
            "threshold": "Threshold",
            "pr": "Precision Recall",
            "error": "Prediction Error",
            "class_report": "Class Report",
            "rfe": "Feature Selection",
            "learning": "Learning Curve",
            "manifold": "Manifold Learning",
            "calibration": "Calibration Curve",
            "vc": "Validation Curve",
            "dimension": "Dimensions",
            "feature": "Feature Importance",
            "feature_all": "Feature Importance (All)",
            "boundary": "Decision Boundary",
            "lift": "Lift Chart",
            "gain": "Gain Chart",
            "tree": "Decision Tree",
            "ks": "KS Statistic Plot",
        }

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        all_models = {
            k: v
            for k, v in get_all_model_containers(
                self, raise_errors=raise_errors
            ).items()
            if not v.is_special
        }
        all_models_internal = get_all_model_containers(self, raise_errors=raise_errors)
        return all_models, all_models_internal

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return get_all_metric_containers(self.variables, raise_errors=raise_errors)


    def build_naloxone_model(
        self,
        data: Optional[DATAFRAME_LIKE] = None,
        target: TARGET_LIKE = -3,
        index: Union[bool, int, str, SEQUENCE_LIKE] = True,
        train_size: float = 0.7,
        test_data: Optional[DATAFRAME_LIKE] = None,
        preprocess: bool = True,
        data_split_shuffle: bool = True,
        data_split_stratify: Union[bool, List[str]] = True,
        fold_strategy: Union[str, Any] = "stratifiedkfold",
        fold: int = 10,
        fold_shuffle: bool = False,
        fold_groups: Optional[Union[str, pd.DataFrame]] = None,
        session_id: Optional[int] = 123,
        engine: Optional[Dict[str, str]] = None,
        verbose: bool = True,

    ):

        self.all_allowed_engines = ALL_ALLOWED_ENGINES
        self._initialize_setup(
            session_id=session_id,
            verbose=verbose,
        )

        self.data = self._prepare_dataset(data, target)
        self.target_param = self.data.columns[-1]
        self.index = index
        self._prepare_folds(
            fold_strategy=fold_strategy,
            fold=fold,
            fold_shuffle=fold_shuffle,
            fold_groups=fold_groups,
            data_split_shuffle=data_split_shuffle,
        )

        self._prepare_train_test(
            train_size=train_size,
            test_data=test_data,
            data_split_stratify=data_split_stratify,
            data_split_shuffle=data_split_shuffle,
        )


        self._set_exp_model_engines(
            container_default_engines=get_container_default_engines(),
            engine=engine,
        )

        self.pipeline = InternalPipeline(
            steps=[("placeholder", None)],
        )

        if preprocess:

            y_unique = self.y.unique()
            if sorted(list(y_unique)) != list(range(len(y_unique))):
                self._encode_target_column()

        if any(re.search("[^A-Za-z0-9_]", col) for col in self.dataset):
            self._clean_column_names()

        # Remove placeholder step
        if ("placeholder", None) in self.pipeline.steps and len(self.pipeline) > 1:
            self.pipeline.steps.remove(("placeholder", None))

        self.pipeline.fit(self.X_train, self.y_train)
        container = []
        container.append(["Outcome", self.target_param])

        le = get_label_encoder(self.pipeline)
        if le:
            mapping = {str(v): i for i, v in enumerate(le.classes_)}
            container.append(
                ["Encoding", ", ".join([f"{k}: {v}" for k, v in mapping.items()])]
            )
        container.append(["Dataset shape", self.data.shape])
        container.append([" Train_set shape", self.train_transformed.shape])
        container.append([" Test_set shape", self.test_transformed.shape])
        for fx, cols in self._fxs.items():
            if len(cols) > 0:
                container.append([f"{fx} features", len(cols)])
        if self.data.isna().sum().sum():
            n_nans = 100 * self.data.isna().any(axis=1).sum() / len(self.data)
            container.append(["Rows with missing values", f"{round(n_nans, 1)}%"])

        container.append(["Fold Generator", self.fold_generator.__class__.__name__])
        container.append(["Fold Number", fold])

        self._display_container = [
            pd.DataFrame(container, columns=["Information", "Value"])
        ]
        display = CommonDisplay(
            verbose=self.verbose,
            html_param=self.html_param,
        )
        if self.verbose:

            display.display(self._display_container[0].style.set_properties(**{'text-align': 'left'}))
    
        self._all_models, self._all_models_internal = self._get_models()
        self._all_metrics = self._get_metrics()

        self._setup_ran = True
        return self

    def Classifier_comparison_naloxone(
    self,
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    sort: str = "Accuracy",
    n_select: int = 1,
    turbo: bool = True,
    errors: str = "ignore",
    groups: Optional[Union[str, Any]] = None,
    probability_threshold: Optional[float] = None,
    engine: Optional[Dict[str, str]] = None,
    verbose: bool = True,

) -> Union[Any, List[Any]]:


        caller_params = dict(locals())

        if engine is not None:
            initial_model_engines = self.exp_model_engines.copy()
            for estimator, eng in engine.items():
                self._set_engine(estimator=estimator, engine=eng, severity="error")

        try:
            return_values = super().Classifier_comparison_naloxone(fold=fold,
                                                                   round=round, cross_validation=cross_validation,
                                                                   sort=sort, n_select=n_select,
                                                                   turbo=turbo, errors=errors,
                                                                   groups=groups,
                                                                   probability_threshold=probability_threshold,
                                                                   verbose=verbose,
                                                                   caller_params=caller_params)
        finally:
            if engine is not None:
                
                self._set_exp_model_engines(
                    container_default_engines=get_container_default_engines(),
                    engine=initial_model_engines,
                )

        return return_values

    def make_machine_learning_model(
        self,
        estimator: Union[str, Any],
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        groups: Optional[Union[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        engine: Optional[str] = None,
        verbose: bool = True,
        return_train_score: bool = False,
        **kwargs,
    ) -> Any:
        if engine is not None:
            initial_default_model_engines = self.exp_model_engines.copy()
            self._set_engine(estimator=estimator, engine=engine, severity="error")

        try:
            return_values = super().make_machine_learning_model(estimator=estimator, fold=fold, round=round,
                                                                cross_validation=cross_validation, groups=groups,
                                                                probability_threshold=probability_threshold,
                                                                verbose=verbose, return_train_score=return_train_score,
                                                                **kwargs)
        finally:
            if engine is not None:
                self._set_exp_model_engines(
                    container_default_engines=get_container_default_engines(),
                    engine=initial_default_model_engines,
                )

        return return_values



    def plot_machine(
        self,
        estimator,
        plot: str = "auc",
        scale: float = 1,
        save: bool = False,
        fold: Optional[Union[int, Any]] = None,
        plot_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
        display_format: Optional[str] = None,
    ) -> Optional[str]:


        return super().plot_machine(
            estimator=estimator,
            plot=plot,
            scale=scale,
            save=save,
            fold=fold,
            # fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
            groups=groups,
            verbose=verbose,
            display_format=display_format,
        )

    def evaluate_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        plot_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
    ):


        return super().evaluate_model(
            estimator=estimator,
            fold=fold,
            plot_kwargs=plot_kwargs,
            groups=groups,
        )



    def predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
        probability_threshold: Optional[float] = None,
        encoded_labels: bool = False,
        raw_score: bool = False,
        round: int = 4,
        verbose: bool = True,
    ) -> pd.DataFrame:


        return super().predict_model(
            estimator=estimator,
            data=data,
            probability_threshold=probability_threshold,
            encoded_labels=encoded_labels,
            raw_score=raw_score,
            round=round,
            verbose=verbose,
        )

    def models(
        self,
        type: Optional[str] = None,
        internal: bool = False,
        raise_errors: bool = True,
    ) -> pd.DataFrame:

        return super().models(type=type, internal=internal, raise_errors=raise_errors)

    def get_metrics(
        self,
        reset: bool = False,
        include_custom: bool = True,
        raise_errors: bool = True,
    ) -> pd.DataFrame:


        return super().get_metrics(
            reset=reset,
            include_custom=include_custom,
            raise_errors=raise_errors,
        )

    def add_metric(
        self,
        id: str,
        name: str,
        score_func: type,
        target: str = "pred",
        greater_is_better: bool = True,
        multiclass: bool = True,
        **kwargs,
    ) -> pd.Series:


        return super().add_metric(
            id=id,
            name=name,
            score_func=score_func,
            target=target,
            greater_is_better=greater_is_better,
            multiclass=multiclass,
            **kwargs,
        )

##------------------------------Salio(Mohtarami)---------------------------



_EXPERIMENT_CLASS = ClassificationExperiment
_CURRENT_EXPERIMENT: Optional[ClassificationExperiment] = None
_CURRENT_EXPERIMENT_EXCEPTION = (
    "_CURRENT_EXPERIMENT global variable is not set. Please run setup() first."
)
_CURRENT_EXPERIMENT_DECORATOR_DICT = {
    "_CURRENT_EXPERIMENT": _CURRENT_EXPERIMENT_EXCEPTION
}


def build_naloxone_model(
    data: Optional[DATAFRAME_LIKE] = None,
    target: TARGET_LIKE = -3,
    index: Union[bool, int, str, SEQUENCE_LIKE] = True,
    train_size: float = 0.7,
    test_data: Optional[DATAFRAME_LIKE] = None,
    preprocess: bool = True,
    data_split_shuffle: bool = True,
    data_split_stratify: Union[bool, List[str]] = True,
    fold_strategy: Union[str, Any] = "stratifiedkfold",
    fold: int = 10,
    fold_shuffle: bool = False,
    fold_groups: Optional[Union[str, pd.DataFrame]] = None,
    session_id: Optional[int] = None,
    verbose: bool = True,
):

    exp = _EXPERIMENT_CLASS()
    set_current_experiment(exp)
    return exp.build_naloxone_model(data=data, target=target, index=index, train_size=train_size, test_data=test_data,
                                    preprocess=preprocess, data_split_shuffle=data_split_shuffle,
                                    data_split_stratify=data_split_stratify, fold_strategy=fold_strategy, fold=fold,
                                    fold_shuffle=fold_shuffle, fold_groups=fold_groups, session_id=session_id,
                                    verbose=verbose)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def Classifier_comparison_naloxone(
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    sort: str = "Accuracy",
    n_select: int = 1,
    turbo: bool = True,
    errors: str = "ignore",
    groups: Optional[Union[str, Any]] = None,
    probability_threshold: Optional[float] = None,
    engine: Optional[Dict[str, str]] = None,
    verbose: bool = True,
) -> Union[Any, List[Any]]:


    return _CURRENT_EXPERIMENT.Classifier_comparison_naloxone(fold=fold, round=round, cross_validation=cross_validation,
                                                              sort=sort, n_select=n_select, turbo=turbo, errors=errors,
                                                              groups=groups,
                                                              probability_threshold=probability_threshold,
                                                              engine=engine, verbose=verbose)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_allowed_engines(estimator: str) -> Optional[str]:
    return _CURRENT_EXPERIMENT.get_allowed_engines(estimator=estimator)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_engine(estimator: str) -> Optional[str]:
    return _CURRENT_EXPERIMENT.get_engine(estimator=estimator)


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def make_machine_learning_model(
    estimator: Union[str, Any],
    fold: Optional[Union[int, Any]] = None,
    round: int = 4,
    cross_validation: bool = True,
    groups: Optional[Union[str, Any]] = None,
    probability_threshold: Optional[float] = None,
    engine: Optional[str] = None,
    verbose: bool = True,
    return_train_score: bool = False,
    **kwargs,
) -> Any:

    return _CURRENT_EXPERIMENT.make_machine_learning_model(estimator=estimator, fold=fold, round=round,
                                                           cross_validation=cross_validation, groups=groups,
                                                           probability_threshold=probability_threshold, engine=engine,
                                                           verbose=verbose, return_train_score=return_train_score,
                                                           **kwargs)

@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def plot_machine(
    estimator,
    plot: str = "auc",
    scale: float = 1,
    save: bool = False,
    fold: Optional[Union[int, Any]] = None,
    plot_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
    verbose: bool = True,
    display_format: Optional[str] = None,
) -> Optional[str]:

    return _CURRENT_EXPERIMENT.plot_machine(
        estimator=estimator,
        plot=plot,
        scale=scale,
        save=save,
        fold=fold,
        plot_kwargs=plot_kwargs,
        groups=groups,
        verbose=verbose,
        display_format=display_format,
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def evaluate_model(
    estimator,
    fold: Optional[Union[int, Any]] = None,
    plot_kwargs: Optional[dict] = None,
    groups: Optional[Union[str, Any]] = None,
):

    return _CURRENT_EXPERIMENT.evaluate_model(
        estimator=estimator,
        fold=fold,
        plot_kwargs=plot_kwargs,
        groups=groups,
    )

def predict_model(
    estimator,
    data: Optional[pd.DataFrame] = None,
    probability_threshold: Optional[float] = None,
    encoded_labels: bool = False,
    raw_score: bool = False,
    round: int = 4,
    verbose: bool = True,
) -> pd.DataFrame:

    experiment = _CURRENT_EXPERIMENT
    if experiment is None:
        experiment = _EXPERIMENT_CLASS()

    return experiment.predict_model(
        estimator=estimator,
        data=data,
        probability_threshold=probability_threshold,
        encoded_labels=encoded_labels,
        raw_score=raw_score,
        round=round,
        verbose=verbose,
    )

@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def models(
    type: Optional[str] = None,
    internal: bool = False,
    raise_errors: bool = True,
) -> pd.DataFrame:
    return _CURRENT_EXPERIMENT.models(
        type=type, internal=internal, raise_errors=raise_errors
    )


@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def get_metrics(
    reset: bool = False,
    include_custom: bool = True,
    raise_errors: bool = True,
) -> pd.DataFrame:

    return _CURRENT_EXPERIMENT.get_metrics(
        reset=reset,
        include_custom=include_custom,
        raise_errors=raise_errors,
    )
@check_if_global_is_not_none(globals(), _CURRENT_EXPERIMENT_DECORATOR_DICT)
def add_metric(
    id: str,
    name: str,
    score_func: type,
    target: str = "pred",
    greater_is_better: bool = True,
    multiclass: bool = True,
    **kwargs,
) -> pd.Series:
    return _CURRENT_EXPERIMENT.add_metric(
        id=id,
        name=name,
        score_func=score_func,
        target=target,
        greater_is_better=greater_is_better,
        multiclass=multiclass,
        **kwargs,
    )

def set_current_experiment(experiment: ClassificationExperiment) -> None:
    global _CURRENT_EXPERIMENT
    if not isinstance(experiment, ClassificationExperiment):
        raise TypeError(
            f"experiment must be a naloxlib ClassificationExperiment object, got {type(experiment)}."
        )
    _CURRENT_EXPERIMENT = experiment


def get_current_experiment() -> ClassificationExperiment:
    return _CURRENT_EXPERIMENT
#---------------------------------------Salio(Mohtarami)-------------------------


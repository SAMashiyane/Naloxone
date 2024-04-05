
import datetime
import gc
import time
import traceback
import warnings
from abc import abstractmethod
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from unittest.mock import patch
import numpy as np
import pandas as pd
import pandas.io.formats.style
from sklearn.base import clone
from sklearn.pipeline import Pipeline as skPipeline
from sklearn.utils.validation import check_is_fitted as check_fitted
import naloxlib.inward.add.sklearn
import naloxlib.inward.add.yellowbrick
import naloxlib.inward.preprocess
from naloxlib.box.metrics.base_metric import (
    get_all_metric_containers as get_all_class_metric_containers,
)
from naloxlib.inward.display import CommonDisplay, DummyDisplay
from naloxlib.inward.logging import get_logger, redirect_output
from naloxlib.inward.meta_estimators import (
    CustomProbabilityThresholdClassifier,
     get_estimator_from_meta_estimator,
)
from naloxlib.inward.metrics import EncodedDecodedLabelsReplaceScoreFunc, get_pos_label

from naloxlib.inward.add.sklearn import fit_and_score as fs
from naloxlib.inward.pipeline import (
    Pipeline,
    estimator_pipeline,
    get_pipeline_fit_kwargs,
)
from naloxlib.inward.naloxlib_experiment.tabular_experiment import _TabularExperiment
from naloxlib.inward.validation import is_fitted, is_sklearn_cv_generator

from naloxlib.efficacy.depend import DATAFRAME_LIKE, LABEL_COLUMN, SCORE_COLUMN
from naloxlib.efficacy.depend import (
    MLUsecase,
    color_df,
    get_label_encoder,
    get_ml_task,
    id_or_display_name,
)

try:
    from collections.abc import Iterable
except Exception:
    from collections import Iterable

# LOGGER = get_logger()


class _SupervisedExperiment(_TabularExperiment):
    _create_app_predict_kwargs = {}

    def __init__(self) -> None:
        super().__init__()
        self.transform_target_param = False  # Default False for both class/reg
        self._variable_keys = self._variable_keys.union(
            {
                "X",
                "y",
                "X_train",
                "X_test",
                "y_train",
                "y_test",
                "target_param",
                "fold_shuffle_param",
                "fold_generator",
                "fold_groups_param",
            }
        )

    def _calculate_metrics(
        self,
        y_test,
        pred,
        pred_prob,
        weights: Optional[list] = None,
        **additional_kwargs,
    ) -> dict:

        from naloxlib.efficacy.depend import calculate_metrics

        with redirect_output(self.logger):
            try:
                return calculate_metrics(
                    metrics=self._all_metrics,
                    y_test=y_test,
                    pred=pred,
                    pred_proba=pred_prob,
                    weights=weights,
                    **additional_kwargs,
                )
            except Exception:
                ml_usecase = get_ml_task(y_test)
                if ml_usecase == MLUsecase.CLASSIFICATION:
                    metrics = get_all_class_metric_containers(self.variables, True)
                elif ml_usecase == MLUsecase.REGRESSION:
                    pass

                return calculate_metrics(
                    metrics=metrics,  # type: ignore
                    y_test=y_test,
                    pred=pred,
                    pred_proba=pred_prob,
                    weights=weights,
                    **additional_kwargs,
                )

    def _is_unsupervised(self) -> bool:
        return False

    def _get_final_model_from_pipeline(
        self,
        pipeline: Pipeline,
        check_is_fitted: bool = False,
    ) -> Any:

        model = pipeline._final_estimator
        if check_is_fitted:
            check_fitted(model)

        return model

    def _choose_better(
        self,
        models_and_results: list,
        compare_dimension: str,
        fold: int,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        display: Optional[CommonDisplay] = None,
    ):


        self.logger.info("choose_better activated")
        if display is not None:
            display.update_monitor(1, "Compiling Final Results")

        if not fit_kwargs:
            fit_kwargs = {}

        for i, x in enumerate(models_and_results):
            if not isinstance(x, tuple):
                models_and_results[i] = (x, None)
            elif isinstance(x[0], str):
                models_and_results[i] = (x[1], None)
            elif len(x) != 2:
                raise ValueError(f"{x} must have length 2 but has {len(x)}")

        metric = self._get_metric_by_name_or_id(compare_dimension)

        best_result = None
        best_model = None
        for model, result in models_and_results:
            if result is not None and is_fitted(model):
                try:
                    indices = self._get_return_train_score_indices_for_logging(
                        return_train_score=True
                    )
                    result = result.loc[indices][compare_dimension]
                except KeyError:
                    indices = self._get_return_train_score_indices_for_logging(
                        return_train_score=False
                    )
                    result = result.loc[indices][compare_dimension]
            else:
                self.logger.info(
                    "SubProcess create_model() called ============Mohtarami================="
                )
                model, _ = self._create_model(
                    model,
                    verbose=False,
                    system=False,
                    fold=fold,
                    fit_kwargs=fit_kwargs,
                    groups=groups,
                )
                self.logger.info(
                    "SubProcess create_model() end =================================="
                )
                result = self.pull(pop=True).loc[
                    self._get_return_train_score_indices_for_logging(
                        return_train_score=False
                    )
                ][compare_dimension]
            self.logger.info(f"{model} result for {compare_dimension} is {result}")
            if not metric.greater_is_better:
                result *= -1
            if best_result is None or best_result < result:
                best_result = result
                best_model = model

        self.logger.info(f"{best_model} is best model")

        self.logger.info("choose_better completed")
        return best_model

    def _get_cv_n_folds(self, fold, X, y=None, groups=None):
        import naloxlib.efficacy.depend

        return naloxlib.efficacy.depend.get_cv_n_folds(
            fold, default=self.fold_generator, X=X, y=y, groups=groups
        )

    def _set_up_logging(
        self, runtime, log_data, log_profile, experiment_custom_tags=None
    ):
        # experiment custom tags
        if experiment_custom_tags is not None:
            if not isinstance(experiment_custom_tags, dict):
                raise TypeError(
                    "experiment_custom_tags parameter must be dict if not None"
                )

        if self.logging_param:
            self.logging_param.log_experiment(
                self,
                log_profile,
                log_data,
                experiment_custom_tags,
                runtime,
            )



    def _get_greater_is_worse_columns(self) -> Set[str]:
        input_ml_usecase = self._ml_usecase
        target_ml_usecase = MLUsecase.TIME_SERIES

        greater_is_worse_columns = {
            id_or_display_name(v, input_ml_usecase, target_ml_usecase).upper()
            for k, v in self._all_metrics.items()
            if not v.greater_is_better
        }
        greater_is_worse_columns.add("TT (Sec)")
        return greater_is_worse_columns

    def _highlight_models(self, master_display_: Any) -> Any:
        def highlight_max(s):
            to_highlight = s == s.max()
            return ["background-color: coral" if v else "" for v in to_highlight]

        def highlight_min(s):
            to_highlight = s == s.min()
            return ["background-color: coral" if v else "" for v in to_highlight]

        def highlight_cols(s):
            color = "lightblue"
            return f"background-color: {color}"

        greater_is_worse_columns = self._get_greater_is_worse_columns()
        if master_display_ is not None:
            return (
                master_display_.apply(
                    highlight_max,
                    subset=[
                        x
                        for x in master_display_.columns[1:]
                        if x not in greater_is_worse_columns
                    ],
                )
                .apply(
                    highlight_min,
                    subset=[
                        x
                        for x in master_display_.columns[1:]
                        if x in greater_is_worse_columns
                    ],
                )
                .applymap(highlight_cols, subset=["TT (Sec)"])
            )
        else:
            return pd.DataFrame().style

    def _process_sort(self, sort: Any) -> Tuple[str, bool]:
        input_ml_usecase = self._ml_usecase
        target_ml_usecase = MLUsecase.TIME_SERIES

        if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
            sort = self._get_metric_by_name_or_id(sort)
            if sort is None:
                raise ValueError(
                    "Sort method not supported. See docstring for list of available parameters."
                )

        if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
            sort_ascending = not sort.greater_is_better
            sort = id_or_display_name(sort, input_ml_usecase, target_ml_usecase)
        else:
            sort_ascending = True
            sort = "TT (Sec)"

        if self._ml_usecase == MLUsecase.TIME_SERIES:
            sort = sort.upper()

        return sort, sort_ascending

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
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        probability_threshold: Optional[float] = None,
        verbose: bool = True,
        caller_params: Optional[dict] = None,
    ) -> List[Any]:

        self._check_setup_ran()

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing compare_models()")
        self.logger.info(f"compare_models({function_params_str})")

        self.logger.info("Checking exceptions")

        available_estimators = self._all_models


        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )


        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")

        if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
            sort = self._get_metric_by_name_or_id(sort)
            if sort is None:
                raise ValueError(
                    "Sort method not supported. See docstring for list of available parameters."
                )

        possible_errors = ["ignore", "raise"]
        if errors not in possible_errors:
            raise ValueError(
                f"errors parameter must be one of: {', '.join(possible_errors)}."
            )

        if self._ml_usecase != MLUsecase.TIME_SERIES:
            fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        pd.set_option("display.max_columns", 500)

        self.logger.info("Preparing display monitor")

        len_mod = (
            len({k: v for k, v in self._all_models.items() if v.is_turbo})
            if turbo
            else len(self._all_models)
        )


        progress_args = {"max": (4 * len_mod) + 4 + min(len_mod, abs(n_select))}
        master_display_columns = (
            ["Model"]
            + [v.display_name for k, v in self._all_metrics.items()]
            + ["TT (Sec)"]
        )
        master_display = pd.DataFrame(columns=master_display_columns)
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            [
                "Status",
                ". . . . . . . . . . . . . . . . . .",
                "Loading Dependencies",
            ],
            [
                "Estimator",
                ". . . . . . . . . . . . . . . . . .",
                "Compiling Library",
            ],
        ]
        display = (
            DummyDisplay()
            if self._remote
            else CommonDisplay(
                verbose=verbose,
                html_param=self.html_param,
                progress_args=progress_args,
                monitor_rows=monitor_rows,
            )
        )
        if display.can_update_text:
            display.display(master_display, final_display=False)

        input_ml_usecase = self._ml_usecase
        target_ml_usecase = MLUsecase.TIME_SERIES

        np.random.seed(self.seed)

        display.move_progress()


        if not (isinstance(sort, str) and (sort == "TT" or sort == "TT (Sec)")):
            sort_ascending = not sort.greater_is_better
            sort = id_or_display_name(sort, input_ml_usecase, target_ml_usecase)
        else:
            sort_ascending = True
            sort = "TT (Sec)"



        display.update_monitor(1, "Loading Estimator")
        if turbo:
            model_library = [k for k, v in self._all_models.items() if v.is_turbo]
        else:
            model_library = list(self._all_models.keys())


        if self._ml_usecase == MLUsecase.TIME_SERIES:
            if "ensemble_forecaster" in model_library:
                warnings.warn(
                    "Unsupported estimator `ensemble_forecaster` for method `compare_models()`, removing from model_library"
                )
                model_library.remove("ensemble_forecaster")

        display.move_progress()


        import secrets

        URI = secrets.token_hex(nbytes=4)

        master_display = None
        master_display_ = None

        total_runtime_start = time.time()
        total_runtime = 0
        over_time_budget = False


        for i, model in enumerate(model_library):
            model_id = (
                model
                if (
                    isinstance(model, str)
                    and all(isinstance(m, str) for m in model_library)
                )
                else str(i)
            )
            model_name = self._get_model_name(model)

            if isinstance(model, str):
                self.logger.info(f"Initializing {model_name}")
            else:
                self.logger.info(f"Initializing custom model {model_name}")


            runtime_start = time.time()
            total_runtime += (runtime_start - total_runtime_start) / 60
            self.logger.info(f"Total runtime is {total_runtime} minutes")
            total_runtime_start = runtime_start
            display.update_monitor(2, model_name)
            self.logger.info(
                "SubProcess create_model() called =================================="
            )
            create_model_args = dict(
                estimator=model,
                system=False,
                verbose=False,
                display=display,
                fold=fold,
                round=round,
                cross_validation=cross_validation,
                # fit_kwargs=fit_kwargs,
                groups=groups,
                probability_threshold=probability_threshold,
                refit=False,
                error_score="raise" if errors == "raise" else 0.0,
            )
            results_columns_to_ignore = ["Object", "runtime", "cutoff"]

            try:
                model, model_fit_time = self._create_model(**create_model_args)
                model_results = self.pull(pop=True)
                assert (
                    np.sum(
                        model_results.drop(
                            results_columns_to_ignore, axis=1, errors="ignore"
                        ).iloc[0]
                    )
                    != 0.0
                )
            except Exception as ex:
                if errors == "raise":
                    raise RuntimeError(
                        f"create_model() failed for model {model}. {type(ex).__name__}: {ex}"
                    )

                self.logger.warning(
                    f"create_model() for {model} raised an exception or returned all 0.0, trying without fit_kwargs:"
                )
                self.logger.warning(traceback.format_exc())
                try:
                    model, model_fit_time = self._create_model(**create_model_args)
                    model_results = self.pull(pop=True)
                    assert (
                        np.sum(
                            model_results.drop(
                                results_columns_to_ignore, axis=1, errors="ignore"
                            ).iloc[0]
                        )
                        != 0.0
                    )
                except Exception:
                    self.logger.error(
                        f"create_model() for {model} raised an exception or returned all 0.0:"
                    )
                    self.logger.error(traceback.format_exc())
                    continue

            self.logger.info(
                "SubProcess create_model() end =================================="
            )

            if model is None:
                over_time_budget = True
                self.logger.info(
                    "Time budged exceeded in create_model(), breaking loop"
                )
                break

            runtime_end = time.time()
            runtime = np.array(runtime_end - runtime_start).round(2)

            self.logger.info("Creating metrics dataframe")
            if cross_validation:
                # cutoff only present in time series and when cv = True
                if "cutoff" in model_results.columns:
                    model_results.drop("cutoff", axis=1, errors="ignore")
                compare_models_ = pd.DataFrame(
                    model_results.loc[
                        self._get_return_train_score_indices_for_logging(
                            return_train_score=False
                        )
                    ]
                ).T.reset_index(drop=True)
            else:
                compare_models_ = pd.DataFrame(model_results.iloc[0]).T
            compare_models_.insert(
                len(compare_models_.columns), "TT (Sec)", model_fit_time
            )
            compare_models_.insert(0, "Model", model_name)
            compare_models_.insert(0, "Object", [model])
            compare_models_.insert(0, "runtime", runtime)
            compare_models_.index = [model_id]
            if master_display is None:
                master_display = compare_models_
            else:
                master_display = pd.concat(
                    [master_display, compare_models_], ignore_index=False
                )
            master_display = master_display.round(round)
            if self._ml_usecase != MLUsecase.TIME_SERIES:
                master_display = master_display.sort_values(
                    by=sort, ascending=sort_ascending
                )
            else:
                master_display = master_display.sort_values(
                    by=sort.upper(), ascending=sort_ascending
                )

            master_display_ = master_display.drop(
                results_columns_to_ignore, axis=1, errors="ignore"
            ).style.format(precision=round)
            master_display_ = master_display_.set_properties(**{"text-align": "left"})
            master_display_ = master_display_.set_table_styles(
                [dict(selector="th", props=[("text-align", "left")])]
            )

            if display.can_update_text:
                display.display(master_display_, final_display=False)

        display.move_progress()

        compare_models_ = self._highlight_models(master_display_)

        display.update_monitor(1, "Compiling Final Models")

        display.move_progress()

        sorted_models = []

        if master_display is not None:
            clamped_n_select = min(len(master_display), abs(n_select))
            if n_select < 0:
                n_select_range = range(
                    len(master_display) - clamped_n_select, len(master_display)
                )
            else:
                n_select_range = range(0, clamped_n_select)

            if self.logging_param:
                self.logging_param.log_model_comparison(
                    master_display, "compare_models"
                )

            for index, row in enumerate(master_display.iterrows()):
                _, row = row
                model = row["Object"]

                results = row.to_frame().T.drop(
                    ["Object", "Model", "runtime", "TT (Sec)"], errors="ignore", axis=1
                )

                avgs_dict_log = {k: v for k, v in results.iloc[0].items()}

                full_logging = False

                if index in n_select_range:
                    display.update_monitor(2, self._get_model_name(model))
                    create_model_args = dict(
                        estimator=model,
                        system=False,
                        verbose=False,
                        fold=fold,
                        round=round,
                        cross_validation=False,
                        predict=False,
                        # fit_kwargs=fit_kwargs,
                        groups=groups,
                        probability_threshold=probability_threshold,
                    )
                    if errors == "raise":
                        model, model_fit_time = self._create_model(**create_model_args)
                        sorted_models.append(model)
                    else:
                        try:
                            model, model_fit_time = self._create_model(
                                **create_model_args
                            )
                            sorted_models.append(model)
                            assert (
                                np.sum(
                                    model_results.drop(
                                        results_columns_to_ignore,
                                        axis=1,
                                        errors="ignore",
                                    ).iloc[0]
                                )
                                != 0.0
                            )
                        except Exception:
                            self.logger.error(
                                f"create_model() for {model} raised an exception or returned all 0.0:"
                            )
                            self.logger.error(traceback.format_exc())
                            model = None
                            display.move_progress()
                            continue
                    display.move_progress()
                    full_logging = True

                if self.logging_param and cross_validation and model is not None:
                    self._log_model(
                        model=model,
                        model_results=results,
                        score_dict=avgs_dict_log,
                        source="compare_models",
                        runtime=row["runtime"],
                        model_fit_time=row["TT (Sec)"],
                        pipeline=self.pipeline,
                        log_plots=self.log_plots_param if full_logging else [],
                        log_holdout=full_logging,
                        URI=URI,
                        display=display,
                        experiment_custom_tags=experiment_custom_tags,
                    )

        if len(sorted_models) == 1:
            sorted_models = sorted_models[0]

        display.display(compare_models_, final_display=True)

        pd.reset_option("display.max_columns")

        # store in display container
        self._display_container.append(compare_models_.data)

        self.logger.info(
            f"_master_model_container: {len(self._master_model_container)}"
        )
        self.logger.info(f"_display_container: {len(self._display_container)}")

        self.logger.info(str(sorted_models))
        self.logger.info(
            "compare_models() successfully completed......................................"
        )

        return sorted_models

    def _create_model_without_cv(
        self,
        model,
        data_X,
        data_y,
        fit_kwargs,
        round,
        predict,
        system,
        display: CommonDisplay,
        model_only: bool = True,
        return_train_score: bool = False,
    ):
        with estimator_pipeline(self.pipeline, model) as pipeline_with_model:
            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
            self.logger.info("Cross validation set to False")

            self.logger.info("Fitting Model")
            model_fit_start = time.time()
            with redirect_output(self.logger):
                pipeline_with_model.fit(data_X, data_y, **fit_kwargs)
            model_fit_end = time.time()

            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

            display.move_progress()

            if predict:
                if return_train_score:

                    _SupervisedExperiment.predict_model(
                        self,
                        pipeline_with_model,
                        data=pd.concat([data_X, data_y], axis=1),
                        verbose=False,
                    )
                    train_results = self.pull(pop=True).drop("Model", axis=1)
                    train_results.index = ["Train"]
                else:
                    train_results = None

                self.predict_model(pipeline_with_model, verbose=False)
                model_results = self.pull(pop=True).drop("Model", axis=1)
                model_results.index = ["Test"]
                if train_results is not None:
                    model_results = pd.concat([model_results, train_results])

                self._display_container.append(model_results)

                model_results = model_results.style.format(precision=round)

                if system:
                    display.display(model_results)

                self.logger.info(f"_display_container: {len(self._display_container)}")

            if not model_only:
                return pipeline_with_model, model_fit_time

        return model, model_fit_time

    def _create_model_with_cv(
        self,
        model,
        data_X,
        data_y,
        fit_kwargs,
        round,
        cv,
        groups,
        metrics,
        refit,

        display,
        error_score,
        return_train_score: bool = False,
    ):

        display.update_monitor(
            1,
            f"Fitting {self._get_cv_n_folds(cv, data_X, y=data_y, groups=groups)} Folds",
        )


        from sklearn.model_selection import cross_validate

        metrics_dict = dict([(k, v.scorer) for k, v in metrics.items()])

        self.logger.info("Starting cross validation")

        n_jobs = self.gpu_n_jobs_param
        from sklearn.gaussian_process import (
            GaussianProcessClassifier,
            GaussianProcessRegressor,
        )


        if isinstance(model, (GaussianProcessClassifier, GaussianProcessRegressor)):
            n_jobs = 1

        with estimator_pipeline(self.pipeline, model) as pipeline_with_model:
            fit_kwargs = get_pipeline_fit_kwargs(pipeline_with_model, fit_kwargs)
            self.logger.info(f"Cross validating with {cv}, n_jobs={n_jobs}")


            with patch("sklearn.model_selection._validation._fit_and_score", fs):
                model_fit_start = time.time()
                with redirect_output(self.logger):
                    scores = cross_validate(
                        pipeline_with_model,
                        data_X,
                        data_y,
                        cv=cv,
                        groups=groups,
                        scoring=metrics_dict,
                        params=fit_kwargs,
                        n_jobs=n_jobs,
                        return_train_score=return_train_score,
                        error_score=error_score,
                    )

            model_fit_end = time.time()
            model_fit_time = np.array(model_fit_end - model_fit_start).round(2)

            score_dict = {}
            for k, v in metrics.items():
                score_dict[v.display_name] = []
                if return_train_score:
                    train_score = scores[f"train_{k}"] * (
                        1 if v.greater_is_better else -1
                    )
                    train_score = train_score.tolist()
                    score_dict[v.display_name] = train_score
                test_score = scores[f"test_{k}"] * (1 if v.greater_is_better else -1)
                test_score = test_score.tolist()
                score_dict[v.display_name] += test_score

            self.logger.info("Calculating mean and std")

            avgs_dict = {}
            for k, v in metrics.items():
                avgs_dict[v.display_name] = []
                if return_train_score:
                    train_score = scores[f"train_{k}"] * (
                        1 if v.greater_is_better else -1
                    )
                    train_score = train_score.tolist()
                    avgs_dict[v.display_name] = [
                        np.mean(train_score),
                        np.std(train_score),
                    ]
                test_score = scores[f"test_{k}"] * (1 if v.greater_is_better else -1)
                test_score = test_score.tolist()
                avgs_dict[v.display_name] += [np.mean(test_score), np.std(test_score)]

            display.move_progress()

            self.logger.info("Creating metrics dataframe")

            if hasattr(cv, "n_splits"):
                fold = cv.n_splits
            elif hasattr(cv, "get_n_splits"):
                fold = cv.get_n_splits(groups=groups)
            else:
                raise ValueError(
                    "The cross validation class should implement a n_splits "
                    f"attribute or a get_n_splits method. {cv.__class__.__name__} "
                    "has neither."
                )

            if return_train_score:
                model_results = pd.DataFrame(
                    {
                        "Split": ["CV-Train"] * fold
                        + ["CV-Val"] * fold
                        + ["CV-Train"] * 2
                        + ["CV-Val"] * 2,
                        "Fold": np.arange(fold).tolist()
                        + np.arange(fold).tolist()
                        + ["Mean", "Std"] * 2,
                    }
                )
            else:
                model_results = pd.DataFrame(
                    {
                        "Fold": np.arange(fold).tolist() + ["Mean", "Std"],
                    }
                )

            model_scores = pd.concat(
                [pd.DataFrame(score_dict), pd.DataFrame(avgs_dict)]
            ).reset_index(drop=True)

            model_results = pd.concat([model_results, model_scores], axis=1)
            model_results.set_index(
                self._get_return_train_score_columns_for_display(return_train_score),
                inplace=True,
            )

            if refit:

                display.update_monitor(1, "Finalizing Model")
                model_fit_start = time.time()
                self.logger.info("Finalizing model")
                with redirect_output(self.logger):
                    pipeline_with_model.fit(data_X, data_y, **fit_kwargs)
                    model_fit_end = time.time()


                if return_train_score:

                    _SupervisedExperiment.predict_model(
                        self,
                        pipeline_with_model,
                        data=pd.concat([data_X, data_y], axis=1),
                        verbose=False,
                    )
                    metrics = self.pull(pop=True).drop("Model", axis=1)
                    df_score = pd.DataFrame({"Split": ["Train"], "Fold": [None]})
                    df_score = pd.concat([df_score, metrics], axis=1)
                    df_score.set_index(["Split", "Fold"], inplace=True)
                    model_results = pd.concat([model_results, df_score])

                model_fit_time = np.array(model_fit_end - model_fit_start).round(2)
            else:
                model_fit_time /= self._get_cv_n_folds(
                    cv, data_X, y=data_y, groups=groups
                )

        model_results = model_results.round(round)

        return model, model_fit_time, model_results, avgs_dict

    def _get_return_train_score_columns_for_display(
        self, return_train_score: bool
    ) -> List[str]:
        if return_train_score:
            columns = ["Split", "Fold"]
        else:
            columns = ["Fold"]
        return columns

    def _get_return_train_score_indices_for_logging(self, return_train_score: bool):
        if return_train_score:
            indices = ("CV-Val", "Mean")
        else:
            indices = "Mean"
        return indices

    def _highlight_and_round_model_results(
        self, model_results: pd.DataFrame, return_train_score: bool, round: int
    ) -> pandas.io.formats.style.Styler:
        # yellow the mean
        if return_train_score:
            indices = [("CV-Val", "Mean"), ("CV-Train", "Mean")]
        else:
            indices = ["Mean"]
        model_results = color_df(model_results, "yellow", indices, axis=1)
        model_results = model_results.format(precision=round)
        return model_results

    @abstractmethod
    def _create_model_get_train_X_y(self, X_train, y_train):

        pass

    def _create_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        predict: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        refit: bool = True,
        probability_threshold: Optional[float] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        system: bool = True,
        add_to_model_list: bool = True,
        X_train_data: Optional[pd.DataFrame] = None,  # added in naloxlib==2.2.0
        y_train_data: Optional[pd.DataFrame] = None,  # added in naloxlib==2.2.0
        metrics=None,
        display: Optional[CommonDisplay] = None,  # added in naloxlib==2.2.0
        model_only: bool = True,
        return_train_score: bool = False,
        error_score: Union[str, float] = 0.0,
        **kwargs,
    ) -> Any:

        self._check_setup_ran()

        function_params_str = ", ".join(
            [
                f"{k}={v}"
                for k, v in locals().items()
                if k not in ("X_train_data", "y_train_data")
            ]
        )

        self.logger.info("Initializing create_model()")
        self.logger.info(f"create_model({function_params_str})")

        self.logger.info("Checking exceptions")

        # run_time
        runtime_start = time.time()

        available_estimators = set(self._all_models_internal.keys())

        if not fit_kwargs:
            fit_kwargs = {}


        if isinstance(estimator, str):
            if estimator not in available_estimators:
                raise ValueError(
                    f"Estimator {estimator} not available. Please see docstring for list of available estimators."
                )
        elif not hasattr(estimator, "fit"):
            raise ValueError(
                f"Estimator {estimator} does not have the required fit() method."
            )


        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None, an integer or a scikit-learn compatible CV generator object."
            )


        if type(round) is not int:
            raise TypeError("Round parameter only accepts integer value.")


        if type(verbose) is not bool:
            raise TypeError(
                "Verbose parameter can only take argument as True or False."
            )


        if type(system) is not bool:
            raise TypeError("System parameter can only take argument as True or False.")


        if type(cross_validation) is not bool:
            raise TypeError(
                "cross_validation parameter can only take argument as True or False."
            )


        if type(return_train_score) is not bool:
            raise TypeError(
                "return_train_score can only take argument as True or False"
            )



        if not display:
            progress_args = {"max": 4}
            timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
            monitor_rows = [
                ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
                [
                    "Status",
                    ". . . . . . . . . . . . . . . . . .",
                    "Loading Dependencies",
                ],
                [
                    "Estimator",
                    ". . . . . . . . . . . . . . . . . .",
                    "Compiling Library",
                ],
            ]
            display = CommonDisplay(
                verbose=verbose,
                html_param=self.html_param,
                progress_args=progress_args,
                monitor_rows=monitor_rows,
            )

        self.logger.info("Importing libraries")



        np.random.seed(self.seed)

        self.logger.info("Copying training dataset")


        data_X, data_y = self._create_model_get_train_X_y(
            X_train=X_train_data, y_train=y_train_data
        )

        groups = self._get_groups(groups, data=data_X)

        if metrics is None:
            metrics = self._all_metrics

        display.move_progress()

        self.logger.info("Defining folds")

        # cross validation setup starts here
        if self._ml_usecase == MLUsecase.TIME_SERIES:
            cv = self.get_fold_generator(fold=fold)

            # Add forecast horizon
            fit_kwargs = self.update_fit_kwargs_with_fh_from_cv(
                fit_kwargs=fit_kwargs, cv=cv
            )

        else:
            cv = self._get_cv_splitter(fold)

        self.logger.info("Declaring metric variables")

        display.update_monitor(1, "Selecting Estimator")

        self.logger.info("Importing untrained model")

        if isinstance(estimator, str) and estimator in available_estimators:
            model_definition = self._all_models_internal[estimator]
            model_args = model_definition.args
            model_args = {**model_args, **kwargs}
            model = model_definition.class_def(**model_args)
            full_name = model_definition.name
        else:
            self.logger.info("Declaring custom model")

            model = clone(estimator)
            model.set_params(**kwargs)

            full_name = self._get_model_name(model)


        model = clone(model)

        display.update_monitor(2, full_name)

        if probability_threshold is not None:
            if self._ml_usecase != MLUsecase.CLASSIFICATION or self.is_multiclass:
                raise ValueError(
                    "Cannot use probability_threshold with non-binary "
                    "classification usecases."
                )
            if not isinstance(model, CustomProbabilityThresholdClassifier):
                model = CustomProbabilityThresholdClassifier(
                    classifier=model,
                    probability_threshold=probability_threshold,
                )
            else:
                model.set_params(probability_threshold=probability_threshold)
        self.logger.info(f"{full_name} Imported successfully")

        display.move_progress()

        """
        MONITOR UPDATE STARTS
        """
        if not cross_validation:
            display.update_monitor(1, f"Fitting {str(full_name)}")
        else:
            display.update_monitor(1, "Initializing CV")


        if not cross_validation:
            model, model_fit_time = self._create_model_without_cv(
                model=model,
                data_X=data_X,
                data_y=data_y,
                fit_kwargs=fit_kwargs,
                round=round,
                predict=predict,
                system=system,
                display=display,
                model_only=model_only,
                return_train_score=return_train_score,
            )

            display.move_progress()

            self.logger.info(str(model))
            self.logger.info(
                "create_model() successfully completed......................................"
            )

            gc.collect()

            if not system:
                return model, model_fit_time
            return model

        model, model_fit_time, model_results, _ = self._create_model_with_cv(model=model, data_X=data_X, data_y=data_y,
                                                                             fit_kwargs=fit_kwargs, round=round, cv=cv,
                                                                             groups=groups, metrics=metrics,
                                                                             refit=refit, display=display,
                                                                             error_score=error_score,
                                                                             return_train_score=return_train_score)

        # end runtime
        runtime_end = time.time()
        runtime = np.array(runtime_end - runtime_start).round(2)


        if self.logging_param and system and refit:
            indices = self._get_return_train_score_indices_for_logging(
                return_train_score
            )
            avgs_dict_log = {k: v for k, v in model_results.loc[indices].items()}

            self._log_model(
                model=model,
                model_results=model_results,
                score_dict=avgs_dict_log,
                source="create_model",
                runtime=runtime,
                model_fit_time=model_fit_time,
                pipeline=self.pipeline,
                log_plots=self.log_plots_param,
                experiment_custom_tags=experiment_custom_tags,
                display=display,
            )

        display.move_progress()

        self.logger.info("Uploading results into container")

        if not self._ml_usecase == MLUsecase.TIME_SERIES:
            model_results.drop("cutoff", axis=1, inplace=True, errors="ignore")

        self._display_container.append(model_results)

        # storing results in _master_model_container
        if add_to_model_list:
            self.logger.info("Uploading model into container now")
            self._master_model_container.append(
                {"model": model, "scores": model_results, "cv": cv}
            )


        model_results = self._highlight_and_round_model_results(
            model_results, return_train_score, round
        )
        if system:
            display.display(model_results)

        self.logger.info(
            f"_master_model_container: {len(self._master_model_container)}"
        )
        self.logger.info(f"_display_container: {len(self._display_container)}")

        self.logger.info(str(model))
        self.logger.info(
            "create_model() successfully completed......................................"
        )
        gc.collect()

        if not system:
            return model, model_fit_time

        return model

    def make_machine_learning_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        round: int = 4,
        cross_validation: bool = True,
        predict: bool = True,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        refit: bool = True,
        probability_threshold: Optional[float] = None,
        experiment_custom_tags: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        return_train_score: bool = False,
        **kwargs,
    ) -> Any:


        # TODO improve error message
        assert not any(
            x
            in (
                "system",
                "add_to_model_list",
                "X_train_data",
                "y_train_data",
                "metrics",
            )
            for x in kwargs
        )
        return self._create_model(
            estimator=estimator,
            fold=fold,
            round=round,
            cross_validation=cross_validation,
            predict=predict,
            fit_kwargs=fit_kwargs,
            groups=groups,
            refit=refit,
            probability_threshold=probability_threshold,
            experiment_custom_tags=experiment_custom_tags,
            verbose=verbose,
            return_train_score=return_train_score,
            **kwargs,
        )

    def models(
        self,
        type: Optional[str] = None,
        internal: bool = False,
        raise_errors: bool = True,
    ) -> pd.DataFrame:


        model_type = {
            "linear": [
                "lr",
                "ridge",
                "svm",
                "lasso",
                "en",
                "lar",
                "llar",
                "omp",
                "br",
                "ard",
                "par",
                "ransac",
                "tr",
                "huber",
                "kr",
            ],
            "tree": ["dt"],
            "ensemble": [
                "rf",
                "et",
                "gbc",
                "gbr",
                "xgboost",
                "lightgbm",
                "catboost",
                "ada",
            ],
        }

        def filter_model_df_by_type(df):
            if not type:
                return df
            return df[df.index.isin(model_type[type])]

        # Check if type is valid
        if type not in list(model_type) + [None]:
            raise ValueError(
                f"type parameter only accepts {', '.join(list(model_type) + str(None))}."
            )

        self.logger.info(f"gpu_param set to {self.gpu_param}")

        _, model_containers = self._get_models(raise_errors)

        rows = [
            v.get_dict(internal)
            for k, v in model_containers.items()
            if (internal or not v.is_special)
        ]

        df = pd.DataFrame(rows)
        df.set_index("ID", inplace=True, drop=True)

        return filter_model_df_by_type(df)

    def get_metrics(
        self,
        reset: bool = False,
        include_custom: bool = True,
        raise_errors: bool = True,
    ) -> pd.DataFrame:

        if reset and not self._setup_ran:
            raise ValueError("setup() needs to be ran first.")

        np.random.seed(self.seed)

        if reset:
            self._all_metrics = self._get_metrics(raise_errors=raise_errors)

        metric_containers = self._all_metrics
        rows = [v.get_dict() for k, v in metric_containers.items()]

        df = pd.DataFrame(rows)
        df.set_index("ID", inplace=True, drop=True)

        if not include_custom:
            df = df[df["Custom"] is False]

        return df

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

        if not self._setup_ran:
            raise ValueError("setup() needs to be ran first.")

        if id in self._all_metrics:
            raise ValueError("id already present in metrics dataframe.")

        if self._ml_usecase == MLUsecase.CLASSIFICATION:
            new_metric = (
                naloxlib.containers.metrics.classification.ClassificationMetricContainer(
                    id=id,
                    name=name,
                    score_func=EncodedDecodedLabelsReplaceScoreFunc(
                        score_func, get_pos_label(self.__dict__)
                    ),
                    target=target,
                    args=kwargs,
                    display_name=name,
                    greater_is_better=greater_is_better,
                    is_multiclass=bool(multiclass),
                    is_custom=True,
                )
            )
        elif self._ml_usecase == MLUsecase.TIME_SERIES:
            new_metric = (
                naloxlib.containers.metrics.time_series.TimeSeriesMetricContainer(
                    id=id,
                    name=name,
                    score_func=score_func,
                    args=kwargs,
                    display_name=name,
                    greater_is_better=greater_is_better,
                    is_custom=True,
                )
            )
        else:
            new_metric = (
                naloxlib.containers.metrics.regression.RegressionMetricContainer(
                    id=id,
                    name=name,
                    score_func=score_func,
                    args=kwargs,
                    display_name=name,
                    greater_is_better=greater_is_better,
                    is_custom=True,
                )
            )

        self._all_metrics[id] = new_metric

        new_metric = new_metric.get_dict()

        new_metric = pd.Series(new_metric, name=id.replace(" ", "_")).drop("ID")

        return new_metric



    def predict_model(
        self,
        estimator,
        data: Optional[pd.DataFrame] = None,
        probability_threshold: Optional[float] = None,
        encoded_labels: bool = False,  # added in naloxlib==2.1.0
        raw_score: bool = False,
        round: int = 4,  # added in naloxlib==2.2.0
        verbose: bool = True,
        ml_usecase: Optional[MLUsecase] = None,
        preprocess: Union[bool, str] = True,
    ) -> pd.DataFrame:


        def encode_labels(label_encoder, labels: pd.Series) -> pd.Series:
            # Check if there is a LabelEncoder in the pipeline
            if label_encoder:
                return pd.Series(
                    data=label_encoder.transform(labels),
                    name=labels.name,
                    index=labels.index,
                )
            else:
                return labels

        function_params_str = ", ".join(
            [f"{k}={v}" for k, v in locals().items() if k != "data"]
        )

        self.logger.info("Initializing predict_model()")
        self.logger.info(f"predict_model({function_params_str})")

        self.logger.info("Checking exceptions")


        if ml_usecase is None:
            ml_usecase = self._ml_usecase

        if data is None and not self._setup_ran:
            raise ValueError(
                "data parameter may not be None without running setup() first."
            )

        if probability_threshold is not None:
            # probability_threshold allowed types
            allowed_types = [int, float]
            if (
                type(probability_threshold) not in allowed_types
                or probability_threshold > 1
                or probability_threshold < 0
            ):
                raise TypeError(
                    "probability_threshold parameter only accepts value between 0 to 1."
                )


        self.logger.info("Preloading libraries")

        try:
            np.random.seed(self.seed)
            display = CommonDisplay(
                verbose=verbose,
                html_param=self.html_param,
            )
        except Exception:
            display = CommonDisplay(
                verbose=False,
                html_param=False,
            )

        if isinstance(estimator, skPipeline):
            if not hasattr(estimator, "feature_names_in_"):
                raise ValueError(
                    "If estimator is a Pipeline, it must implement `feature_names_in_`."
                )

            pipeline = copy(estimator)


            final_step = pipeline.steps[-1]
            estimator = final_step[-1]
            pipeline.steps = pipeline.steps[:-1]
        elif not self._setup_ran:
            raise ValueError(
                "If estimator is not a Pipeline, you must run setup() first."
            )
        else:
            pipeline = self.pipeline
            final_step = None

        X_columns = pipeline.feature_names_in_[:-1]
        y_name = pipeline.feature_names_in_[-1]
        y_test_ = None
        if data is None:
            X_test_, y_test_ = self.X_test_transformed, self.y_test_transformed
            X_test_untransformed = self.X_test[self.X_test.index.isin(X_test_.index)]
            y_test_untransformed = self.y_test[self.y_test.index.isin(y_test_.index)]
        else:
            if y_name in data.columns:
                data = self._set_index(self._prepare_dataset(data, y_name))
                target = data[y_name]
                data = data.drop(y_name, axis=1)
            else:
                data = self._set_index(self._prepare_dataset(data))
                target = None
            X_test_untransformed = data
            y_test_untransformed = target
            data = data[X_columns]  # Ignore all columns but the originals
            if preprocess:
                X_test_ = pipeline.transform(
                    X=data,
                    y=(target if preprocess != "features" else None),
                )
                if final_step:
                    pipeline.steps.append(final_step)

                if isinstance(X_test_, tuple):
                    X_test_, y_test_ = X_test_
                elif target is not None:
                    y_test_ = target
            else:
                X_test_ = data
                y_test_ = target


            X_test_untransformed = X_test_untransformed[
                X_test_untransformed.index.isin(X_test_.index)
            ]
            if target is not None:
                y_test_untransformed = y_test_untransformed[
                    y_test_untransformed.index.isin(X_test_.index)
                ]


        if isinstance(estimator, CustomProbabilityThresholdClassifier):
            if probability_threshold is None:
                probability_threshold = estimator.probability_threshold
            estimator = get_estimator_from_meta_estimator(estimator)

        pred = np.nan_to_num(estimator.predict(X_test_))
        pred = pipeline.inverse_transform(pred)
        # Need to convert labels back to numbers
        # TODO optimize
        label_encoder = get_label_encoder(pipeline)
        if isinstance(pred, pd.Series):
            pred = pred.values

        try:

            score = estimator.predict_proba(X_test_)

            if len(np.unique(pred)) <= 2:
                pred_prob = score[:, 1]
            else:
                pred_prob = score

        except Exception:

            score = None
            pred_prob = None

        y_test_metrics = y_test_untransformed

        if probability_threshold is not None and pred_prob is not None:
            try:
                pred = (pred_prob >= probability_threshold).astype(int)
                if label_encoder:
                    pred = label_encoder.inverse_transform(pred)
            except Exception:
                pass

        if pred_prob is None:
            pred_prob = pred

        df_score = None
        if y_test_ is not None and self._setup_ran:
            # model name
            full_name = self._get_model_name(estimator)
            metrics = self._calculate_metrics(y_test_metrics, pred, pred_prob)  # type: ignore
            df_score = pd.DataFrame(metrics, index=[0])
            df_score.insert(0, "Model", full_name)
            df_score = df_score.round(round)
            display.display(df_score.style.format(precision=round))

        if ml_usecase == MLUsecase.CLASSIFICATION:
            try:
                pred = pred.astype(int)
            except Exception:
                pass

        label = pd.DataFrame(
            pred, columns=[LABEL_COLUMN], index=X_test_untransformed.index
        )

        if encoded_labels:
            label[LABEL_COLUMN] = encode_labels(label_encoder, label[LABEL_COLUMN])
        old_index = X_test_untransformed.index
        X_test_ = pd.concat([X_test_untransformed, y_test_untransformed, label], axis=1)
        X_test_.index = old_index

        if score is not None:
            if not raw_score:
                if label_encoder:
                    pred = label_encoder.transform(pred)

                score = pd.DataFrame(
                    data=[s[pred[i]] for i, s in enumerate(score)],
                    index=X_test_.index,
                    columns=[SCORE_COLUMN],
                )
            else:
                if not encoded_labels:
                    if label_encoder:
                        columns = label_encoder.classes_
                    else:
                        columns = range(score.shape[1])
                else:
                    columns = range(score.shape[1])

                score = pd.DataFrame(
                    data=score,
                    index=X_test_.index,
                    columns=[f"{SCORE_COLUMN}_{col}" for col in columns],
                )

            score = score.round(round)
            X_test_ = pd.concat((X_test_, score), axis=1)

        # store predictions on hold-out in _display_container
        if df_score is not None:
            self._display_container.append(df_score)

        gc.collect()
        return X_test_

    def get_leaderboard(
        self,
        finalize_models: bool = False,
        model_only: bool = False,
        fit_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        verbose: bool = True,
    ):
        """
        generates leaderboard for all models run in current run.
        """
        model_container = self._master_model_container

        progress_args = {"max": len(model_container) + 1}
        timestampStr = datetime.datetime.now().strftime("%H:%M:%S")
        monitor_rows = [
            ["Initiated", ". . . . . . . . . . . . . . . . . .", timestampStr],
            [
                "Status",
                ". . . . . . . . . . . . . . . . . .",
                "Loading Dependencies",
            ],
            [
                "Estimator",
                ". . . . . . . . . . . . . . . . . .",
                "Compiling Library",
            ],
        ]
        display = CommonDisplay(
            verbose=verbose,
            html_param=self.html_param,
            progress_args=progress_args,
            monitor_rows=monitor_rows,
        )

        result_container_mean = []
        finalized_models = []

        display.update_monitor(
            1, "Finalizing models" if finalize_models else "Collecting models"
        )
        for i, model_results_tuple in enumerate(model_container):
            model_results = model_results_tuple["scores"]
            model = model_results_tuple["model"]
            try:
                mean_scores = model_results.loc[["Mean"]]
            except KeyError:
                continue
            model_name = self._get_model_name(model)
            mean_scores["Index"] = i
            mean_scores["Model Name"] = model_name
            display.update_monitor(2, model_name)
            if finalize_models:
                model = self.finalize_model(
                    model,
                    fit_kwargs=fit_kwargs,
                    groups=groups,
                    model_only=model_only,
                )
            else:
                model = deepcopy(model)
                if not is_fitted(model):
                    model, _ = self._create_model(
                        estimator=model,
                        verbose=False,
                        system=False,
                        fit_kwargs=fit_kwargs,
                        groups=groups,
                        add_to_model_list=False,
                    )
                if not model_only:
                    pipeline = deepcopy(self.pipeline)
                    pipeline.steps.append(["trained_model", model])
                    model = pipeline
            display.move_progress()
            finalized_models.append(model)
            result_container_mean.append(mean_scores)

        display.update_monitor(1, "Creating dataframe")
        results = pd.concat(result_container_mean)
        results["Model"] = list(range(len(results)))
        results["Model"] = results["Model"].astype("object")
        model_loc = results.columns.get_loc("Model")
        for x in range(len(results)):
            results.iat[x, model_loc] = finalized_models[x]
        rearranged_columns = list(results.columns)
        rearranged_columns.remove("Model")
        rearranged_columns.remove("Model Name")
        rearranged_columns = ["Model Name", "Model"] + rearranged_columns
        results = results[rearranged_columns]
        results.set_index("Index", inplace=True, drop=True)
        display.close()
        # display.clear_output()
        return results

    @property
    @abstractmethod
    def X(self):
        """Feature set."""
        pass

    @property
    @abstractmethod
    def dataset_transformed(self):
        """Transformed dataset."""
        pass

    @property
    @abstractmethod
    def X_train_transformed(self):
        """Transformed feature set of the training set."""
        pass

    @property
    @abstractmethod
    def train_transformed(self):
        """Transformed training set."""
        pass

    @property
    @abstractmethod
    def X_transformed(self):
        """Transformed feature set."""
        pass

    @property
    def y(self):
        """Target column."""
        return self.dataset[self.target_param]

    @property
    @abstractmethod
    def X_train(self):
        """Feature set of the training set."""
        pass

    @property
    @abstractmethod
    def X_test(self):
        """Feature set of the test set."""
        pass

    @property
    def train(self):
        """Training set."""
        return self.dataset.loc[self.idx[0], :]

    @property
    @abstractmethod
    def test(self):
        """Test set."""
        pass

    @property
    def y_train(self):
        """Target column of the training set."""
        return self.train[self.target_param]

    @property
    def y_test(self):
        """Target column of the test set."""
        return self.test[self.target_param]

    @property
    @abstractmethod
    def test_transformed(self):
        """Transformed test set."""
        pass

    @property
    @abstractmethod
    def y_transformed(self):
        """Transformed target column."""
        pass

    @property
    @abstractmethod
    def X_test_transformed(self):
        """Transformed feature set of the test set."""
        pass

    @property
    @abstractmethod
    def y_train_transformed(self):
        """Transformed target column of the training set."""
        pass

    @property
    @abstractmethod
    def y_test_transformed(self):
        """Transformed target column of the test set."""
        pass

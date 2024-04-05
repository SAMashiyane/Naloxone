import gc
import logging
import os
import random
import secrets
import traceback
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import patch

import numpy as np
import pandas as pd
import plotly.express as px
import scikitplot as skplt
from IPython.display import display as ipython_display
from joblib.memory import Memory
from packaging import version
from pandas.io.formats.style import Styler
from sklearn.model_selection import BaseCrossValidator  # type: ignore
from sklearn.pipeline import Pipeline

import naloxlib.inward.add.sklearn
import naloxlib.inward.add.yellowbrick
import naloxlib.inward.preprocess
from naloxlib.inward.display import CommonDisplay
from naloxlib.inward.logging import create_logger, get_logger, redirect_output
from naloxlib.inward.pipeline import Pipeline as InternalPipeline
from naloxlib.inward.plots.helper import MatplotlibDefaultDPI
from naloxlib.inward.plots.yellowbrick import show_yellowbrick_plot
from naloxlib.inward.naloxlib_experiment.naloxlib_experiment import _naloxlibExperiment
from naloxlib.inward.validation import is_sklearn_cv_generator


from naloxlib.efficacy.depend import _check_soft_dependencies
from naloxlib.efficacy.depend import (
    MLUsecase,
    get_allowed_engines,
    get_label_encoder,
    get_model_name,
)

# LOGGER = get_logger()


class _TabularExperiment(_naloxlibExperiment):
    def __init__(self) -> None:
        super().__init__()
        self.all_allowed_engines = None
        self.fold_shuffle_param = False
        self.fold_groups_param = None
        self.exp_model_engines = {}
        self._variable_keys = self._variable_keys.union(
            {
                "_ml_usecase",
                "_available_plots",
                "USI",
                "html_param",
                "seed",
                "pipeline",
                "n_jobs_param",
                "gpu_n_jobs_param",
                "exp_name_log",
                "exp_id",
                "logging_param",
                "log_plots_param",
                "data",
                "idx",
                "gpu_param",
                "memory",
            }
        )
        return

    def _pack_for_remote(self) -> dict:
        pack = super()._pack_for_remote()
        for k in ["_all_metrics", "seed"]:
            if hasattr(self, k):
                pack[k] = getattr(self, k)
        return pack

    def _get_setup_display(self, **kwargs) -> Styler:
        return pd.DataFrame().style

    def _get_groups(
        self,
        groups,
        data: Optional[pd.DataFrame] = None,
        fold_groups=None,
    ):
        import naloxlib.efficacy.depend

        data = data if data is not None else self.X_train
        fold_groups = fold_groups if fold_groups is not None else self.fold_groups_param
        return naloxlib.efficacy.depend.get_groups(groups, data, fold_groups)

    def _get_cv_splitter(
        self, fold, ml_usecase: Optional[MLUsecase] = None
    ) -> BaseCrossValidator:
        """Returns the cross validator object used to perform cross validation"""
        if not ml_usecase:
            ml_usecase = self._ml_usecase

        import naloxlib.efficacy.depend

        return naloxlib.efficacy.depend.get_cv_splitter(
            fold,
            default=self.fold_generator,
            seed=self.seed,
            shuffle=self.fold_shuffle_param,
            int_default="stratifiedkfold"
            if ml_usecase == MLUsecase.CLASSIFICATION
            else "kfold",
        )

    def _is_unsupervised(self) -> bool:
        return False

    def _get_model_id(self, e, models=None) -> str:

        if models is None:
            models = self._all_models_internal

        return naloxlib.efficacy.depend.get_model_id(e, models)

    def _get_metric_by_name_or_id(self, name_or_id: str, metrics: Optional[Any] = None):

        if metrics is None:
            metrics = self._all_metrics
        metric = None
        try:
            metric = metrics[name_or_id]
            return metric
        except Exception:
            pass

        try:
            metric = next(
                v for k, v in metrics.items() if name_or_id in (v.display_name, v.name)
            )
            return metric
        except Exception:
            pass

        return metric

    def _get_model_name(self, e, deep: bool = True, models=None) -> str:
        """
        Get model name.
        """
        if models is None:
            models = getattr(self, "_all_models_internal", None)

        return get_model_name(e, models, deep=deep)

    def _log_model(
        self,
        model,
        model_results,
        score_dict: dict,
        source: str,
        runtime: float,
        model_fit_time: float,
        pipeline,
        log_holdout: bool = True,
        log_plots: Optional[List[str]] = None,
        tune_cv_results=None,
        URI=None,
        experiment_custom_tags=None,
        display: Optional[CommonDisplay] = None,
    ):
        log_plots = log_plots or []
        try:
            self.logging_param.log_model(
                experiment=self,
                model=model,
                model_results=model_results,
                pipeline=pipeline,
                score_dict=score_dict,
                source=source,
                runtime=runtime,
                model_fit_time=model_fit_time,
                log_plots=log_plots,
                experiment_custom_tags=experiment_custom_tags,
                log_holdout=log_holdout,
                tune_cv_results=tune_cv_results,
                URI=URI,
                display=display,
            )
        except Exception:
            self.logger.error(
                f"_log_model() for {model} raised an exception:\n"
                f"{traceback.format_exc()}"
            )

    def _profile(self, profile, profile_kwargs):
        """Create a profile report"""
        if profile:
            profile_kwargs = profile_kwargs or {}

            if self.verbose:
                print("Loading profile... Please Wait!")
            try:
                import ydata_profiling

                self.report = ydata_profiling.ProfileReport(self.data, **profile_kwargs)
            except Exception as ex:
                print("Profiler Failed. No output to show, continue with modeling.")
                self.logger.error(
                    f"Data Failed with exception:\n {ex}\n"
                    "No output to show, continue with modeling."
                )

        return self

    def _validate_log_experiment(self, obj: Any) -> None:
        return isinstance(obj, (bool, None)) or (
            isinstance(obj, str)
            and obj.lower() in ["mlflow", "wandb", "dagshub", "comet_ml"]
        )

    def _convert_log_experiment(
        self, log_experiment: Any
    ) -> Union[bool, None]:
        if not (
            (
                isinstance(log_experiment, list)
                and all(self._validate_log_experiment(x) for x in log_experiment)
            )
            or self._validate_log_experiment(log_experiment)
        ):
            raise TypeError(
                "log_experiment parameter must be a bool, BaseLogger, one of 'mlflow', 'wandb', 'dagshub', 'comet_ml'; or a list of the former."
            )

            if loggers_list:
                return naloxlib.loggers.DashboardLogger(loggers_list)
        return False

    def _initialize_setup(
        self,
        n_jobs: Optional[int] = -1,
        use_gpu: bool = False,
        html: bool = True,
        session_id: Optional[int] = None,
        system_log: Union[bool, str, logging.Logger] = True,
        log_experiment: Union[
            bool, str, List[Union[str, None]]
        ] = False,
        experiment_name: Optional[str] = None,
        memory: Union[bool, str, Memory] = True,
        verbose: bool = True,
    ):


        # Parameter attrs
        self.n_jobs_param = n_jobs
        self.gpu_param = use_gpu
        self.html_param = html
        self.logging_param = self._convert_log_experiment(log_experiment)
        # self.memory = get_memory(memory)
        self.verbose = verbose

        # Global attrs
        self.USI = secrets.token_hex(nbytes=2)
        self.seed = int(random.randint(150, 9000) if session_id is None else session_id)
        np.random.seed(self.seed)

        # Initialization =========================================== >>

        if experiment_name:
            if not isinstance(experiment_name, str):
                raise TypeError(
                    "The experiment_name parameter must be a non-empty str if not None."
                )
            self.exp_name_log = experiment_name

        self.logger = create_logger(system_log)
        self.logger.info(f"naloxlib {type(self).__name__}")
        self.logger.info(f"Logging name: {self.exp_name_log}")
        self.logger.info(f"ML Usecase: {self._ml_usecase}")
        self.logger.info("3.1.0")
        self.logger.info("Initializing setup()")
        self.logger.info(f"self.USI: {self.USI}")

        self.logger.info(f"self._variable_keys: {self._variable_keys}")

        self._check_environment()

        # Set up GPU usage ========================================= >>

        if self.gpu_param != "force" and type(self.gpu_param) is not bool:
            raise TypeError(
                f"Invalid value for the use_gpu parameter, got {self.gpu_param}. "
                "Possible values are: 'force', True or False."
            )

        cuml_version = None
        if self.gpu_param:
            self.logger.info("Set up GPU usage.")

            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                # from cuml import __version__

                # cuml_version = __version__
                self.logger.info(f"cuml=={cuml_version}")

                try:
                    import cuml.internals.memory_utils

                    cuml.internals.memory_utils.set_global_output_type("numpy")
                except Exception:
                    self.logger.exception("Couldn't set cuML global output type")

            if cuml_version is None or not version.parse(cuml_version) >= version.parse(
                "23.08"
            ):
                message = """cuML is outdated or not found. Required version is >=23.08.
                Please visit https://rapids.ai/install for installation instructions."""
                if use_gpu == "force":
                    raise ImportError(message)
                else:
                    self.logger.warning(message)

        return self



    def _plot_machine(
        self,
        estimator,
        plot: str = "auc",
        scale: float = 1,  # added in naloxlib==2.1.0
        save: Union[str, bool] = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        feature_name: Optional[str] = None,
        label: bool = False,
        verbose: bool = True,
        system: bool = True,
        display: Optional[CommonDisplay] = None,  # added in naloxlib==2.2.0
        display_format: Optional[str] = None,
    ) -> str:
        """Internal version of ``plot_machine`` with ``system`` arg."""
        self._check_setup_ran()

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing plot_machine()")
        self.logger.info(f"plot_machine({function_params_str})")

        self.logger.info("Checking exceptions")

        if not fit_kwargs:
            fit_kwargs = {}

        if not hasattr(estimator, "fit"):
            raise ValueError(
                f"Estimator {estimator} does not have the required fit() method."
            )

        if plot not in self._available_plots:
            raise ValueError(
                "Plot Not Available. Please see docstring for list of available Plots."
            )





        # checking for auc plot
        if not hasattr(estimator, "predict_proba") and plot == "auc":
            raise TypeError(
                "AUC plot not available for estimators with no predict_proba attribute."
            )

        # checking for calibration plot
        if not hasattr(estimator, "predict_proba") and plot == "calibration":
            raise TypeError(
                "Calibration plot not available for estimators with no predict_proba attribute."
            )

        def is_tree(e):
            from sklearn.ensemble._forest import BaseForest
            from sklearn.tree import BaseDecisionTree

            if "final_estimator" in e.get_params():
                e = e.final_estimator
            if "base_estimator" in e.get_params():
                e = e.base_estimator
            if isinstance(e, BaseForest) or isinstance(e, BaseDecisionTree):
                return True

        # checking for calibration plot
        if plot == "tree" and not is_tree(estimator):
            raise TypeError(
                "Decision Tree plot is only available for scikit-learn Decision Trees and Forests, Ensemble models using those or Stacked models using those as meta (final) estimators."
            )

        # checking for feature plot
        if not (
            hasattr(estimator, "coef_") or hasattr(estimator, "feature_importances_")
        ) and (plot == "feature" or plot == "feature_all" or plot == "rfe"):
            raise TypeError(
                "Feature Importance and RFE plots not available for estimators that doesnt support coef_ or feature_importances_ attribute."
            )

        # checking fold parameter
        if fold is not None and not (
            type(fold) is int or is_sklearn_cv_generator(fold)
        ):
            raise TypeError(
                "fold parameter must be either None,."
            )

        if type(label) is not bool:
            raise TypeError("Label parameter only accepts True or False.")

        if feature_name is not None and type(feature_name) is not str:
            raise TypeError(
                "feature parameter must be string ."
            )



        cv = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        if not display:
            display = CommonDisplay(verbose=verbose, html_param=self.html_param)

        plot_kwargs = plot_kwargs or {}

        self.logger.info("Preloading libraries")

        import matplotlib.pyplot as plt

        np.random.seed(self.seed)


        if isinstance(estimator, InternalPipeline):
            estimator = estimator.steps[-1][1]
        estimator = deepcopy(estimator)
        model = estimator

        self.logger.info("Copying training dataset")

        self.logger.info(f"Plot type: {plot}")
        plot_name = self._available_plots[plot]


        model_name = self._get_model_name(model)
        base_plot_filename = f"{plot_name}.png"
        with patch(
            "yellowbrick.utils.types.is_estimator",
            naloxlib.inward.add.yellowbrick.is_estimator,
        ):
            with patch(
                "yellowbrick.utils.helpers.is_estimator",
                naloxlib.inward.add.yellowbrick.is_estimator,
            ):
                _base_dpi = 100

                def pipeline():
                    from schemdraw import Drawing
                    from schemdraw.flow import Arrow, Data, RoundBox, Subroutine

                    # Create schematic drawing
                    d = Drawing(backend="matplotlib")
                    d.config(fontsize=plot_kwargs.get("fontsize", 14))
                    d += Subroutine(w=10, h=5, s=1).label("Raw data").drop("E")
                    for est in self.pipeline:
                        name = getattr(est, "transformer", est).__class__.__name__
                        d += Arrow().right()
                        d += RoundBox(w=max(len(name), 7), h=5, cornerradius=1).label(
                            name
                        )

                    # Add the model box
                    name = estimator.__class__.__name__
                    d += Arrow().right()
                    d += Data(w=max(len(name), 7), h=5).label(name)

                    display.clear_output()

                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        fig, ax = plt.subplots(
                            figsize=((2 + len(self.pipeline) * 5), 6)
                        )

                        d.draw(ax=ax, showframe=False, show=False)
                        ax.set_aspect("equal")
                        plt.axis("off")
                        plt.tight_layout()

                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")



                    # display.clear_output()
                    if system:
                        pass
                        # resplots.show()

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        # resplots.write_html(plot_filename)

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def cluster():
                    self.logger.info(
                        "SubProcess assign_model() called =================================="
                    )
                    b = self.assign_model(  # type: ignore
                        estimator, verbose=False, transformation=True
                    ).reset_index(drop=True)
                    self.logger.info(
                        "SubProcess assign_model() end =================================="
                    )
                    cluster = b["Cluster"].values
                    b.drop("Cluster", axis=1, inplace=True)
                    b = pd.get_dummies(b)  # casting categorical variable

                    from sklearn.decomposition import PCA

                    pca = PCA(n_components=2, random_state=self.seed)
                    self.logger.info("Fitting PCA()")
                    pca_ = pca.fit_transform(b)
                    pca_ = pd.DataFrame(pca_)
                    pca_ = pca_.rename(columns={0: "PCA1", 1: "PCA2"})
                    pca_["Cluster"] = cluster

                    if feature_name is not None:
                        pca_["Feature"] = self.data[feature_name]
                    else:
                        pca_["Feature"] = self.data[self.data.columns[0]]

                    if label:
                        pca_["Label"] = pca_["Feature"]

                    """
                    sorting
                    """

                    self.logger.info("Sorting dataframe")

                    clus_num = [int(i.split()[1]) for i in pca_["Cluster"]]

                    pca_["cnum"] = clus_num
                    pca_.sort_values(by="cnum", inplace=True)



                    # display.clear_output()

                    self.logger.info("Rendering Visual")

                    if label:
                        fig = px.scatter(
                            pca_,
                            x="PCA1",
                            y="PCA2",
                            text="Label",
                            color="Cluster",
                            opacity=0.5,
                        )
                    else:
                        fig = px.scatter(
                            pca_,
                            x="PCA1",
                            y="PCA2",
                            hover_data=["Feature"],
                            color="Cluster",
                            opacity=0.5,
                        )

                    fig.update_traces(textposition="top center")
                    fig.update_layout(plot_bgcolor="rgb(240,240,240)")

                    fig.update_layout(
                        height=600 * scale, title_text="2D Cluster PCA Plot"
                    )

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)



                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def umap():
                    self.logger.info(
                        "SubProcess assign_model() called =================================="
                    )
                    b = self.assign_model(  # type: ignore
                        model, verbose=False, transformation=True, score=False
                    ).reset_index(drop=True)
                    self.logger.info(
                        "SubProcess assign_model() end =================================="
                    )

                    label = pd.DataFrame(b["Anomaly"])
                    b.dropna(axis=0, inplace=True)  # droping rows with NA's
                    b.drop(["Anomaly"], axis=1, inplace=True)

                    _check_soft_dependencies(
                        "umap",
                        extra="analysis",
                        severity="error",
                        install_name="umap-learn",
                    )
                    import umap

                    reducer = umap.UMAP()
                    self.logger.info("Fitting UMAP()")
                    embedding = reducer.fit_transform(b)
                    X = pd.DataFrame(embedding)

                    import plotly.express as px

                    df = X
                    df["Anomaly"] = label

                    if feature_name is not None:
                        df["Feature"] = self.data[feature_name]
                    else:
                        df["Feature"] = self.data[self.data.columns[0]]

                    # display.clear_output()

                    self.logger.info("Rendering Visual")

                    fig = px.scatter(
                        df,
                        x=0,
                        y=1,
                        color="Anomaly",
                        title="uMAP Plot for Outliers",
                        hover_data=["Feature"],
                        opacity=0.7,
                        width=900 * scale,
                        height=800 * scale,
                    )

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)



                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def tsne():
                    if self._ml_usecase == MLUsecase.CLUSTERING:
                        return _tsne_clustering()
                    else:
                        return _tsne_anomaly()

                def _tsne_anomaly():
                    self.logger.info(
                        "SubProcess assign_model() called =================================="
                    )
                    b = self.assign_model(  # type: ignore
                        model, verbose=False, transformation=True, score=False
                    ).reset_index(drop=True)
                    self.logger.info(
                        "SubProcess assign_model() end =================================="
                    )
                    cluster = b["Anomaly"].values
                    b.dropna(axis=0, inplace=True)  # droping rows with NA's
                    b.drop("Anomaly", axis=1, inplace=True)

                    self.logger.info("Getting dummies to cast categorical variables")

                    from sklearn.manifold import TSNE

                    self.logger.info("Fitting TSNE()")
                    X_embedded = TSNE(n_components=3).fit_transform(b)

                    X = pd.DataFrame(X_embedded)
                    X["Anomaly"] = cluster
                    if feature_name is not None:
                        X["Feature"] = self.data[feature_name]
                    else:
                        X["Feature"] = self.data[self.data.columns[0]]

                    df = X

                    # display.clear_output()

                    self.logger.info("Rendering Visual")

                    if label:
                        fig = px.scatter_3d(
                            df,
                            x=0,
                            y=1,
                            z=2,
                            text="Feature",
                            color="Anomaly",
                            title="3d TSNE Plot for Outliers",
                            opacity=0.7,
                            width=900 * scale,
                            height=800 * scale,
                        )
                    else:
                        fig = px.scatter_3d(
                            df,
                            x=0,
                            y=1,
                            z=2,
                            hover_data=["Feature"],
                            color="Anomaly",
                            title="3d TSNE Plot for Outliers",
                            opacity=0.7,
                            width=900 * scale,
                            height=800 * scale,
                        )

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)


                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def _tsne_clustering():
                    self.logger.info(
                        "SubProcess assign_model() called =================================="
                    )
                    b = self.assign_model(  # type: ignore
                        estimator,
                        verbose=False,
                        score=False,
                        transformation=True,
                    ).reset_index(drop=True)
                    self.logger.info(
                        "SubProcess assign_model() end =================================="
                    )

                    cluster = b["Cluster"].values
                    b.drop("Cluster", axis=1, inplace=True)

                    from sklearn.manifold import TSNE

                    self.logger.info("Fitting TSNE()")
                    X_embedded = TSNE(
                        n_components=3, random_state=self.seed
                    ).fit_transform(b)
                    X_embedded = pd.DataFrame(X_embedded)
                    X_embedded["Cluster"] = cluster

                    if feature_name is not None:
                        X_embedded["Feature"] = self.data[feature_name]
                    else:
                        X_embedded["Feature"] = self.data[self.data.columns[0]]

                    if label:
                        X_embedded["Label"] = X_embedded["Feature"]

                    """
                    sorting
                    """
                    self.logger.info("Sorting dataframe")

                    clus_num = [int(i.split()[1]) for i in X_embedded["Cluster"]]

                    X_embedded["cnum"] = clus_num
                    X_embedded.sort_values(by="cnum", inplace=True)

                    """
                    sorting ends
                    """

                    df = X_embedded

                    # display.clear_output()

                    self.logger.info("Rendering Visual")

                    if label:
                        fig = px.scatter_3d(
                            df,
                            x=0,
                            y=1,
                            z=2,
                            color="Cluster",
                            title="3d TSNE Plot for Clusters",
                            text="Label",
                            opacity=0.7,
                            width=900 * scale,
                            height=800 * scale,
                        )

                    else:
                        fig = px.scatter_3d(
                            df,
                            x=0,
                            y=1,
                            z=2,
                            color="Cluster",
                            title="3d TSNE Plot for Clusters",
                            hover_data=["Feature"],
                            opacity=0.7,
                            width=900 * scale,
                            height=800 * scale,
                        )

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)



                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def distribution():
                    self.logger.info(
                        "SubProcess assign_model() called =================================="
                    )
                    d = self.assign_model(  # type: ignore
                        estimator, verbose=False
                    ).reset_index(drop=True)
                    self.logger.info(
                        "SubProcess assign_model() end =================================="
                    )

                    """
                    sorting
                    """
                    self.logger.info("Sorting dataframe")

                    clus_num = []
                    for i in d.Cluster:
                        a = int(i.split()[1])
                        clus_num.append(a)

                    d["cnum"] = clus_num
                    d.sort_values(by="cnum", inplace=True)
                    d.reset_index(inplace=True, drop=True)

                    clus_label = []
                    for i in d.cnum:
                        a = "Cluster " + str(i)
                        clus_label.append(a)

                    d.drop(["Cluster", "cnum"], inplace=True, axis=1)
                    d["Cluster"] = clus_label

                    """
                    sorting ends
                    """

                    if feature_name is None:
                        x_col = "Cluster"
                    else:
                        x_col = feature_name

                    # display.clear_output()

                    self.logger.info("Rendering Visual")

                    fig = px.histogram(
                        d,
                        x=x_col,
                        color="Cluster",
                        marginal="box",
                        opacity=0.7,
                        hover_data=d.columns,
                    )

                    fig.update_layout(
                        height=600 * scale,
                    )

                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        fig.write_html(plot_filename)



                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def elbow():
                    try:
                        from yellowbrick.cluster import KElbowVisualizer

                        visualizer = KElbowVisualizer(
                            estimator, timings=False, **plot_kwargs
                        )
                        return show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=self.X_train_transformed,
                            y_train=None,
                            X_test=None,
                            y_test=None,
                            name=plot_name,
                            handle_test="",
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            display_format=display_format,
                        )

                    except Exception:
                        self.logger.error("Elbow plot failed. Exception:")
                        self.logger.error(traceback.format_exc())
                        raise TypeError("Plot Type not supported for this model.")

                def silhouette():
                    from yellowbrick.cluster import SilhouetteVisualizer

                    try:
                        visualizer = SilhouetteVisualizer(
                            estimator, colors="yellowbrick", **plot_kwargs
                        )
                        return show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=self.X_train_transformed,
                            y_train=None,
                            X_test=None,
                            y_test=None,
                            name=plot_name,
                            handle_test="",
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            display_format=display_format,
                        )
                    except Exception:
                        self.logger.error("Silhouette plot failed. Exception:")
                        self.logger.error(traceback.format_exc())
                        raise TypeError("Plot Type not supported for this model.")

                def distance():
                    from yellowbrick.cluster import InterclusterDistance

                    try:
                        visualizer = InterclusterDistance(estimator, **plot_kwargs)
                        return show_yellowbrick_plot(
                            visualizer=visualizer,
                            X_train=self.X_train_transformed,
                            y_train=None,
                            X_test=None,
                            y_test=None,
                            name=plot_name,
                            handle_test="",
                            scale=scale,
                            save=save,
                            fit_kwargs=fit_kwargs,
                            display_format=display_format,
                        )
                    except Exception:
                        self.logger.error("Distance plot failed. Exception:")
                        self.logger.error(traceback.format_exc())
                        raise TypeError("Plot Type not supported for this model.")

                def residuals():
                    from yellowbrick.regressor import ResidualsPlot

                    visualizer = ResidualsPlot(estimator, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def auc():
                    from yellowbrick.classifier import ROCAUC

                    visualizer = ROCAUC(estimator, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def threshold():
                    from yellowbrick.classifier import DiscriminationThreshold

                    visualizer = DiscriminationThreshold(
                        estimator, random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def pr():
                    from yellowbrick.classifier import PrecisionRecallCurve

                    visualizer = PrecisionRecallCurve(
                        estimator, random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def confusion_matrix():
                    from yellowbrick.classifier import ConfusionMatrix

                    plot_kwargs.setdefault("fontsize", 15)
                    plot_kwargs.setdefault("cmap", "Greens")

                    visualizer = ConfusionMatrix(
                        estimator, random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def error():
                    if self._ml_usecase == MLUsecase.CLASSIFICATION:
                        from yellowbrick.classifier import ClassPredictionError

                        visualizer = ClassPredictionError(
                            estimator, random_state=self.seed, **plot_kwargs
                        )

                    elif self._ml_usecase == MLUsecase.REGRESSION:
                        from yellowbrick.regressor import PredictionError

                        visualizer = PredictionError(
                            estimator, random_state=self.seed, **plot_kwargs
                        )

                    return show_yellowbrick_plot(
                        visualizer=visualizer,  # type: ignore
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def cooks():
                    from yellowbrick.regressor import CooksDistance

                    visualizer = CooksDistance()
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        handle_test="",
                        display_format=display_format,
                    )

                def class_report():
                    from yellowbrick.classifier import ClassificationReport

                    visualizer = ClassificationReport(
                        estimator, random_state=self.seed, support=True, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def boundary():
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    from yellowbrick.contrib.classifier import DecisionViz

                    data_X_transformed = self.X_train_transformed.select_dtypes(
                        include="number"
                    )
                    test_X_transformed = self.X_test_transformed.select_dtypes(
                        include="number"
                    )
                    self.logger.info("Fitting StandardScaler()")
                    data_X_transformed = StandardScaler().fit_transform(
                        data_X_transformed
                    )
                    test_X_transformed = StandardScaler().fit_transform(
                        test_X_transformed
                    )
                    pca = PCA(n_components=2, random_state=self.seed)
                    self.logger.info("Fitting PCA()")
                    data_X_transformed = pca.fit_transform(data_X_transformed)
                    test_X_transformed = pca.fit_transform(test_X_transformed)

                    viz_ = DecisionViz(estimator, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=viz_,
                        X_train=data_X_transformed,
                        y_train=np.array(self.y_train_transformed),
                        X_test=test_X_transformed,
                        y_test=np.array(self.y_test_transformed),
                        name=plot_name,
                        scale=scale,
                        handle_test="draw",
                        save=save,
                        fit_kwargs=fit_kwargs,
                        features=["Feature One", "Feature Two"],
                        classes=["A", "B"],
                        display_format=display_format,
                    )

                def rfe():
                    from yellowbrick.model_selection import RFECV

                    visualizer = RFECV(estimator, cv=cv, groups=groups, **plot_kwargs)
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def learning():
                    from yellowbrick.model_selection import LearningCurve

                    sizes = np.linspace(0.3, 1.0, 10)
                    visualizer = LearningCurve(
                        estimator,
                        cv=cv,
                        train_sizes=sizes,
                        groups=groups,
                        n_jobs=self.gpu_n_jobs_param,
                        random_state=self.seed,
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def lift():
                    self.logger.info("Generating predictions / predict_proba on X_test")
                    y_test__ = self.y_test_transformed
                    predict_proba__ = estimator.predict_proba(self.X_test_transformed)
                    # display.clear_output()
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        skplt.metrics.plot_lift_curve(
                            y_test__, predict_proba__, figsize=(10, 6)
                        )
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def gain():
                    self.logger.info("Generating predictions / predict_proba on X_test")
                    y_test__ = self.y_test_transformed
                    predict_proba__ = estimator.predict_proba(self.X_test_transformed)
                    # display.clear_output()
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        skplt.metrics.plot_cumulative_gain(
                            y_test__, predict_proba__, figsize=(10, 6)
                        )
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def manifold():
                    from yellowbrick.features import Manifold

                    data_X_transformed = self.X_train_transformed.select_dtypes(
                        include="number"
                    )
                    visualizer = Manifold(
                        manifold="tsne", random_state=self.seed, **plot_kwargs
                    )
                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=data_X_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_train="fit_transform",
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def tree():
                    from sklearn.tree import plot_tree

                    is_stacked_model = False
                    is_ensemble_of_forests = False

                    if isinstance(estimator, Pipeline):
                        fitted_estimator = estimator._final_estimator
                    else:
                        fitted_estimator = estimator

                    if "final_estimator" in fitted_estimator.get_params():
                        tree_estimator = fitted_estimator.final_estimator
                        is_stacked_model = True
                    else:
                        tree_estimator = fitted_estimator

                    if (
                        "base_estimator" in tree_estimator.get_params()
                        and "n_estimators" in tree_estimator.base_estimator.get_params()
                    ):
                        n_estimators = (
                            tree_estimator.get_params()["n_estimators"]
                            * tree_estimator.base_estimator.get_params()["n_estimators"]
                        )
                        is_ensemble_of_forests = True
                    elif "n_estimators" in tree_estimator.get_params():
                        n_estimators = tree_estimator.get_params()["n_estimators"]
                    else:
                        n_estimators = 1
                    if n_estimators > 10:
                        rows = (n_estimators // 10) + 1
                        cols = 10
                    else:
                        rows = 1
                        cols = n_estimators
                    figsize = (cols * 20, rows * 16)
                    fig, axes = plt.subplots(
                        nrows=rows,
                        ncols=cols,
                        figsize=figsize,
                        dpi=_base_dpi * scale,
                        squeeze=False,
                    )
                    axes = list(axes.flatten())

                    fig.suptitle("Decision Trees")

                    self.logger.info("Plotting decision trees")
                    trees = []
                    feature_names = list(self.X_train_transformed.columns)
                    class_names = None
                    if self._ml_usecase == MLUsecase.CLASSIFICATION:
                        label_encoder = get_label_encoder(self.pipeline)
                        if label_encoder:
                            class_names = {
                                i: class_name
                                for i, class_name in enumerate(label_encoder.classes_)
                            }
                    fitted_estimator = tree_estimator
                    if is_stacked_model:
                        stacked_feature_names = []
                        if self._ml_usecase == MLUsecase.CLASSIFICATION:
                            classes = list(self.y_train_transformed.unique())
                            if len(classes) == 2:
                                classes.pop()
                            for c in classes:
                                stacked_feature_names.extend(
                                    [
                                        f"{k}_{class_names[c]}"
                                        for k, v in fitted_estimator.estimators
                                    ]
                                )
                        else:
                            stacked_feature_names.extend(
                                [f"{k}" for k, v in fitted_estimator.estimators]
                            )
                        if not fitted_estimator.passthrough:
                            feature_names = stacked_feature_names
                        else:
                            feature_names = stacked_feature_names + feature_names
                        fitted_estimator = fitted_estimator.final_estimator_
                    if is_ensemble_of_forests:
                        for tree_estimator in fitted_estimator.estimators_:
                            trees.extend(tree_estimator.estimators_)
                    else:
                        try:
                            trees = fitted_estimator.estimators_
                        except Exception:
                            trees = [fitted_estimator]
                    if self._ml_usecase == MLUsecase.CLASSIFICATION:
                        class_names = list(class_names.values())
                    for i, tree in enumerate(trees):
                        self.logger.info(f"Plotting tree {i}")
                        plot_tree(
                            tree,
                            feature_names=feature_names,
                            class_names=class_names,
                            filled=True,
                            rounded=True,
                            precision=4,
                            ax=axes[i],
                        )
                        axes[i].set_title(f"Tree {i}")
                    for i in range(len(trees), len(axes)):
                        axes[i].set_visible(False)

                    # display.clear_output()
                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def calibration():
                    from sklearn.calibration import calibration_curve

                    plt.figure(figsize=(7, 6), dpi=_base_dpi * scale)
                    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

                    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
                    self.logger.info("Scoring test/hold-out set")
                    prob_pos = estimator.predict_proba(self.X_test_transformed)[:, 1]
                    prob_pos = (prob_pos - prob_pos.min()) / (
                        prob_pos.max() - prob_pos.min()
                    )
                    (
                        fraction_of_positives,
                        mean_predicted_value,
                    ) = calibration_curve(self.y_test_transformed, prob_pos, n_bins=10)
                    ax1.plot(
                        mean_predicted_value,
                        fraction_of_positives,
                        "s-",
                        label=f"{model_name}",
                    )

                    ax1.set_ylabel("Fraction of positives")
                    ax1.set_ylim([0, 1])
                    ax1.set_xlim([0, 1])
                    ax1.legend(loc="lower right")
                    ax1.set_title("Calibration plots (reliability curve)")
                    ax1.set_facecolor("white")
                    ax1.grid(True, color="grey", linewidth=0.5, linestyle="-")
                    plt.tight_layout()
                    # display.clear_output()
                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def vc():
                    self.logger.info("Determining param_name")

                    try:
                        try:
                            # catboost special case
                            model_params = estimator.get_all_params()
                        except Exception:
                            model_params = estimator.get_params()
                    except Exception:
                        # display.clear_output()
                        self.logger.error("VC plot failed. Exception:")
                        self.logger.error(traceback.format_exc())
                        raise TypeError(
                            "Plot not supported for this estimator. Try different estimator."
                        )

                    param_name = ""
                    param_range = None

                    if self._ml_usecase == MLUsecase.CLASSIFICATION:
                        # Catboost
                        if "depth" in model_params:
                            param_name = "depth"
                            param_range = np.arange(1, 8 if self.gpu_param else 11)

                        # SGD Classifier
                        elif "l1_ratio" in model_params:
                            param_name = "l1_ratio"
                            param_range = np.arange(0, 1, 0.01)

                        # tree based models
                        elif "max_depth" in model_params:
                            param_name = "max_depth"
                            param_range = np.arange(1, 11)

                        # knn
                        elif "n_neighbors" in model_params:
                            param_name = "n_neighbors"
                            param_range = np.arange(1, 11)

                        # MLP / Ridge
                        elif "alpha" in model_params:
                            param_name = "alpha"
                            param_range = np.arange(0, 1, 0.1)

                        # Logistic Regression
                        elif "C" in model_params:
                            param_name = "C"
                            param_range = np.arange(1, 11)

                        # Bagging / Boosting
                        elif "n_estimators" in model_params:
                            param_name = "n_estimators"
                            param_range = np.arange(1, 1000, 10)

                        # Naive Bayes
                        elif "var_smoothing" in model_params:
                            param_name = "var_smoothing"
                            param_range = np.arange(0.1, 1, 0.01)

                        # QDA
                        elif "reg_param" in model_params:
                            param_name = "reg_param"
                            param_range = np.arange(0, 1, 0.1)

                        # GPC
                        elif "max_iter_predict" in model_params:
                            param_name = "max_iter_predict"
                            param_range = np.arange(100, 1000, 100)

                        else:
                            # display.clear_output()
                            raise TypeError(
                                "Plot not supported for this estimator. Try different estimator."
                            )

                    elif self._ml_usecase == MLUsecase.REGRESSION:
                        # Catboost
                        if "depth" in model_params:
                            param_name = "depth"
                            param_range = np.arange(1, 8 if self.gpu_param else 11)

                        # lasso/ridge/en/llar/huber/kr/mlp/br/ard
                        elif "alpha" in model_params:
                            param_name = "alpha"
                            param_range = np.arange(0, 1, 0.1)

                        elif "alpha_1" in model_params:
                            param_name = "alpha_1"
                            param_range = np.arange(0, 1, 0.1)

                        # par/svm
                        elif "C" in model_params:
                            param_name = "C"
                            param_range = np.arange(1, 11)

                        # tree based models (dt/rf/et)
                        elif "max_depth" in model_params:
                            param_name = "max_depth"
                            param_range = np.arange(1, 11)

                        # knn
                        elif "n_neighbors" in model_params:
                            param_name = "n_neighbors"
                            param_range = np.arange(1, 11)

                        # Bagging / Boosting (ada/gbr)
                        elif "n_estimators" in model_params:
                            param_name = "n_estimators"
                            param_range = np.arange(1, 1000, 10)

                        # Bagging / Boosting (ada/gbr)
                        elif "n_nonzero_coefs" in model_params:
                            param_name = "n_nonzero_coefs"
                            if len(self.X_train_transformed.columns) >= 10:
                                param_max = 11
                            else:
                                param_max = len(self.X_train_transformed.columns) + 1
                            param_range = np.arange(1, param_max, 1)

                        elif "eps" in model_params:
                            param_name = "eps"
                            param_range = np.arange(0, 1, 0.1)

                        elif "max_subpopulation" in model_params:
                            param_name = "max_subpopulation"
                            param_range = np.arange(1000, 100000, 2000)

                        elif "min_samples" in model_params:
                            param_name = "min_samples"
                            param_range = np.arange(0.01, 1, 0.1)

                        else:
                            # display.clear_output()
                            raise TypeError(
                                "Plot not supported for this estimator. Try different estimator."
                            )

                    self.logger.info(f"param_name: {param_name}")

                    from yellowbrick.model_selection import ValidationCurve

                    viz = ValidationCurve(
                        estimator,
                        param_name=param_name,
                        param_range=param_range,
                        cv=cv,
                        groups=groups,
                        random_state=self.seed,
                        n_jobs=self.gpu_n_jobs_param,
                    )
                    return show_yellowbrick_plot(
                        visualizer=viz,
                        X_train=self.X_train_transformed,
                        y_train=self.y_train_transformed,
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_train="fit",
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def dimension():
                    from sklearn.decomposition import PCA
                    from sklearn.preprocessing import StandardScaler
                    from yellowbrick.features import RadViz

                    data_X_transformed = self.X_train_transformed.select_dtypes(
                        include="number"
                    )
                    self.logger.info("Fitting StandardScaler()")
                    data_X_transformed = StandardScaler().fit_transform(
                        data_X_transformed
                    )

                    features = min(
                        round(len(self.X_train_transformed.columns) * 0.3, 0), 5
                    )
                    features = int(features)

                    pca = PCA(n_components=features, random_state=self.seed)
                    self.logger.info("Fitting PCA()")
                    data_X_transformed = pca.fit_transform(data_X_transformed)
                    classes = self.y_train_transformed.unique().tolist()
                    visualizer = RadViz(classes=classes, alpha=0.25, **plot_kwargs)

                    return show_yellowbrick_plot(
                        visualizer=visualizer,
                        X_train=data_X_transformed,
                        y_train=np.array(self.y_train_transformed),
                        X_test=self.X_test_transformed,
                        y_test=self.y_test_transformed,
                        handle_train="fit_transform",
                        handle_test="",
                        name=plot_name,
                        scale=scale,
                        save=save,
                        fit_kwargs=fit_kwargs,
                        display_format=display_format,
                    )

                def feature():
                    return _feature(10)

                def feature_all():
                    return _feature(len(self.X_train_transformed.columns))

                def _feature(n: int):
                    variables = None
                    temp_model = estimator
                    if hasattr(estimator, "steps"):
                        temp_model = estimator.steps[-1][1]
                    if hasattr(temp_model, "coef_"):
                        try:
                            coef = temp_model.coef_.flatten()
                            if len(coef) > len(self.X_train_transformed.columns):
                                coef = coef[: len(self.X_train_transformed.columns)]
                            variables = abs(coef)
                        except Exception:
                            pass
                    if variables is None:
                        self.logger.warning(
                            "No coef_ found. Trying feature_importances_"
                        )
                        variables = abs(temp_model.feature_importances_)
                    coef_df = pd.DataFrame(
                        {
                            "Variable": self.X_train_transformed.columns,
                            "Value": variables,
                        }
                    )
                    sorted_df = (
                        coef_df.sort_values(by="Value", ascending=False)
                        .head(n)
                        .sort_values(by="Value")
                    )
                    my_range = range(1, len(sorted_df.index) + 1)
                    plt.figure(figsize=(8, 5 * (n // 10)), dpi=_base_dpi * scale)
                    plt.hlines(
                        y=my_range,
                        xmin=0,
                        xmax=sorted_df["Value"],
                        color="skyblue",
                    )
                    plt.plot(sorted_df["Value"], my_range, "o")
                    plt.yticks(my_range, sorted_df["Variable"])
                    plt.title("Feature Importance Plot")
                    plt.xlabel("Variable Importance")
                    plt.ylabel("Features")
                    # display.clear_output()
                    plot_filename = None
                    if save:
                        if not isinstance(save, bool):
                            plot_filename = os.path.join(save, base_plot_filename)
                        else:
                            plot_filename = base_plot_filename
                        self.logger.info(f"Saving '{plot_filename}'")
                        plt.savefig(plot_filename, bbox_inches="tight")
                    elif system:
                        plt.show()
                    plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                def parameter():
                    try:
                        params = estimator.get_all_params()
                    except Exception:
                        params = estimator.get_params(deep=False)

                    param_df = pd.DataFrame.from_dict(
                        {str(k): str(v) for k, v in params.items()},
                        orient="index",
                        columns=["Parameters"],
                    )
                    # use ipython directly to show it in the widget
                    ipython_display(param_df)
                    self.logger.info("Visual Rendered Successfully")

                def ks():
                    self.logger.info("Generating predictions / predict_proba on X_test")
                    predict_proba__ = estimator.predict_proba(self.X_train_transformed)
                    # display.clear_output()
                    with MatplotlibDefaultDPI(base_dpi=_base_dpi, scale_to_set=scale):
                        skplt.metrics.plot_ks_statistic(
                            self.y_train_transformed, predict_proba__, figsize=(10, 6)
                        )
                        plot_filename = None
                        if save:
                            if not isinstance(save, bool):
                                plot_filename = os.path.join(save, base_plot_filename)
                            else:
                                plot_filename = base_plot_filename
                            self.logger.info(f"Saving '{plot_filename}'")
                            plt.savefig(plot_filename, bbox_inches="tight")
                        elif system:
                            plt.show()
                        plt.close()

                    self.logger.info("Visual Rendered Successfully")
                    return plot_filename

                # execute the plot method
                with redirect_output(self.logger):
                    ret = locals()[plot]()
                if ret:
                    plot_filename = ret
                else:
                    plot_filename = base_plot_filename

                try:
                    plt.close()
                except Exception:
                    pass

        gc.collect()

        self.logger.info(
            "plot_machine() successfully completed......................................"
        )

        if save:
            return plot_filename

    def plot_machine(
        self,
        estimator,
        plot: str = "auc",
        scale: float = 1,  # added in naloxlib==2.1.0
        save: Union[str, bool] = False,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        groups: Optional[Union[str, Any]] = None,
        feature_name: Optional[str] = None,
        label: bool = False,
        verbose: bool = True,
        display_format: Optional[str] = None,
    ) -> Optional[str]:

        return self._plot_machine(
            estimator=estimator,
            plot=plot,
            scale=scale,
            save=save,
            fold=fold,
            fit_kwargs=fit_kwargs,
            plot_kwargs=plot_kwargs,
            groups=groups,
            feature_name=feature_name,
            label=label,
            verbose=verbose,
            display_format=display_format,
        )

    def evaluate_model(
        self,
        estimator,
        fold: Optional[Union[int, Any]] = None,
        fit_kwargs: Optional[dict] = None,
        plot_kwargs: Optional[dict] = None,
        feature_name: Optional[str] = None,
        groups: Optional[Union[str, Any]] = None,
    ):

        function_params_str = ", ".join([f"{k}={v}" for k, v in locals().items()])

        self.logger.info("Initializing evaluate_model()")
        self.logger.info(f"evaluate_model({function_params_str})")

        from ipywidgets import widgets
        from ipywidgets.widgets import fixed, interact

        if not fit_kwargs:
            fit_kwargs = {}

        a = widgets.ToggleButtons(
            options=[(v, k) for k, v in self._available_plots.items()],
            description="Plot Type:",
            disabled=False,
            button_style="",  # 'success', 'info', 'warning', 'danger' or ''
            icons=[""],
        )

        fold = self._get_cv_splitter(fold)

        groups = self._get_groups(groups)

        interact(
            self._plot_machine,
            estimator=fixed(estimator),
            plot=a,
            save=fixed(False),
            verbose=fixed(False),
            scale=fixed(1),
            fold=fixed(fold),
            fit_kwargs=fixed(fit_kwargs),
            plot_kwargs=fixed(plot_kwargs),
            feature_name=fixed(feature_name),
            label=fixed(False),
            groups=fixed(groups),
            system=fixed(True),
            display=fixed(None),
            display_format=fixed(None),
        )

    def predict_model(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    # def finalize_model(self) -> None:
    #     return

    def _get_models(self, raise_errors: bool = True) -> Tuple[dict, dict]:
        return ({}, {})

    def _get_metrics(self, raise_errors: bool = True) -> dict:
        return {}

    def models(
        self,
        type: Optional[str] = None,
        internal: bool = False,
        raise_errors: bool = True,
    ) -> pd.DataFrame:

        self.logger.info(f"gpu_param set to {self.gpu_param}")

        _, model_containers = self._get_models(raise_errors)

        rows = [
            v.get_dict(internal)
            for k, v in model_containers.items()
            if (internal or not v.is_special)
        ]

        df = pd.DataFrame(rows)
        df.set_index("ID", inplace=True, drop=True)

        return df

    def _set_all_models(self) -> "_TabularExperiment":

        self._all_models, self._all_models_internal = self._get_models()
        return self

    def get_allowed_engines(self, estimator: str) -> Optional[List[str]]:

        allowed_engines = get_allowed_engines(
            estimator=estimator, all_allowed_engines=self.all_allowed_engines
        )
        return allowed_engines

    def get_engine(self, estimator: str) -> Optional[str]:

        engine = self.exp_model_engines.get(estimator, None)
        if engine is None:
            msg = (
                f"Engine for model '{estimator}' has not been set explicitly, "
                "hence returning None."
            )
            self.logger.info(msg)

        return engine

    def _set_engine(self, estimator: str, engine: str, severity: str = "error"):

        if severity not in ("error", "warning"):
            raise ValueError(
                "Error in calling set_engine, severity "
                f'argument must be "error" or "warning", got "{severity}".'
            )

        allowed_engines = self.get_allowed_engines(estimator=estimator)
        if allowed_engines is None:
            msg = (
                f"Either model '{estimator}' has only 1 engine and hence can not be changed, "
                "or the model is not in the allowed list of models for this setup."
            )

            if severity == "error":
                raise ValueError(msg)
            elif severity == "warning":
                self.logger.warning(msg)
                print(msg)

        elif engine not in allowed_engines:
            msg = (
                f"Engine '{engine}' for estimator '{estimator}' is not allowed."
                f" Allowed values are: {', '.join(allowed_engines)}."
            )

            if severity == "error":
                raise ValueError(msg)
            elif severity == "warning":
                self.logger.warning(msg)
                print(msg)

        else:
            self.exp_model_engines[estimator] = engine
            self.logger.info(
                f"Engine successfully changes for model '{estimator}' to '{engine}'."
            )

        # Need to do this, else internal class variables are not reset with new engine.
        self._set_all_models()

    def _set_exp_model_engines(
        self,
        container_default_engines: Dict[str, str],
        engine: Optional[Dict[str, str]] = None,
    ) -> "_TabularExperiment":


        engine = engine or {}
        for key in container_default_engines:
            # If provided by user, then use that, else get from the defaults
            eng = engine.get(key, container_default_engines.get(key))
            self._set_engine(estimator=key, engine=eng, severity="error")

        return self

    def _set_all_metrics(self) -> "_TabularExperiment":
        self._all_metrics = self._get_metrics()
        return self

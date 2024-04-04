# Author: Salio(Mohtarami)
# https://github.com/SAMashiyane
# Date: 2023-2024

import logging
from typing import  Union
import numpy as np
from packaging import version
import naloxlib.box.base_box

from naloxlib.inward.distributions import (
    Distribution,
    IntUniformDistribution,
    UniformDistribution,
)
from naloxlib.efficacy.depend import _check_soft_dependencies
from naloxlib.efficacy.depend import (
    get_class_name,
    np_list_arange,
    param_grid_to_lists,
)
from typing import Any, Dict, List, Optional
from naloxlib.box.base_box import BaseContainer
from naloxlib.inward.distributions import CategoricalDistribution
from naloxlib.inward.naloxlib_experiment.naloxlib_experiment import _naloxlibExperiment
from naloxlib.efficacy.depend import get_allowed_engines, get_logger


class ModelContainer(BaseContainer):


    def __init__(
        self,
        id: str,
        name: str,
        class_def: type,
        eq_function: Optional[type] = None,
        args: Dict[str, Any] = None,
        is_special: bool = False,
    ) -> None:
        super().__init__(id=id, name=name, class_def=class_def, args=args)
        self.reference = self.get_class_name()
        if not eq_function:
            eq_function = lambda x: isinstance(x, self.class_def)
        self.eq_function = eq_function
        self.is_special = is_special
        self.logger = get_logger()
        self.allowed_engines = None
        self.default_engine = None

    def is_estimator_equal(self, estimator):
        return self.eq_function(estimator)

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:

        d = [("ID", self.id), ("Name", self.name), ("Reference", self.reference)]

        if internal:
            d += [
                ("Special", self.is_special),
                ("Class", self.class_def),
                ("Equality", self.eq_function),
                ("Args", self.args),
            ]

        return dict(d)

    def _set_engine(
        self,
        id: str,
        experiment: _naloxlibExperiment,
        severity: str = "error",
    ):

        engine_to_use = experiment.get_engine(id)
        # If not specified, use the default engine
        if engine_to_use is None:
            engine_to_use = self.default_engine

        if engine_to_use is not None and engine_to_use not in self.allowed_engines:
            msg = (
                f"Engine '{engine_to_use}' for estimator '{id}' is not allowed.\n"
                f"Allowed values are '{self.allowed_engines}'."
            )

            if severity == "error":
                raise ValueError(msg)
            elif severity == "warning":
                self.logger.warning(msg)
                print(msg)
            else:
                raise ValueError(
                    "Error in calling set_engine, severity "
                    f'argument must be "error" or "warning", got "{severity}".'
                )

        self.engine = engine_to_use

    def _set_engine_related_vars(
        self,
        id: str,
        all_allowed_engines: Dict[str, List[str]],
        experiment: _naloxlibExperiment,
    ):

        self.allowed_engines = get_allowed_engines(
            estimator=id, all_allowed_engines=all_allowed_engines
        )
        self.default_engine = self.allowed_engines[0]
        self._set_engine(
            id=id,
            experiment=experiment,
            severity="error",
        )

def leftover_parameters_to_categorical_distributions(
    tune_grid: dict, tune_distributions: dict
) -> None:

    for k, v in tune_grid.items():
        if k not in tune_distributions:
            tune_distributions[k] = CategoricalDistribution(v)

#------------------------------start model classification by Salio---------------------

ALL_ALLOWED_ENGINES: Dict[str, List[str]] = {
    "lr": ["sklearn", "sklearnex"],
    "knn": ["sklearn", "sklearnex"],
    "rbfsvm": ["sklearn", "sklearnex"],
}


def get_container_default_engines() -> Dict[str, str]:
    default_engines = {}
    for id, all_engines in ALL_ALLOWED_ENGINES.items():
        default_engines[id] = all_engines[0]
    return default_engines


class ClassifierContainer(ModelContainer):

    def __init__(
            self,
            id: str,
            name: str,
            class_def: type,
            is_turbo: bool = True,
            eq_function: Optional[type] = None,
            args: Dict[str, Any] = None,
            is_special: bool = False,
            tune_grid: Dict[str, list] = None,
            tune_distribution: Dict[str, Distribution] = None,
            tune_args: Dict[str, Any] = None,
            shap: Union[bool, str] = False,
            is_gpu_enabled: Optional[bool] = None,
            is_boosting_supported: Optional[bool] = None,
            is_soft_voting_supported: Optional[bool] = None,
            tunable: Optional[type] = None,
    ) -> None:
        self.shap = shap
        if not (isinstance(shap, bool) or shap in ["type1", "type2"]):
            raise ValueError("shap must be either bool or 'type1', 'type2'.")

        if not args:
            args = {}

        if not tune_grid:
            tune_grid = {}

        if not tune_distribution:
            tune_distribution = {}

        if not tune_args:
            tune_args = {}

        super().__init__(
            id=id,
            name=name,
            class_def=class_def,
            eq_function=eq_function,
            args=args,
            is_special=is_special,
        )
        self.is_turbo = is_turbo
        self.tune_grid = param_grid_to_lists(tune_grid)
        self.tune_distribution = tune_distribution
        self.tune_args = tune_args
        self.tunable = tunable

        try:
            model_instance = class_def()

            self.is_boosting_supported = bool(
                hasattr(model_instance, "class_weights")
                or hasattr(model_instance, "predict_proba")
            )

            self.is_soft_voting_supported = bool(
                hasattr(model_instance, "predict_proba")
            )

            del model_instance
        except Exception:
            self.is_boosting_supported = False
            self.is_soft_voting_supported = False
        finally:
            if is_boosting_supported is not None:
                self.is_boosting_supported = is_boosting_supported
            if is_soft_voting_supported is not None:
                self.is_soft_voting_supported = is_soft_voting_supported

        if is_gpu_enabled is not None:
            self.is_gpu_enabled = is_gpu_enabled
        else:
            self.is_gpu_enabled = bool(self.get_package_name() == "cuml")

    def get_dict(self, internal: bool = True) -> Dict[str, Any]:

        d = [
            ("ID", self.id),
            ("Name", self.name),
            ("Reference", self.reference),
            ("Turbo", self.is_turbo),
        ]

        if internal:
            d += [
                ("Special", self.is_special),
                ("Class", self.class_def),
                ("Equality", self.eq_function),
                ("Args", self.args),
                ("Tune Grid", self.tune_grid),
                ("Tune Distributions", self.tune_distribution),
                ("Tune Args", self.tune_args),
                ("SHAP", self.shap),
                ("GPU Enabled", self.is_gpu_enabled),
                ("Boosting Supported", self.is_boosting_supported),
                ("Soft Voting", self.is_soft_voting_supported),
                ("Tunable Class", self.tunable),
            ]

        return dict(d)


class LogisticRegressionClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        id = "lr"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if self.engine == "sklearn":
            from sklearn.linear_model import LogisticRegression
        elif self.engine == "sklearnex":
            if _check_soft_dependencies("sklearnex", extra=None, severity="warning"):
                from sklearnex.linear_model import LogisticRegression
            else:
                from sklearn.linear_model import LogisticRegression

        if experiment.gpu_param == "force":
            from cuml.linear_model import LogisticRegression

            logger.info("Imported cuml.linear_model.LogisticRegression")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.linear_model import LogisticRegression

                logger.info("Imported cuml.linear_model.LogisticRegression")
                gpu_imported = True

        args = {"max_iter": 1000}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # common
        tune_grid["C"] = np_list_arange(0.001, 10, 0.001, inclusive=True)

        if gpu_imported:
            tune_grid["penalty"] = ["l2", "l1"]
        else:
            args["random_state"] = experiment.seed

            tune_grid["class_weight"] = ["balanced", {}]

        tune_distributions["C"] = UniformDistribution(0.001, 10)
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id=id,
            name="Logistic Regression",
            class_def=LogisticRegression,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class KNeighborsClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        id = "knn"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if self.engine == "sklearn":
            from sklearn.neighbors import KNeighborsClassifier
        elif self.engine == "sklearnex":
            if _check_soft_dependencies("sklearnex", extra=None, severity="warning"):
                from sklearnex.neighbors import KNeighborsClassifier
            else:
                from sklearn.neighbors import KNeighborsClassifier

        if experiment.gpu_param == "force":
            from cuml.neighbors import KNeighborsClassifier

            logger.info("Imported cuml.neighbors.KNeighborsClassifier")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.neighbors import KNeighborsClassifier

                logger.info("Imported cuml.neighbors.KNeighborsClassifier")
                gpu_imported = True

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        # common
        tune_grid["n_neighbors"] = range(1, 51)
        tune_grid["weights"] = ["uniform"]
        tune_grid["metric"] = ["minkowski", "euclidean", "manhattan"]

        if not gpu_imported:
            args["n_jobs"] = experiment.n_jobs_param
            tune_grid["weights"] += ["distance"]

        tune_distributions["n_neighbors"] = IntUniformDistribution(1, 51)
        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id=id,
            name="K Neighbors Classifier",
            class_def=KNeighborsClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class GaussianNBClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.naive_bayes import GaussianNB

        args = {}
        tune_args = {}
        tune_grid = {
            "var_smoothing": [
                0.000000001,
                0.000000002,
                0.000000005,
                0.000000008,
                0.000000009,
                0.0000001,
                0.0000002,
                0.0000003,
                0.0000005,
                0.0000007,
                0.0000009,
                0.00001,
                0.001,
                0.002,
                0.003,
                0.004,
                0.005,
                0.007,
                0.009,
                0.004,
                0.005,
                0.006,
                0.007,
                0.008,
                0.009,
                0.01,
                0.1,
                1,
            ]
        }
        tune_distributions = {
            "var_smoothing": UniformDistribution(0.000000001, 1, log=True)
        }

        super().__init__(
            id="nb",
            name="Naive Bayes",
            class_def=GaussianNB,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class DecisionTreeClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.tree import DecisionTreeClassifier

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "max_depth": np_list_arange(1, 16, 1, inclusive=True),
            "max_features": [1.0, "sqrt", "log2"],
            "min_samples_leaf": [2, 3, 4, 5, 6],
            "min_samples_split": [2, 5, 7, 9, 10],
            "criterion": ["gini", "entropy"],
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
        }
        tune_distributions = {
            "max_depth": IntUniformDistribution(1, 16),
            "max_features": UniformDistribution(0.4, 1),
            "min_samples_leaf": IntUniformDistribution(2, 6),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="dt",
            name="Decision Tree Classifier",
            class_def=DecisionTreeClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
        )


class SGDClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        from sklearn.linear_model import SGDClassifier

        if experiment.gpu_param == "force":
            from cuml import MBSGDClassifier as SGDClassifier

            logger.info("Imported cuml.MBSGDClassifier")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml import MBSGDClassifier as SGDClassifier

                logger.info("Imported cuml.MBSGDClassifier")
                gpu_imported = True

        args = {"tol": 0.001, "loss": "hinge", "penalty": "l2", "eta0": 0.001}
        tune_args = {}
        tune_grid = {
            "penalty": ["elasticnet", "l2", "l1"],
            "l1_ratio": np_list_arange(0.0000000001, 1, 0.01, inclusive=False),
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "fit_intercept": [True, False],
            "learning_rate": ["constant", "invscaling", "adaptive", "optimal"],
            "eta0": [0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
        }
        tune_distributions = {
            "l1_ratio": UniformDistribution(0.0000000001, 0.9999999999),
            "alpha": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "eta0": UniformDistribution(0.001, 0.5, log=True),
        }

        if gpu_imported:
            tune_grid["learning_rate"].remove("optimal")
            batch_size = [
                (512, 50000),
                (256, 25000),
                (128, 10000),
                (64, 5000),
                (32, 1000),
                (16, 0),
            ]
            for arg, x_len in batch_size:
                if len(experiment.X_train) >= x_len:
                    args["batch_size"] = arg
                    break
        else:
            args["random_state"] = experiment.seed
            args["n_jobs"] = experiment.n_jobs_param

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="svm",
            name="SVM - Linear Kernel",
            class_def=SGDClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class SVCClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        id = "rbfsvm"
        self._set_engine_related_vars(
            id=id, all_allowed_engines=ALL_ALLOWED_ENGINES, experiment=experiment
        )

        if self.engine == "sklearn":
            from sklearn.svm import SVC
        elif self.engine == "sklearnex":
            if _check_soft_dependencies("sklearnex", extra=None, severity="warning"):
                from sklearnex.svm import SVC
            else:
                from sklearn.svm import SVC

        if experiment.gpu_param == "force":
            from cuml.svm import SVC

            logger.info("Imported cuml.svm.SVC")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                from cuml.svm import SVC

                logger.info("Imported cuml.svm.SVC")
                gpu_imported = True

        args = {
            "gamma": "auto",
            "C": 1.0,
            "probability": True,
            "kernel": "rbf",
            "random_state": experiment.seed,
        }
        tune_args = {}
        tune_grid = {
            "C": np_list_arange(0, 50, 0.01, inclusive=True),
            "class_weight": ["balanced", {}],
        }
        tune_distributions = {
            "C": UniformDistribution(0, 50),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        if gpu_imported:
            SVC = get_svc_classifier()

        super().__init__(
            id=id,
            name="SVM - Radial Kernel",
            class_def=SVC,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_turbo=False,
        )


class GaussianProcessClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.gaussian_process import GaussianProcessClassifier

        args = {
            "copy_X_train": False,
            "random_state": experiment.seed,
            "n_jobs": experiment.n_jobs_param,
        }
        tune_args = {}
        tune_grid = {
            "max_iter_predict": [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        }
        tune_distributions = {"max_iter_predict": IntUniformDistribution(100, 1000)}

        super().__init__(
            id="gpc",
            name="Gaussian Process Classifier",
            class_def=GaussianProcessClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_turbo=False,
        )


class MLPClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.neural_network import MLPClassifier

        # from naloxlib.inward.tunable import TunableMLPClassifier

        args = {"random_state": experiment.seed, "max_iter": 500}
        tune_args = {}
        tune_grid = {
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                0.9,
            ],
            "hidden_layer_size_0": [50, 100],
            "hidden_layer_size_1": [0, 50, 100],
            "hidden_layer_size_2": [0, 50, 100],
            "activation": ["tanh", "identity", "logistic", "relu"],
        }
        tune_distributions = {
            "alpha": UniformDistribution(0.0000000001, 0.9999999999, log=True),
            "hidden_layer_size_0": IntUniformDistribution(50, 100),
            "hidden_layer_size_1": IntUniformDistribution(0, 100),
            "hidden_layer_size_2": IntUniformDistribution(0, 100),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="mlp",
            name="MLP Classifier",
            class_def=MLPClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_turbo=False,
            # tunable=TunableMLPClassifier,
        )


class RidgeClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        from sklearn.linear_model import RidgeClassifier

        if experiment.gpu_param == "force":
            import cuml.linear_model

            logger.info("Imported cuml.linear_model")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                import cuml.linear_model

                logger.info("Imported cuml.linear_model")
                gpu_imported = True

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        if gpu_imported:
            RidgeClassifier = naloxlib.inward.cuml_wrappers.get_ridge_classifier()
        else:
            args = {"random_state": experiment.seed}

        tune_grid = {}
        tune_grid["alpha"] = np_list_arange(0.01, 10, 0.01, inclusive=False)
        tune_grid["fit_intercept"] = [True, False]
        tune_distributions["alpha"] = UniformDistribution(0.001, 10)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ridge",
            name="Ridge Classifier",
            class_def=RidgeClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_gpu_enabled=gpu_imported,
        )
        if gpu_imported:
            self.reference = get_class_name(cuml.linear_model.Ridge)


class RandomForestClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        gpu_imported = False

        from sklearn.ensemble import RandomForestClassifier

        if experiment.gpu_param == "force":
            import cuml.ensemble

            logger.info("Imported cuml.ensemble")
            gpu_imported = True
        elif experiment.gpu_param:
            if _check_soft_dependencies("cuml", extra=None, severity="warning"):
                import cuml.ensemble

                logger.info("Imported cuml.ensemble")
                gpu_imported = True

        if gpu_imported:
            RandomForestClassifier = (
                naloxlib.inward.cuml_wrappers.get_random_forest_classifier()
            )

        if not gpu_imported:
            args = {
                "random_state": experiment.seed,
                "n_jobs": experiment.n_jobs_param,
            }
        else:
            import cuml

            if version.parse(cuml.__version__) >= version.parse("0.19"):
                args = {"random_state": experiment.seed}
            else:
                args = {"seed": experiment.seed}

        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "max_depth": IntUniformDistribution(1, 11),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
            "max_features": UniformDistribution(0.4, 1),
        }

        if gpu_imported:
            tune_grid["split_criterion"] = [0, 1]
        else:
            tune_grid["criterion"] = ["gini", "entropy"]
            tune_grid["class_weight"] = ["balanced", "balanced_subsample", {}]
            tune_grid["min_samples_split"] = [2, 5, 7, 9, 10]
            tune_grid["min_samples_leaf"] = [2, 3, 4, 5, 6]
            tune_distributions["min_samples_split"] = IntUniformDistribution(2, 10)
            tune_distributions["min_samples_leaf"] = IntUniformDistribution(2, 6)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="rf",
            name="Random Forest Classifier",
            class_def=RandomForestClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
            is_gpu_enabled=gpu_imported,
        )
        if gpu_imported:
            self.reference = get_class_name(cuml.ensemble.RandomForestClassifier)


class QuadraticDiscriminantAnalysisContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

        args = {}
        tune_args = {}
        tune_grid = {"reg_param": np_list_arange(0, 1, 0.01, inclusive=True)}
        tune_distributions = {"reg_param": UniformDistribution(0, 1)}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="qda",
            name="Quadratic Discriminant Analysis",
            class_def=QuadraticDiscriminantAnalysis,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class AdaBoostClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.ensemble import AdaBoostClassifier

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "algorithm": ["SAMME"],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="ada",
            name="Ada Boost Classifier",
            class_def=AdaBoostClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class GradientBoostingClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.ensemble import GradientBoostingClassifier

        args = {"random_state": experiment.seed}
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "subsample": np_list_arange(0.2, 1, 0.05, inclusive=True),
            "min_samples_split": [2, 4, 5, 7, 9, 10],
            "min_samples_leaf": [1, 2, 3, 4, 5],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "subsample": UniformDistribution(0.2, 1),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_samples_leaf": IntUniformDistribution(1, 5),
            "max_depth": IntUniformDistribution(1, 11),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
            "max_features": UniformDistribution(0.4, 1),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="gbc",
            name="Gradient Boosting Classifier",
            class_def=GradientBoostingClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class LinearDiscriminantAnalysisContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        args = {}
        tune_args = {}
        tune_grid = {
            "solver": ["lsqr", "eigen"],
            "shrinkage": [
                "empirical",
                "auto",
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
                1,
            ],
        }
        tune_distributions = {
            "shrinkage": UniformDistribution(0.0001, 1, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="lda",
            name="Linear Discriminant Analysis",
            class_def=LinearDiscriminantAnalysis,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )


class ExtraTreesClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.ensemble import ExtraTreesClassifier

        args = {
            "random_state": experiment.seed,
            "n_jobs": experiment.n_jobs_param,
        }
        tune_args = {}
        tune_grid = {
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "criterion": ["gini", "entropy"],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "min_impurity_decrease": [
                0,
                0.0001,
                0.001,
                0.01,
                0.0002,
                0.002,
                0.02,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "max_features": [1.0, "sqrt", "log2"],
            "bootstrap": [True, False],
            "min_samples_split": [2, 5, 7, 9, 10],
            "min_samples_leaf": [2, 3, 4, 5, 6],
            "class_weight": ["balanced", "balanced_subsample", {}],
        }
        tune_distributions = {
            "n_estimators": IntUniformDistribution(10, 300),
            "max_depth": IntUniformDistribution(1, 11),
            "min_samples_split": IntUniformDistribution(2, 10),
            "min_samples_leaf": IntUniformDistribution(1, 5),
            "max_features": UniformDistribution(0.4, 1),
            "min_impurity_decrease": UniformDistribution(0.000000001, 0.5, log=True),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="et",
            name="Extra Trees Classifier",
            class_def=ExtraTreesClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
        )


class XGBClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        if _check_soft_dependencies("xgboost", extra="models", severity="warning"):
            import xgboost
        else:
            self.active = False
            return

        if version.parse(xgboost.__version__) < version.parse("1.1.0"):
            logger.warning(
                f"Wrong xgboost version. Expected xgboost>=1.1.0, got xgboost=={xgboost.__version__}"
            )
            self.active = False
            return

        from xgboost import XGBClassifier

        args = {
            "random_state": experiment.seed,
            "n_jobs": experiment.n_jobs_param,
            "verbosity": 0,
            "booster": "gbtree",
        }

        # If using XGBoost version 2.0 or higher
        if version.parse(xgboost.__version__) >= version.parse("2.0.0"):
            args["tree_method"] = "hist" if experiment.gpu_param else "auto"
            args["device"] = "gpu" if experiment.gpu_param else "cpu"
        else:
            args["tree_method"] = "gpu_hist" if experiment.gpu_param else "auto"

        tune_args = {}
        tune_grid = {
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "subsample": [0.2, 0.3, 0.5, 0.7, 0.9, 1],
            "max_depth": np_list_arange(1, 11, 1, inclusive=True),
            "colsample_bytree": [0.5, 0.7, 0.9, 1],
            "min_child_weight": [1, 2, 3, 4],
            "reg_alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "reg_lambda": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "scale_pos_weight": np_list_arange(0, 50, 0.1, inclusive=True),
        }
        tune_distributions = {
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "n_estimators": IntUniformDistribution(10, 300),
            "subsample": UniformDistribution(0.2, 1),
            "max_depth": IntUniformDistribution(1, 11),
            "colsample_bytree": UniformDistribution(0.5, 1),
            "min_child_weight": IntUniformDistribution(1, 4),
            "reg_alpha": UniformDistribution(0.0000000001, 10, log=True),
            "reg_lambda": UniformDistribution(0.0000000001, 10, log=True),
            "scale_pos_weight": UniformDistribution(1, 50),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="xgboost",
            name="Extreme Gradient Boosting",
            class_def=XGBClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type2",
            is_gpu_enabled=bool(experiment.gpu_param),
        )


class LGBMClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from lightgbm import LGBMClassifier
        from lightgbm.basic import LightGBMError

        args = {
            "random_state": experiment.seed,
            "n_jobs": experiment.n_jobs_param,
        }
        tune_args = {}
        tune_grid = {
            "num_leaves": [
                2,
                4,
                6,
                8,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                150,
                200,
                256,
            ],
            "learning_rate": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "min_split_gain": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "reg_alpha": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "reg_lambda": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
                0.7,
                1,
                2,
                3,
                4,
                5,
                10,
            ],
            "feature_fraction": np_list_arange(0.4, 1, 0.1, inclusive=True),
            "bagging_fraction": np_list_arange(0.4, 1, 0.1, inclusive=True),
            "bagging_freq": [0, 1, 2, 3, 4, 5, 6, 7],
            "min_child_samples": np_list_arange(1, 100, 5, inclusive=True),
        }
        tune_distributions = {
            "num_leaves": IntUniformDistribution(2, 256),
            "learning_rate": UniformDistribution(0.000001, 0.5, log=True),
            "n_estimators": IntUniformDistribution(10, 300),
            "min_split_gain": UniformDistribution(0, 1),
            "reg_alpha": UniformDistribution(0.0000000001, 10, log=True),
            "reg_lambda": UniformDistribution(0.0000000001, 10, log=True),
            "feature_fraction": UniformDistribution(0.4, 1),
            "bagging_fraction": UniformDistribution(0.4, 1),
            "bagging_freq": IntUniformDistribution(0, 7),
            "min_child_samples": IntUniformDistribution(1, 100),
        }

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        is_gpu_enabled = False
        if experiment.gpu_param:
            try:
                lgb = LGBMClassifier(device="gpu")
                lgb.fit(np.zeros((2, 2)), [0, 1])
                is_gpu_enabled = "gpu"
                del lgb
            except Exception:
                try:
                    lgb = LGBMClassifier(device="cuda")
                    lgb.fit(np.zeros((2, 2)), [0, 1])
                    is_gpu_enabled = "cuda"
                    del lgb
                except LightGBMError:
                    is_gpu_enabled = False
                    if experiment.gpu_param == "force":
                        raise RuntimeError(
                            "LightGBM GPU mode not available. Consult https://lightgbm.readthedocs.io/en/latest/GPU-Tutorial.html."
                        )

        if is_gpu_enabled == "gpu":
            args["device"] = "gpu"
        elif is_gpu_enabled == "cuda":
            args["device"] = "cuda"

        super().__init__(
            id="lightgbm",
            name="Light Gradient Boosting Machine",
            class_def=LGBMClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type1",
            is_gpu_enabled=is_gpu_enabled,
        )


class CatBoostClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        logger = get_logger()
        np.random.seed(experiment.seed)
        if _check_soft_dependencies("catboost", extra="models", severity="warning"):
            import catboost
        else:
            self.active = False
            return

        if version.parse(catboost.__version__) < version.parse("0.23.2"):
            logger.warning(
                f"Wrong catboost version. Expected catboost>=0.23.2, got catboost=={catboost.__version__}"
            )
            self.active = False
            return

        from catboost import CatBoostClassifier

        # suppress output
        logging.getLogger("catboost").setLevel(logging.ERROR)

        use_gpu = experiment.gpu_param

        args = {
            "random_state": experiment.seed,
            "verbose": False,
            "thread_count": experiment.n_jobs_param,
            "task_type": "GPU" if use_gpu else "CPU",
            "border_count": 32 if use_gpu else 254,
        }
        tune_args = {}
        tune_grid = {
            "eta": [
                0.0000001,
                0.000001,
                0.0001,
                0.001,
                0.01,
                0.0005,
                0.005,
                0.05,
                0.1,
                0.15,
                0.2,
                0.3,
                0.4,
                0.5,
            ],
            "depth": list(range(1, 12)),
            "n_estimators": np_list_arange(10, 300, 10, inclusive=True),
            "random_strength": np_list_arange(0, 0.8, 0.1, inclusive=True),
            "l2_leaf_reg": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100, 200],
        }
        tune_distributions = {
            "eta": UniformDistribution(0.000001, 0.5, log=True),
            "depth": IntUniformDistribution(1, 11),
            "n_estimators": IntUniformDistribution(10, 300),
            "random_strength": UniformDistribution(0, 0.8),
            "l2_leaf_reg": IntUniformDistribution(1, 200, log=True),
        }

        if use_gpu:
            tune_grid["depth"] = list(range(1, 9))
            tune_distributions["depth"] = IntUniformDistribution(1, 8)

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="catboost",
            name="CatBoost Classifier",
            class_def=CatBoostClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap="type2",
            is_gpu_enabled=use_gpu,
        )


class DummyClassifierContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.dummy import DummyClassifier

        args = {"strategy": "prior", "random_state": experiment.seed}

        tune_args = {}
        tune_grid = {"strategy": ["most_frequent", "prior", "stratified", "uniform"]}

        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="dummy",
            name="Dummy Classifier",
            class_def=DummyClassifier,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
        )



class CalibratedClassifierCVContainer(ClassifierContainer):
    def __init__(self, experiment):
        get_logger()
        np.random.seed(experiment.seed)
        from sklearn.calibration import CalibratedClassifierCV

        args = {}
        tune_args = {}
        tune_grid = {}
        tune_distributions = {}

        leftover_parameters_to_categorical_distributions(tune_grid, tune_distributions)

        super().__init__(
            id="CalibratedCV",
            name="Calibrated Classifier CV",
            class_def=CalibratedClassifierCV,
            args=args,
            tune_grid=tune_grid,
            tune_distribution=tune_distributions,
            tune_args=tune_args,
            shap=False,
            is_special=True,
            is_gpu_enabled=False,
        )


def get_all_model_containers(
        experiment: Any, raise_errors: bool = True
) -> Dict[str, ClassifierContainer]:
    return naloxlib.box.base_box.get_all_containers(
        globals(), experiment, ClassifierContainer, raise_errors
    )

# -------------------------------------- End models by Salio---------------------------

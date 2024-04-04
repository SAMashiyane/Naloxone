
import inspect
import os
from collections import defaultdict
from typing import  BinaryIO, Callable, Optional, Union
import cloudpickle
import pandas as pd
from naloxlib.efficacy.depend import DATAFRAME_LIKE
from naloxlib.efficacy.depend import LazyExperimentMapping

# LOGGER = get_logger()


class _naloxlibExperiment:

    _attributes_to_not_save = ["data", "test_data", "data_func"]

    def __init__(self) -> None:
        self._ml_usecase = None
        self._available_plots = {}
        self._variable_keys = set()
        self.exp_id = None
        self.gpu_param = False
        self.n_jobs_param = -1
        self._master_model_container = []


        self.data = None
        self.target_param = None
        self.idx = [None, None]


        self.fold_generator = None
        self.pipeline = None
        self._display_container = None
        self._fxs = defaultdict(list)
        self._setup_ran = False
        self._setup_params = None

        self._remote = False

    def _pack_for_remote(self) -> dict:

        return {"_setup_params": self._setup_params, "_remote": True}

    def _unpack_at_remote(self, data: dict) -> None:

        for k, v in data.items():
            setattr(self, k, v)

    def _register_setup_params(self, params: dict) -> None:

        self._setup_params = {
            k: v for k, v in params.items() if k != "self" and v is not None
        }

    @property
    def _property_keys(self) -> set:
        return {
            n
            for n in dir(self)
            if not n.startswith("_")
            and isinstance(getattr(self.__class__, n, None), property)
        }

    @property
    def gpu_n_jobs_param(self) -> int:
        return self.n_jobs_param if not self.gpu_param else 1

    @property
    def variables(self) -> dict:
        return LazyExperimentMapping(self)


    @property
    def variable_and_property_keys(self) -> set:
        return self._variable_keys.union(self._property_keys)

    def _check_environment(self) -> None:


        from platform import machine, platform, python_build, python_version

        self.logger.info(f"python_version: {python_version()}")


    def setup(self, *args, **kwargs) -> None:
        return

    def _check_setup_ran(self):

        if not self._setup_ran:
            raise RuntimeError(
                "This function/method requires the users to run setup() first."
                "\nMore info: https://github.com/SAMashiyane/Naloxone"
            )


    @classmethod
    def _load_experiment(
        cls,
        path_or_file: Union[str, os.PathLike, BinaryIO],
        cloudpickle_kwargs=None,
        preprocess_data: bool = True,
        **kwargs,
    ):
        cloudpickle_kwargs = cloudpickle_kwargs or {}
        try:
            loaded_exp: _naloxlibExperiment = cloudpickle.load(
                path_or_file, **cloudpickle_kwargs
            )
        except TypeError:
            with open(path_or_file, mode="rb") as f:
                loaded_exp: _naloxlibExperiment = cloudpickle.load(
                    f, **cloudpickle_kwargs
                )
        original_state = loaded_exp.__dict__.copy()
        new_params = kwargs
        setup_params = loaded_exp._setup_params or {}
        setup_params = setup_params.copy()
        setup_params.update(
            {
                k: v
                for k, v in new_params.items()
                if k in inspect.signature(cls.setup).parameters
            }
        )

        if preprocess_data and not setup_params.get("data_func", None):
            loaded_exp.setup(
                **setup_params,
            )
        else:
            data = new_params.get("data", None)
            data_func = new_params.get("data_func", None)
            if (data is None and data_func is None) or (
                data is not None and data_func is not None
            ):
                raise ValueError("One and only one of data and data_func must be set")
            for key, value in new_params.items():
                setattr(loaded_exp, key, value)
            original_state["_setup_params"] = setup_params

        loaded_exp.__dict__.update(original_state)
        return loaded_exp

    @classmethod
    def load_experiment(
        cls,
        path_or_file: Union[str, os.PathLike, BinaryIO],
        data: Optional[DATAFRAME_LIKE] = None,
        data_func: Optional[Callable[[], DATAFRAME_LIKE]] = None,
        preprocess_data: bool = True,
        **cloudpickle_kwargs,
    ) -> "_naloxlibExperiment":

        return cls._load_experiment(
            path_or_file,
            cloudpickle_kwargs=cloudpickle_kwargs,
            preprocess_data=preprocess_data,
            data=data,
            data_func=data_func,
        )

    def save_experiment(
        self, path_or_file: Union[str, os.PathLike, BinaryIO], **cloudpickle_kwargs
    ) -> None:

        try:
            cloudpickle.dump(self, path_or_file, **cloudpickle_kwargs)
        except TypeError:
            with open(path_or_file, mode="wb") as f:
                cloudpickle.dump(self, f, **cloudpickle_kwargs)

    def pull(self, pop=False) -> pd.DataFrame:  # added in naloxlib==2.2.0

        return self._display_container.pop(-1) if pop else self._display_container[-1]

    @property
    def dataset(self):

        return self.data[[c for c in self.data.columns if c not in self._fxs["Ignore"]]]

    @property
    def X(self):

        return self.dataset

    @property
    def dataset_transformed(self):

        return self.train_transformed

    @property
    def X_train(self):

        return self.train

    @property
    def train(self):

        return self.dataset

    @property
    def X_train_transformed(self):

        return self.pipeline.transform(self.X_train, filter_train_only=False)

    @property
    def train_transformed(self):

        return self.X_train_transformed

    @property
    def X_transformed(self):

        return self.X_train_transformed

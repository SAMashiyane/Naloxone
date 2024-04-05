# Author: Salio(Mohtarami)
# https://github.com/SAMashiyane
# Date: 2023-2024

from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    SequentialFeatureSelector,
    VarianceThreshold,
    f_classif,
    f_regression,
)

from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.preprocessing import (
    KBinsDiscretizer,
    LabelEncoder,
    MaxAbsScaler,
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from naloxlib.box.models.base_model import (
    get_all_model_containers as get_all_class_model_containers,
)


from naloxlib.inward.preprocess.transformers import TransformerWrapperWithInverse,TransformerWrapper,CleanColumnNames

from naloxlib.efficacy.depend import SEQUENCE
from naloxlib.efficacy.depend import (
    MLUsecase,
    check_features_exist,
    df_shrink_dtypes,
    get_columns_to_stratify_by,
    normalize_custom_transformers,
    to_df,
    to_series,
)


class Preprocessor:


    def _prepare_dataset(self, X, y=None):

        self.logger.info("Set up data.")
        X = to_df(deepcopy(X))
        if len(set(X.columns)) != len(X.columns):
            raise ValueError("Duplicate column names found in X.")
        if isinstance(y, (list, tuple, np.ndarray, pd.Series)):
            if not isinstance(y, pd.Series):
                # Check that y is one-dimensional
                ndim = np.array(y).ndim
                if ndim != 1:
                    raise ValueError(f"y should be one-dimensional, got ndim={ndim}.")
                if len(X) != len(y):
                    raise ValueError(
                        "X and y don't have the same number of rows,"
                        f" got len(X)={len(X)} and len(y)={len(y)}."
                    )

                y = to_series(y, index=X.index)

            elif not X.index.equals(y.index):
                raise ValueError("X and y don't have the same indices!")

        elif isinstance(y, str):
            if y not in X.columns:
                raise ValueError(
                    "Invalid value for the target parameter. "
                    f"Column {y} not found in the data."
                )

            X, y = X.drop(y, axis=1), X[y]

        elif isinstance(y, int):
            X, y = X.drop(X.columns[y], axis=1), X[X.columns[y]]

        else:  # y=None
            return df_shrink_dtypes(X)


        if y.isna().any():
            raise ValueError(
                f"{y.isna().sum()} missing values found in the target column: "
                f"{y.name}. To proceed, remove the respective rows from the data. "
            )

        return df_shrink_dtypes(
            X.merge(y.to_frame(), left_index=True, right_index=True)
        )

    def _set_index(self, df):
        """Assign an index to the dataframe."""
        self.logger.info("Set up index.")

        target = df.columns[-1]

        if getattr(self, "index", True) is True:  # True gets caught by isinstance(int)
            pass
        elif self.index is False:
            df = df.reset_index(drop=True)
        elif isinstance(self.index, int):
            if -df.shape[1] <= self.index <= df.shape[1]:
                df = df.set_index(df.columns[self.index], drop=True)
            else:
                raise ValueError(
                    f"Invalid value for the index parameter. Value {self.index} "
                    f"is out of range for a dataset with {df.shape[1]} columns."
                )
        elif isinstance(self.index, str):
            if self.index in df:
                df = df.set_index(self.index, drop=True)
            else:
                raise ValueError(
                    "Invalid value for the index parameter. "
                    f"Column {self.index} not found in the dataset."
                )

        if df.index.name == target:
            raise ValueError(
                "Invalid value for the index parameter. The index column "
                f"can not be the same as the target column, got {target}."
            )

        if df.index.duplicated().any():
            raise ValueError(
                "Invalid value for the index parameter. There are duplicate indices "

            )

        return df

    def _prepare_train_test(
        self,
        train_size,
        test_data,
        data_split_stratify,
        data_split_shuffle,
    ):

        self.logger.info("Set up train/test split.")

        if test_data is None:
            if isinstance(self.index, SEQUENCE):
                if len(self.index) != len(self.data):
                    raise ValueError(
                        "Invalid value for the index parameter. Length of "
                        f"index ({len(self.index)}) doesn't match that of "
                        f"the dataset ({len(self.data)})."
                    )
                self.data.index = self.index


            train, test = train_test_split(
                self.data,
                train_size=train_size,
                stratify=get_columns_to_stratify_by(
                    self.X, self.y, data_split_stratify
                ),
                random_state=self.seed,
                shuffle=data_split_shuffle,
            )
            self.data = self._set_index(pd.concat([train, test]))
            self.idx = [self.data.index[: len(train)], self.data.index[-len(test) :]]

        else:  # test_data is provided
            test_data = self._prepare_dataset(test_data, self.target_param)

            if isinstance(self.index, SEQUENCE):
                if len(self.index) != len(self.data) + len(test_data):
                    raise ValueError(
                        "Invalid value for the index parameter. Length of "
                        f"index ({len(self.index)}) doesn't match that of "
                        f"the data sets ({len(self.data) + len(test_data)})."
                    )
                self.data.index = self.index[: len(self.data)]
                test_data.index = self.index[-len(test_data) :]

            self.data = self._set_index(pd.concat([self.data, test_data]))
            self.idx = [
                self.data.index[: -len(test_data)],
                self.data.index[-len(test_data) :],
            ]

    def _prepare_column_types(
        self,
        ordinal_features,
        numeric_features,
        categorical_features,
        date_features,
        text_features,
        ignore_features,
        keep_features,
    ):

        self.logger.info("Assigning column types.")

        # Features to be ignored (are not read by self.dataset, self.X, etc...)
        self._fxs["Ignore"] = ignore_features or []

        # Ordinal features
        if ordinal_features:
            check_features_exist(ordinal_features.keys(), self.X)
            self._fxs["Ordinal"] = ordinal_features
        else:
            self._fxs["Ordinal"] = {}

        # Numerical features
        if numeric_features:
            check_features_exist(numeric_features, self.X)
            self._fxs["Numeric"] = numeric_features
        else:
            self._fxs["Numeric"] = [
                col
                for col in self.X.select_dtypes(include="number").columns
                if col not in (categorical_features or [])
            ]

        # Date features
        if date_features:
            check_features_exist(date_features, self.X)
            self._fxs["Date"] = date_features
        else:
            self._fxs["Date"] = list(self.X.select_dtypes(include="datetime").columns)

        # Text features
        if text_features:
            check_features_exist(text_features, self.X)
            self._fxs["Text"] = text_features

        # Categorical features
        if categorical_features:
            check_features_exist(categorical_features, self.X)
            self._fxs["Categorical"] = categorical_features
        else:
            # Default should exclude datetime and text columns
            self._fxs["Categorical"] = [
                col
                for col in self.X.select_dtypes(
                    include=["string", "object", "category"]
                ).columns
                if col not in self._fxs["Date"] + self._fxs["Text"]
            ]

        # Features to keep during all preprocessing
        self._fxs["Keep"] = keep_features or []

    def _prepare_folds(
        self, fold_strategy, fold, fold_shuffle, fold_groups, data_split_shuffle
    ):

        self.logger.info("Set up folding strategy.")
        allowed_fold_strategy = ["kfold", "stratifiedkfold", "groupkfold", "timeseries"]

        if isinstance(fold_strategy, str):
            if fold_strategy == "groupkfold":
                if fold_groups is None or len(fold_groups) == 0:
                    raise ValueError(
                        "Invalid value for the fold_strategy parameter. 'groupkfold' "
                        "requires 'fold_groups' to be a non-empty array-like object."
                    )
            elif fold_strategy not in allowed_fold_strategy:
                raise ValueError(
                    "Invalid value for the fold_strategy parameter. "
                    f"Choose from: {', '.join(allowed_fold_strategy)}."
                )

        if fold_strategy == "timeseries" or isinstance(fold_strategy, TimeSeriesSplit):
            if fold_shuffle or data_split_shuffle:
                raise ValueError(

                    " it can lead to unexpected data split."
                )

        if isinstance(fold_groups, str):
            if fold_groups in self.X.columns:
                if pd.isna(self.X[fold_groups]).any():
                    raise ValueError("The 'fold_groups' column cannot contain NaNs.")
                else:
                    self.fold_groups_param = self.X[fold_groups]
            else:
                raise ValueError(
                    "Invalid value for the fold_groups parameter. "
                    f"Column {fold_groups} is not present in the dataset."
                )

        if fold_strategy == "kfold":
            self.fold_generator = KFold(
                fold,
                shuffle=fold_shuffle,
                random_state=self.seed if fold_shuffle else None,
            )
        elif fold_strategy == "stratifiedkfold":
            self.fold_generator = StratifiedKFold(
                fold,
                shuffle=fold_shuffle,
                random_state=self.seed if fold_shuffle else None,
            )
        elif fold_strategy == "groupkfold":
            self.fold_generator = GroupKFold(fold)
        elif fold_strategy == "timeseries":
            self.fold_generator = TimeSeriesSplit(fold)
        else:
            self.fold_generator = fold_strategy

    def _clean_column_names(self):
        """Add CleanColumnNames to the pipeline."""
        self.logger.info("Set up column name cleaning.")
        self.pipeline.steps.append(
            ("clean_column_names", TransformerWrapper(CleanColumnNames()))
        )

    def _encode_target_column(self):
        """Add LabelEncoder to the pipeline."""
        self.logger.info("Set up label encoding.")
        self.pipeline.steps.append(
            ("label_encoding", TransformerWrapperWithInverse(LabelEncoder()))
        )



    def _iterative_imputation(
        self,

        categorical_iterative_imputer,
    ):

        self.logger.info("Set up iterative imputation.")

        classifiers = {
            k: v
            for k, v in get_all_class_model_containers(self).items()
            if not v.is_special
        }



        if isinstance(categorical_iterative_imputer, str):
            if categorical_iterative_imputer not in classifiers:
                raise ValueError(
                    f"Allowed estimators are: {', '.join(classifiers)}."
                )
            categorical_iterative_imputer = classifiers[
                categorical_iterative_imputer
            ].class_def(**classifiers[categorical_iterative_imputer].args)
        elif not hasattr(categorical_iterative_imputer, "predict"):
            raise ValueError(

                "error."
            )

        categorical_indices = [
            i
            for i in range(len(self.X.columns))
            if self.X.columns[i] in self._fxs["Categorical"]
        ]

        def get_prepare_estimator_for_categoricals_type(estimator, estimators_dict):
            # See naloxlib.internal.preprocess.iterative_imputer
            fit_params = {}
            if not categorical_indices:
                return estimator, fit_params
            if isinstance(estimator, estimators_dict["lightgbm"].class_def):
                return "fit_params_categorical_feature"
            elif "catboost" in estimators_dict and isinstance(
                estimator, estimators_dict["catboost"].class_def
            ):
                return "params_cat_features"
            elif "xgboost" in estimators_dict and isinstance(
                estimator, estimators_dict["xgboost"].class_def
            ):
                return "ordinal"
            elif isinstance(
                estimator,
                (
                    estimators_dict["rf"].class_def,
                    estimators_dict["et"].class_def,
                    estimators_dict["dt"].class_def,
                    estimators_dict["ada"].class_def,
                    estimators_dict.get(
                        "gbr",
                        estimators_dict.get("gbc", estimators_dict["rf"]),
                    ).class_def,
                ),
            ):
                return "ordinal"
            else:
                return "one_hot"





    def _polynomial_features(self, polynomial_degree):
        """Create polynomial features from the existing ones."""
        self.logger.info("Set up polynomial features.")

        polynomial = TransformerWrapper(
            transformer=PolynomialFeatures(
                degree=polynomial_degree,
                interaction_only=False,
                include_bias=False,
                order="C",
            ),
        )

        self.pipeline.steps.append(("polynomial_features", polynomial))

    def _low_variance(self, low_variance_threshold):
        """Drop features with too low variance."""
        self.logger.info("Set up variance threshold.")

        if low_variance_threshold < 0:
            raise ValueError(
                "Invalid value for the ignore_low_variance parameter. "
                f"The value should be >0, got {low_variance_threshold}."
            )
        else:
            variance_estimator = TransformerWrapper(
                transformer=VarianceThreshold(low_variance_threshold),
                exclude=self._fxs["Keep"],
            )

        self.pipeline.steps.append(("low_variance", variance_estimator))



    def _bin_numerical_features(self, bin_numeric_features):
        """Bin numerical features to 5 clusters."""
        self.logger.info("Set up binning of numerical features.")

        check_features_exist(bin_numeric_features, self.X)
        binning_estimator = TransformerWrapper(
            transformer=KBinsDiscretizer(encode="ordinal", strategy="kmeans"),
            include=bin_numeric_features,
        )

        self.pipeline.steps.append(("bin_numeric_features", binning_estimator))



    def _transformation(self, transformation_method):
        """Power transform the data to be more Gaussian-like."""
        self.logger.info("Set up column transformation.")

        if transformation_method == "yeo-johnson":
            transformation_estimator = PowerTransformer(
                method="yeo-johnson", standardize=False, copy=True
            )
        elif transformation_method == "quantile":
            transformation_estimator = QuantileTransformer(
                random_state=self.seed,
                output_distribution="normal",
            )
        else:
            raise ValueError(
                "Invalid value for the transformation_method parameter. "
                "The value should be either yeo-johnson or quantile, "
                f"got {transformation_method}."
            )

        self.pipeline.steps.append(
            ("transformation", TransformerWrapper(transformation_estimator))
        )

    def _normalization(self, normalize_method):
        """Scale the features."""
        self.logger.info("Set up feature normalization.")

        norm_dict = {
            "zscore": StandardScaler(),
            "minmax": MinMaxScaler(),
            "maxabs": MaxAbsScaler(),
            "robust": RobustScaler(),
        }
        if normalize_method in norm_dict:
            normalize_estimator = TransformerWrapper(norm_dict[normalize_method])
        else:
            raise ValueError(
                "Invalid value for the normalize_method parameter, got "
                f"{normalize_method}. Possible values are: {' '.join(norm_dict)}."
            )

        self.pipeline.steps.append(("normalize", normalize_estimator))

    def _pca(self, pca_method, pca_components):
        """Apply Principal Component Analysis."""
        self.logger.info("Set up PCA.")

        pca_dict = {
            "linear": PCA(n_components=pca_components),
            "kernel": KernelPCA(n_components=pca_components, kernel="rbf"),
            "incremental": IncrementalPCA(n_components=pca_components),
        }
        if pca_method in pca_dict:
            pca_estimator = TransformerWrapper(
                transformer=pca_dict[pca_method],
                exclude=self._fxs["Keep"],
            )
        else:
            raise ValueError(
                "Invalid value for the pca_method parameter, got "
                f"{pca_method}. Possible values are: {' '.join(pca_dict)}."
            )

        self.pipeline.steps.append(("pca", pca_estimator))

    def _feature_selection(
        self,
        feature_selection_method,
        feature_selection_estimator,
        n_features_to_select,
    ):
        """Select relevant features."""
        self.logger.info("Set up feature selection.")

        if self._ml_usecase == MLUsecase.CLASSIFICATION:
            func = get_all_class_model_containers
        else:
            # func = get_all_reg_model_containers
            pass

        models = {k: v for k, v in func(self).items() if not v.is_special}
        if isinstance(feature_selection_estimator, str):
            if feature_selection_estimator not in models:
                raise ValueError(
                    "Invalid value for the feature_selection_estimator "
                    f"parameter, got {feature_selection_estimator}. Allowed "
                    f"estimators are: {', '.join(models)}."
                )
            fs_estimator = models[feature_selection_estimator].class_def()
        elif not hasattr(feature_selection_estimator, "predict"):
            raise ValueError(
                "Invalid value for the feature_selection_estimator parameter. "
                "The provided estimator does not adhere to sklearn's API."
            )
        else:
            fs_estimator = feature_selection_estimator

        if 0 < n_features_to_select < 1:
            n_features_to_select = int(n_features_to_select * self.X.shape[1])
        elif n_features_to_select > self.X.shape[1]:
            raise ValueError(
                "Invalid value for the n_features_to_select parameter. The number of "
                "feature to select should be less than the starting number of features."
            )

        if feature_selection_method.lower() == "univariate":
            if self._ml_usecase == MLUsecase.CLASSIFICATION:
                func = f_classif
            else:
                func = f_regression
            feature_selector = TransformerWrapper(
                transformer=SelectKBest(score_func=func, k=n_features_to_select),
                exclude=self._fxs["Keep"],
            )
        elif feature_selection_method.lower() == "classic":
            feature_selector = TransformerWrapper(
                transformer=SelectFromModel(
                    estimator=fs_estimator,
                    threshold=-np.inf,
                    max_features=n_features_to_select,
                ),
                exclude=self._fxs["Keep"],
            )
        elif feature_selection_method.lower() == "sequential":
            feature_selector = TransformerWrapper(
                transformer=SequentialFeatureSelector(
                    estimator=fs_estimator,
                    n_features_to_select=n_features_to_select,
                    n_jobs=self.n_jobs_param,
                ),
                exclude=self._fxs["Keep"],
            )
        else:
            raise ValueError(
                "Invalid value for the feature_selection_method parameter, "
                f"got {feature_selection_method}. Possible values are: "
                "'classic', 'univariate' or 'sequential'."
            )

        self.pipeline.steps.append(("feature_selection", feature_selector))

    def _add_custom_pipeline(self, custom_pipeline, custom_pipeline_position):
        """Add custom transformers to the pipeline."""
        self.logger.info("Set up custom pipeline.")

        # Determine position to insert
        if custom_pipeline_position < 0:
            # -1 becomes last, etc...
            pos = len(self.pipeline.steps) + custom_pipeline_position + 1
        else:
            # +1 because of the placeholder
            pos = custom_pipeline_position + 1

        for name, estimator in normalize_custom_transformers(custom_pipeline):
            self.pipeline.steps.insert(pos, (name, TransformerWrapper(estimator)))
            pos += 1

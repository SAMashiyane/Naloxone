# Author: Salio(Mohtarami)
# https://github.com/SAMashiyane
# Date: 2023-2024


import re
from collections import OrderedDict, defaultdict
from inspect import signature

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin, clone
from naloxlib.efficacy.depend import to_df, to_series, variable_return

class TransformerWrapper(BaseEstimator, TransformerMixin):


    def __init__(self, transformer, include=None, exclude=None):
        self.transformer = transformer
        self.include = include
        self.exclude = exclude

        self._train_only = getattr(transformer, "_train_only", False)
        self._include = self.include
        self._exclude = self.exclude or []
        self._feature_names_in = None

    @property
    def feature_names_in_(self):
        return self._feature_names_in

    def _name_cols(self, array, df):

        # If columns were only transformed, return og names
        if array.shape[1] == len(self._include):
            return self._include


        temp_cols = []
        for i, col in enumerate(array.T, start=2):

            mask = df.apply(
                lambda c: np.array_equal(
                    c,
                    col,
                    equal_nan=is_numeric_dtype(c)
                    and np.issubdtype(col.dtype, np.number),
                )
            )
            if any(mask) and mask[mask].index.values[0] not in temp_cols:
                temp_cols.append(mask[mask].index.values[0])
            else:
                # If the column is new, use a default name
                counter = 1
                while True:
                    n = f"feature {i + counter + df.shape[1] - len(self._include)}"
                    if (n not in df or n in self._include) and n not in temp_cols:
                        temp_cols.append(n)
                        break
                    else:
                        counter += 1

        return temp_cols

    def _reorder_cols(self, df, original_df):

        # Check if columns returned by the transformer are already in the dataset
        for col in df:
            if col in original_df and col not in self._include:
                raise ValueError(
                    f"Column '{col}' returned by transformer {self.transformer} "
                    "already exists in the original dataset."
                )


        try:
            original_df.index = df.index
        except ValueError:  # Length mismatch
            raise IndexError(
                f"Length of values ({len(df)}) does not match length of "
                f"index ({len(original_df)}). This usually happens when "
                "transformations that drop rows aren't applied on all "
                "the columns."
            )


        columns = OrderedDict()
        for col in original_df:
            if col in df or col not in self._include:
                columns[col] = None


            columns.update(
                [
                    (c, None)
                    for c in df.columns
                    if c.startswith(f"{col}_") and c not in original_df
                ]
            )


        columns.update([(col, None) for col in df if col not in columns])

        columns = list(columns.keys())


        new_df = df.merge(
            right=original_df[[col for col in original_df if col in columns]],
            how="outer",
            left_index=True,
            right_index=True,
            suffixes=("", "__drop__"),
        )
        new_df = new_df.drop(new_df.filter(regex="__drop__$").columns, axis=1)

        return new_df[columns]

    def _prepare_df(self, X, out):

        if not isinstance(out, pd.DataFrame):
            if hasattr(self.transformer, "get_feature_names_out"):
                columns = self.transformer.get_feature_names_out()
            elif hasattr(self.transformer, "get_feature_names"):
                # Some estimators have legacy method, e.g. category_encoders
                columns = self.transformer.get_feature_names()
            else:
                columns = self._name_cols(out, X)

            out = to_df(out, index=X.index, columns=columns)

        # Reorder columns if only a subset was used
        if len(self._include) != X.shape[1]:
            return self._reorder_cols(out, X)
        else:
            return out

    def fit(self, X=None, y=None, **fit_params):

        self.target_name_ = None
        feature_names_in = []
        if hasattr(X, "columns"):
            feature_names_in += list(X.columns)
        if hasattr(y, "name"):
            feature_names_in += [y.name]
            self.target_name_ = y.name
        if feature_names_in:
            self._feature_names_in = feature_names_in

        args = []
        transformer_params = signature(self.transformer.fit).parameters
        if "X" in transformer_params and X is not None:
            if self._include is None:
                self._include = [
                    c for c in X.columns if c in X and c not in self._exclude
                ]
            elif not self._include:  # Don't fit if empty list
                return self
            else:
                self._include = [
                    c for c in self._include if c in X and c not in self._exclude
                ]
            args.append(X[self._include])
        if "y" in transformer_params and y is not None:
            args.append(y)

        self.transformer.fit(*args, **fit_params)
        return self

    def transform(self, X=None, y=None):
        X = to_df(X, index=getattr(y, "index", None))
        y = to_series(y, index=getattr(X, "index", None), name=self.target_name_)

        args = []
        transform_params = signature(self.transformer.transform).parameters
        if "X" in transform_params:
            if X is not None:
                if self._include is None:
                    self._include = [
                        c for c in X.columns if c in X and c not in self._exclude
                    ]
                elif not self._include:  # Don't transform if empty list
                    return variable_return(X, y)
            else:
                return variable_return(X, y)
            args.append(X[self._include])
        if "y" in transform_params:
            if y is not None:
                args.append(y)
            elif "X" not in transform_params:
                return X, y

        output = self.transformer.transform(*args)

        # Transform can return X, y or both
        if isinstance(output, tuple):
            new_X = self._prepare_df(X, output[0])
            new_y = to_series(output[1], index=new_X.index, name=y.name)
        else:
            if len(output.shape) > 1:
                new_X = self._prepare_df(X, output)
                new_y = y if y is None else y.set_axis(new_X.index)
            else:
                new_y = to_series(output, index=y.index, name=y.name)
                new_X = X if X is None else X.set_index(new_y.index)

        return variable_return(new_X, new_y)


class TransformerWrapperWithInverse(TransformerWrapper):
    def inverse_transform(self, y):
        y = to_series(y, index=getattr(y, "index", None), name=self.target_name_)
        output = self.transformer.inverse_transform(y)
        return to_series(output, index=y.index, name=y.name)


class CleanColumnNames(BaseEstimator, TransformerMixin):

    def __init__(self, match=r"[\]\[\,\{\}\"\:]+"):
        self.match = match

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.rename(columns=lambda x: re.sub(self.match, "", str(x)))




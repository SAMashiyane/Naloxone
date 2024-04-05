
import pandas as pd
from naloxlib.inward.naloxlib_experiment.supervised_experiment import (
    _SupervisedExperiment,
)

class _NonTSSupervisedExperiment(_SupervisedExperiment):
    def __init__(self) -> None:
        super().__init__()

    @property
    def test(self):

        return self.dataset.loc[self.idx[1], :]

    @property
    def X(self):

        return self.dataset.drop(self.target_param, axis=1)

    @property
    def X_train(self):

        return self.train.drop(self.target_param, axis=1)

    @property
    def X_test(self):

        return self.test.drop(self.target_param, axis=1)

    @property
    def dataset_transformed(self):

        return pd.concat([self.train_transformed, self.test_transformed])

    @property
    def train_transformed(self):

        return pd.concat(
            [
                *self.pipeline.transform(
                    X=self.X_train,
                    y=self.y_train,
                    filter_train_only=False,
                )
            ],
            axis=1,
        )

    @property
    def test_transformed(self):

        return pd.concat(
            [
                *self.pipeline.transform(
                    X=self.X_test,
                    y=self.y_test,
                )
            ],
            axis=1,
        )

    @property
    def X_transformed(self):

        return pd.concat([self.X_train_transformed, self.X_test_transformed])

    @property
    def y_transformed(self):

        return pd.concat([self.y_train_transformed, self.y_test_transformed])

    @property
    def X_train_transformed(self):

        return self.train_transformed.drop(self.target_param, axis=1)

    @property
    def y_train_transformed(self):

        return self.train_transformed[self.target_param]

    @property
    def X_test_transformed(self):

        return self.test_transformed.drop(self.target_param, axis=1)

    @property
    def y_test_transformed(self):

        return self.test_transformed[self.target_param]

    def _create_model_get_train_X_y(self, X_train, y_train):

        if X_train is not None:
            data_X = X_train.copy()
        else:
            if self.X_train is None:
                data_X = None
            else:
                data_X = self.X_train
        data_y = self.y_train if y_train is None else y_train.copy()
        return data_X, data_y

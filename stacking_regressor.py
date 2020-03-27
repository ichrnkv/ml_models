from lightgbm import LGBMRegressor
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import pandas as pd
import warnings

warnings.filterwarnings('ignore')


def train_test_split(X, y, test_size):
    """
    Makes train-test split of shuffeled data
    """

    split_idx = int(y.shape[0] * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


class StackingRegressor(BaseEstimator, RegressorMixin):
    """
    Stacking of multiple regression models through meta_algoritm.
    Uses KFold validation.
    """

    def __init__(self, basic_models, meta_algorithm, use_base_features=False, n_folds=3, random_state=13):
        self.basic_models = basic_models
        self.meta_algorithm = meta_algorithm
        self.use_base_features = use_base_features
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, y):
        kfold = KFold(n_splits=self.n_folds, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.basic_models)))
        for idx, model in enumerate(self.basic_models):
            for train_index, valid_index in kfold.split(X, y):
                model.fit(X.loc[train_index], y[train_index])
                meta_preds = model.predict(X.loc[valid_index])
                meta_features[valid_index, idx] = meta_preds

        if self.use_base_features:
            self.meta_algorithm.fit(np.hstack((X, meta_features)), y)
        else:
            self.meta_algorithm.fit(meta_features, y)
        return self

    def predict(self, X):
        meta_features = np.column_stack(
            [
                model.predict(X) for model in self.basic_models
            ]
        )
        if self.use_base_features:
            return self.meta_algorithm.predict(np.hstack((X, meta_features)))
        else:
            return self.meta_algorithm.predict(meta_features)


if __name__ == '__main__':
    # read data
    data = pd.read_excel('data/bikes.xlsx')
    data.drop('dt', axis=1, inplace=True)
    data.columns = ["".join (c if c.isalnum() else "\_" for c in str(x)) for x in data.columns]

    # random seed
    rs = np.random.seed(13)
    data = data.sample(frac=1, random_state=rs).reset_index(drop=True)

    # prepare data for modeling
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.25)

    models = [
        LinearRegression(),
        RandomForestRegressor(n_jobs=-1, random_state=rs),
        LGBMRegressor(n_jobs=-1, random_state=rs)
    ]

    stacking = StackingRegressor(basic_models=models,
                                 meta_algorithm=LGBMRegressor(n_jobs=-1, random_state=rs),
                                 use_base_features=True,
                                 n_folds=3,
                                 random_state=rs)

    stacking.fit(X_train, y_train)
    preds = stacking.predict(X_test)
    print('MAE is {}'.format(mean_absolute_error(y_test, preds)))
    print('R2 score is {}'.format(r2_score(y_test, preds)))

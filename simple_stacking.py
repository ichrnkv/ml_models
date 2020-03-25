from lightgbm import LGBMClassifier
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings

# disable warnings
warnings.filterwarnings('ignore')


def get_cat_columns(df):
    """
    Returns list of categorical columns
    """
    cat = []
    for column in df.columns:
        if df[column].dtype == 'object':
            cat.append(column)
    return cat


def train_test_split(X, y, test_size):
    """
    Makes train-test split of shuffeled data
    """

    split_idx = int(y.shape[0] * (1 - test_size))
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, basic_models, meta_algorithm, n_folds, random_state):
        self.basic_models = basic_models
        self.meta_algorithm = meta_algorithm
        self.n_folds = n_folds
        self.random_state = random_state

    def fit(self, X, y):
        kfold = KFold(n_splits=self.n_folds, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.basic_models)))
        for idx, model in enumerate(self.basic_models):
            for train_index, valid_index in kfold.split(X, y):
                model.fit(X.loc[train_index], y[train_index])
                meta_preds = model.predict_proba(X.loc[valid_index])[:, 1]
                meta_features[valid_index, idx] = meta_preds

        print('Stacking is done!')
        self.meta_algorithm.fit(meta_features, y)
        return self

    def predict_proba(self, X):
        meta_features = np.column_stack(
            [
                model.predict_proba(X)[:, 1] for model in self.basic_models
            ]
        )
        return self.meta_algorithm.predict_proba(meta_features)

    def predict(self, X):
        meta_features = np.column_stack(
            [
                model.predict_proba(X)[:, 1] for model in self.basic_models
            ]
        )
        return self.meta_algorithm.predict(meta_features)


if __name__ == '__main__':
    # read data
    data = pd.read_csv('../data/data.adult.csv')
    data.rename(columns={'>50K,<=50K': 'target'}, inplace=True)
    data['target'].map({'<=50K': 0, '>50K': 1})

    # encode categorical features
    categorical = get_cat_columns(data)
    le = LabelEncoder()
    data[categorical] = data[categorical].apply(le.fit_transform)

    # random seed
    rs = np.random.seed(13)
    data = data.sample(frac=1, random_state=rs).reset_index(drop=True)

    # prepare data for modeling
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, 0.25)

    models = [
        LogisticRegression(),
        RandomForestClassifier(n_jobs=-1, random_state=rs),
        LGBMClassifier(n_jobs=-1, random_state=rs)
    ]

    stacking = StackingClassifier(basic_models=models,
                                  meta_algorithm=LogisticRegression(),
                                  n_folds=3,
                                  random_state=rs)

    stacking.fit(X_train, y_train)
    pred_probs = stacking.predict_proba(X_test)[:, 1]
    preds = stacking.predict(X_test)
    print('ROC-AUC is {}'.format(roc_auc_score(y_test, pred_probs)))
    print('Accuracy is {}'.format(accuracy_score(y_test, preds)))

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def get_cat_columns(df):
    cat = []
    for column in df.columns:
        if df[column].dtype == 'object':
            cat.append(column)
    return cat


class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, basic_models, meta_algorithm, n_folds):
        self.basic_models = basic_models
        self.meta_algorithm = meta_algorithm
        self.n_folds = n_folds

    def fit(self, X, y):
        kfold = KFold(n_splits=self.n_folds)
        meta_features = np.zeros((X.shape[0], len(self.basic_models)))
        for idx, model in enumerate(self.basic_models):
            for train_index, valid_index in kfold.split(X, y):
                model.fit(X.loc[train_index], y[train_index])
                preds = model.predict_proba(X.loc[valid_index])[:, 1]
                meta_features[valid_index, idx] = preds

        print('Blending is done!')
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
    data = pd.read_csv('../data/data.adult.csv')
    data.rename(columns={'>50K,<=50K': 'target'}, inplace=True)
    data['target'].map({'<=50K': 0, '>50K': 1})

    categorical = get_cat_columns(data)
    le = LabelEncoder()
    data[categorical] = data[categorical].apply(le.fit_transform)

    X = data.drop('target', axis=1)
    y = data['target']

    lr = LogisticRegression()
    rf = RandomForestClassifier(n_jobs=-1)

    stacking = StackingClassifier([lr, rf], LogisticRegression(), n_folds=5)

    stacking.fit(X, y)
    pred_probs = stacking.predict_proba(X)[:, 1]
    preds = stacking.predict(X)
    print('ROC-AUC is {}'.format(roc_auc_score(y, pred_probs)))
    print('Accuracy is {}'.format(accuracy_score(y, preds)))

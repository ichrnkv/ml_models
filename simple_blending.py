import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def get_cat_columns(df):
    cat = []
    for column in df.columns:
        if df[column].dtype == 'object':
            cat.append(column)
    return cat


class Blending(BaseEstimator, ClassifierMixin):
    def __init__(self, models, alpha=1):
        self.models = models
        self.alpha = alpha
        self.weights = [alpha, 1 - alpha]
        assert alpha <= 1

    def fit(self, X, y):
        for model in self.models:
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        predictions_probas = np.column_stack(
            [
                model.predict_proba(X)[:, 1] for model in self.models
            ]
        )
        return np.sum(predictions_probas*self.weights, axis=1)

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models
        ])
        return np.sum(predictions * self.weights, axis=1)


if __name__ == '__main__':
    data = pd.read_csv('../data/data.adult.csv')
    data.rename(columns={'>50K,<=50K': 'target'}, inplace=True)
    data['target'].map({'<=50K': 0, '>50K': 1})

    categorical = get_cat_columns(data)
    le = LabelEncoder()
    data[categorical] = data[categorical].apply(le.fit_transform)

    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=13)

    lr = LogisticRegression()
    rf = RandomForestClassifier(n_jobs=-1)

    blend_params = {'alpha': np.arange(0, 1.1, .1)}
    blend = Blending([lr, rf], alpha=0.2)

    blend.fit(X_train, y_train)
    preds = blend.predict_proba(X_test)
    print('ROC-AUC is {}'.format(roc_auc_score(y_test, preds)))

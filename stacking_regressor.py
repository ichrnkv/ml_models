from sklearn.base import BaseEstimator, RegressorMixin

class Stacking(BaseEstimator, RegressorMixin):
    def __init__(self, models, meta_algorithm):
        self.models = models
        self.meta_algorithm = meta_algorithm
        self.features = [None for i in range(len(models))]
        
    def fit(self, X, y, features=None, cv=5, scoring='accuracy', random_state=None):
        """
        Fit base algotithms on training data and meta-algotithm on validation data.
        :param X_train: train features, pandas DataFrame
        :param X_valid: validation features, pandas DataFrame
        :param y_train: train labels, pandas Series
        :param y_valid: validation labels, pandas Series
        :param cv: cross-validation folds, int
        :param scoring : scoring metric, str
        :param random_state: random state
        """
        # списки признаков
        if features is None:
            self.features = [list(X.columns) for _ in range(len(self.models))]
        else:
            self.features = features
        
        # матрица метапризнаков
        self.meta_features = 0.001*np.random.randn(X.shape[0], len(self.models))
        
        for idx, clf in enumerate(self.models):
            # oob-ответы базовых алгоритмов
            self.meta_features[:, idx] += cross_val_predict(clf, X[self.features[idx]], y, cv=cv,
                                                   n_jobs=-1, method='predict')
            # обучаем базовый алгоритм
            clf.fit(X[self.features[idx]], y)
        # обучаем метаалгоритм на ответах базовых
        self.meta_algorithm.fit(self.meta_features, y )
        return self
    
    def predict(self, X):
        """
        Makes predictions for StackingClassifier
        :param X: test features, pandas DataFrame
        """
        # матрица метапризнаков
        x_meta = np.zeros((X.shape[0], len(self.models)))
        
        # заполняем матрицу метапризнаков
        for idx, clf in enumerate(self.models):
                x_meta[:, idx] = clf.predict(X[self.features[idx]])
        
        predictions = self.meta_algorithm.predict(x_meta)
        return predictions

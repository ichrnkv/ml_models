import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# наследуемся от класса BaseEstimator
class GradientBoosting(BaseEstimator):
    def sigmoid(self, z):
        """
        Сигмоида
        """
        z[z > 100] = 100 # ограничиваем аргумент чтобы не хранить в пямяти большое число
        z[z < -100] = -100 # ограничиваем аргумент чтобы не хранить в пямяти большое число
        return 1. / (1 + np.exp(-z))

    def log_loss_grad(self, y, p):
        """
        Градиент логлосса по прогнозам
        """
        y = y.T
        p = p.T
        return (p - y) / p / (1 - p)

    def __init__(self, n_estimators=10, learning_rate=0.01,
                 max_depth=3, random_state=17,
                 loss='log_loss', debug=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.loss_name = loss
        self.initialization = lambda y: np.mean(y) * np.ones([y.shape[0], ])

        if loss == 'log_loss':
            self.objective = log_loss
            self.objective_grad = self.log_loss_grad

        self.trees_ = []
        self.loss_by_iter = []

    def fit(self, X, y):
        """
        Обучает градиентный бустинг
        """
        self.X = X
        self.y = y
        
        # инициализируем ответы средним значением таргета
        b = self.initialization(y)
        prediction = b.copy()

        for t in range(self.n_estimators):
            # определяем остатки
            resid = -self.objective_grad(y, prediction)

            # обучаем дерево на остатках
            tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                         random_state=self.random_state)
            tree.fit(X, resid)

            b = tree.predict(X)

            self.trees_.append(tree)
            prediction += self.learning_rate * b
            self.loss_by_iter.append(self.objective(y, prediction))

        self.train_pred = prediction

        if self.loss_name == 'log_loss':
            self.train_pred = self.sigmoid(self.train_pred)

        return self

    def predict_proba(self, X):
        """
        Возвращает вероятность метки 1
        """
        
        # инициализируем ответы средним значением таргета
        pred = np.mean(y) * np.ones([y.shape[0], ])

        for t in range(self.n_estimators):
            pred += self.learning_rate * self.trees_[t].predict(X)

        if self.loss_name == 'log_loss':
            return self.sigmoid(pred)

    def predict(self, X):
        """
        Возвращает прогноз
        """
        probas = self.predict_proba(X)

        if self.loss_name == 'log_loss':
            max_accuracy = 0
            best_threshold = 0
            # подбираем порог разбиения
            for threshold in np.linspace(0.01, 1.01, 100):
                accuracy = accuracy_score(self.y, self.train_pred > threshold)
                if accuracy > max_accuracy:
                    max_accuracy = accuracy
                    best_threshold = threshold
            predictions = probas > best_threshold
            return predictions.astype(int)

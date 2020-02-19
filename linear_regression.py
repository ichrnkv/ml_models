import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def error_function(x, y, w):
    l=len(y)
    return (1/l)*(y - np.dot(x, w)).T.dot(y - np.dot(x, w))


class LinearRegression:
    def __init__(self):
        self.weights = None
        # self.x = None
        # self.y = None

    def fit(self, x_train, y_train, lr=0.001):
        """
        Обучает линейную регрессию стохастическим градиентным спуском
        """
        # случайные веса
        self.weights = np.random.randint(0, 10**3, x_train.shape[1])
        self.weights = np.array(self.weights.copy(), dtype=np.float32)

        # большая ошибка для старта обучения
        previous_error = 10**6
        self.weights.reshape((len(self.weights), 1))
        current_error = error_function(x_train, y_train, self.weights)

        while np.abs(previous_error - current_error) > 0.001:
            previous_error = current_error
            i = np.random.randint(0, len(y_train))
            derivatives = [0] * len(self.weights)
            for j in range(len(self.weights)):
                derivatives[j] = (np.dot(x_train[i], self.weights) - y_train[i] ) * x_train[i][j]

            # обновляем веса
            for j in range(len(self.weights)):
                self.weights[j] -= lr * derivatives[j]
            current_error = error_function(x_train, y_train, self.weights)

        return self.weights

    def predict(self, x_test):
        """
        Прогноз обученной модели
        """
        return x_test.dot(self.weights)


if __name__ == '__main__':
    df = pd.read_csv('data/bikes.csv')
    x = df.drop(['dt', 'casual', 'other', 'total'], axis=1).values
    Y = df['total']
    X = np.empty((x.shape[0], x.shape[1]+1))
    x = (x - x.min())/(x.max() - x.min())
    X[:, 0] = 1
    X[:, 1:] = x
    X = np.array(X.copy(), dtype=np.float32)
    lr = LinearRegression()
    lr.fit(X, Y)
    predictions = lr.predict(X)
    error = mean_absolute_error(Y, predictions)
    print('MAE: ', error)

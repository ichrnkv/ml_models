import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def error_function(x, y, w):
    """
    MSE
    :return:
    """
    return (y - np.dot(x, w)).T.dot(y - np.dot(x, w))


def stochastic_gd(x, y, lr=0.001):
    # случайные веса
    w = np.random.randint(0, 10**3, x.shape[1])
    w = np.array(w.copy(), dtype=np.float32)

    # большая ошибка для старта обучения
    previous_error = 10**12
    w.reshape((len(w), 1))
    current_error = error_function(X, Y, w)

    while np.abs(previous_error - current_error) > 0.001:
        previous_error = current_error
        i = np.random.randint(0, len(Y))
        derivatives = [0] * len(w)
        for j in range(len(w)):
            derivatives[j] += (Y[i] - np.dot(X[i], w)) * X[i][j]

        # обновляем веса
        for j in range(len(w)):
            w[j] += 0.01 * derivatives[j]
        current_error = error_function(X, Y, w)
    return w


if __name__ == '__main__':
    df = pd.read_csv('../data/bikes.csv')
    x = df.drop(['dt', 'casual', 'other', 'total'], axis=1).values
    Y = df['total']
    X = np.empty((x.shape[0], x.shape[1]+1))
    x = (x - x.min())/(x.max() - x.min())
    X[:, 0] = 1
    X[:, 1:] = x
    X = np.array(X.copy(), dtype=np.float32)
    w = stochastic_gd(X, Y, lr=0.001)
    predictions = X.dot(w)
    error = mean_absolute_error(Y, predictions)
    print('MAE: ', error)

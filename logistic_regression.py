import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def decide(x):
    return np.where(x >= 0.5, 1, -1)


class LogisticRegression:
    def __init__(self):
        self.weights = None

    def fit(self, x, y, lr=0.01, max_iter=1000, random_state=17):
        w = np.zeros((1, x.shape[1]))
        np.random.seed(random_state)
        iterations = 0

        while iterations < max_iter:
            i = np.random.randint(0, len(y))
            for j in range(len(w[0])):
                w[0][j] = w[0][j] + lr * y[i] * x[i][j] * sigmoid(-y[i] * x[i] @ w.T)

            iterations += 1
            if iterations % 1000 == 0:
                print(iterations)
        self.weights = w
        print('Training is done')

    def predict_proba(self, x_test):
        return sigmoid(x_test@self.weights.T)

    def predict(self, x_test):
        probas = self.predict_proba(x_test)
        return decide(probas)


if __name__ == '__main__':
    df = pd.read_csv('train.csv')
    df['target'] = df['target'].map({0: -1, 1: 1})
    X = df.drop('target', axis=1).values
    Y = df['target']
    log_reg = LogisticRegression()
    log_reg.fit(X, Y, lr=0.001, max_iter=10000)
    predictions = log_reg.predict(X)
    print('Accuracy score is : ', accuracy_score(y, predictions))

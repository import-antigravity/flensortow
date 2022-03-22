import numpy as np

import util
from classifier import Classifier


class Perceptron(Classifier):
    def __init__(self, n_features: int, learning_rate: float = 1e-2, regularizer: float = 0.1):
        self._w = np.random.standard_normal(n_features)
        self._b = np.random.standard_normal(1)
        self._hparams = {'l': regularizer, 'lr': learning_rate}

    def fit(self, X, y):
        y_hat = self.predict(X)
        print('loss:', self.loss(y, y_hat))

        '''
        n_examples = y_hat.size
        w_grad = np.zeros_like(self._w)
        b_grad = np.zeros_like(self._b)
        for i in range(n_examples):
            w_grad += 2 * (y_hat[i] - y[i]) * util.sigmoid_prime(np.dot(X[i], self._w) + self._b) * X[i]
            w_grad += 2 * self._hparams['l'] * self._w
            b_grad += 2 * (y_hat[i] - y[i]) * util.sigmoid_prime(np.dot(X[i], self._w) + self._b)
        
        w_grad /= n_examples
        b_grad /= n_examples
        '''

        mse_grad = 2 * (y_hat - y).reshape(-1, 1)
        sig_grad = util.sigmoid_prime(np.tensordot(X, self._w, 1) + self._b).reshape(-1, 1)
        reg_grad = (2 * self._hparams['l'] * self._w).reshape(1, -1)
        w_grad = np.mean(mse_grad * sig_grad * X + reg_grad, axis=0)
        b_grad = np.mean(mse_grad * sig_grad, axis=0)

        self._w -= w_grad * self._hparams['lr']
        self._b -= b_grad * self._hparams['lr']

    def predict(self, X):
        '''
        y_hat = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                y_hat[i] += self._w[j] * X[i, j]
        y_hat += self._b
        return y_hat
        '''

        return util.sigmoid(np.tensordot(X, self._w, 1) + self._b)

    def loss(self, y, y_hat):
        return util.mse_loss(y, y_hat) + np.mean(np.power(self._w, 2))

    @property
    def parameters(self):
        return {'w': self._w,
                'b': self._b}

    @property
    def hparams(self):
        return self._hparams

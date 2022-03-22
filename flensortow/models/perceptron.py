import numpy as np

import util
from classifier import Classifier


class Perceptron(Classifier):
    def __init__(self, n_features: int, learning_rate: float = 1e-2, regularizer: float = 0.1):
        self._w = np.random.standard_normal(n_features)
        self._b = np.random.standard_normal(1)
        self._hparams = {'l': regularizer, 'lr': learning_rate}

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def loss(self, y, y_hat):
        return util.mse_loss(y, y_hat) + np.mean(np.power(self._w, 2))

    @property
    def parameters(self):
        return {'w': self._w,
                'b': self._b}

    @property
    def hparams(self):
        return self._hparams

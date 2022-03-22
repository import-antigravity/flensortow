import numpy as np
import tensorflow as tf

import datasets
import util
from classifier import Classifier



class Perceptron(Classifier):
    def __init__(self, n_features: int, learning_rate: float = 1e-2, regularizer: float = 0.1):
        self._W = tf.Variable(tf.random.normal(
                  (n_features, 1),
                   mean=0.0,
                  stddev=1.0,
                  dtype=tf.dtypes.float32,
                  seed=None,
                  name='W1'))
        self._b = tf.Variable(tf.random.normal(
                  (1, 1),
                  mean=0.0,
                  stddev=1.0,
                  dtype=tf.dtypes.float32,
                  name='b1'))
        self._hparams = {'l': regularizer, 'lr': learning_rate}

    def fit(self, X, y):
        X = tf.convert_to_tensor(X, dtype=tf.dtypes.float32)
        y = tf.convert_to_tensor(y, dtype=tf.dtypes.float32)
        with tf.GradientTape() as t:
            current_loss = self.loss(y, self.predict(X))
            print('loss', current_loss)
            grads = t.gradient(current_loss, self.parameters)

        for param_name in self.parameters:
            self.parameters[param_name].assign_sub(self.hparams['lr'] * grads[param_name])

    def predict(self, X):
        return tf.sigmoid(tf.linalg.matmul(X, self._W) + self._b)

    def loss(self, y, y_hat):
        return tf.reduce_mean(tf.square(y_hat - y)) + self.hparams['l'] * tf.reduce_mean(tf.linalg.norm(self._W))

    @property
    def parameters(self):
        return {'W': self._W,
                'b': self._b}

    @property
    def hparams(self):
        return self._hparams


if __name__ == '__main__':
    X, y = datasets.iris
    m = Perceptron(n_features=X.shape[1])
    for _ in range(100):
        m.fit(X, y)


from abc import ABC, abstractmethod


class Classifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def loss(self, y, y_hat):
        pass

    @property
    @abstractmethod
    def parameters(self):
        pass

    @property
    @abstractmethod
    def hparams(self):
        pass

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from classifier import Classifier
from util import sigmoid

matplotlib.interactive(True)


class ClassifierVis2D(object):
    def __init__(self, X, y, classifier, resolution=10):
        self.X = X
        self.y = y
        self.classifier = classifier
        self.resolution = resolution

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        data_x, data_y = zip(*self.X)
        self.ax.scatter(data_x, data_y, self.y)

        xmin, xmax = min(data_x), max(data_x)
        ymin, ymax = min(data_y), max(data_y)
        self.spatial_x, self.spatial_y = np.meshgrid(
            np.linspace(xmin, xmax, self.resolution),
            np.linspace(ymin, ymax, self.resolution)
        )

        self.surf = None
        self._make_surf()

    def update(self):
        self.surf.remove()
        self._make_surf()
        plt.draw()
        self.fig.canvas.flush_events()

    def _make_surf(self):
        y_hat = self.classifier.predict(
            np.vstack((self.spatial_x.ravel(), self.spatial_y.ravel())).T
        ).reshape(self.resolution, self.resolution)
        self.ax.set_zlim(
            min(self.y.min(), y_hat.min()),
            max(self.y.max(), y_hat.max())
        )
        self.surf = self.ax.plot_surface(
            self.spatial_x, self.spatial_y, y_hat,
            rstride=1, cstride=1, linewidth=0, antialiased=False)


if __name__ == '__main__':
    class DummyClassifier(Classifier):
        def __init__(self, f):
            self.f = f

        def fit(self, X, y):
            pass

        def predict(self, X):
            return self.f(X)

        def loss(self, y, y_hat):
            return 0.

        @property
        def parameters(self):
            return None

        @property
        def hparams(self):
            return None

    v = ClassifierVis2D(
        np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]),
        np.array([0., 1., 1., 0.]),
        DummyClassifier(lambda x: sigmoid(x.sum(axis=1)))
    )
    plt.show()
    plt.pause(10)

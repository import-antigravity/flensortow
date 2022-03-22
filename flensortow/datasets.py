from collections import namedtuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

Dataset = namedtuple('Dataset', ['X', 'y'])

OR = Dataset(
    np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]),
    np.array([0, 1, 1, 1])
)

AND = Dataset(
    np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]),
    np.array([0, 0, 0, 1])
)

XOR = Dataset(
    np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ]),
    np.array([0, 1, 1, 0])
)

iris_truncated = Dataset(PCA(n_components=2).fit_transform(load_iris()['data']),
                         (load_iris()['target'] == 1).astype(int))
iris = Dataset(load_iris()['data'], (load_iris()['target'] == 1).astype(int))


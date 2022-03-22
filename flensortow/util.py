import abc
import collections.abc
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import List, Callable, Type, Tuple, Union

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

def mse_loss(y, y_hat):
    return np.mean((y_hat - y) ** 2)

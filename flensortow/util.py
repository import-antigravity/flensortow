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


class Tensor(object):
    def __init__(self, array, dtype=None, requires_grad=False):
        self._data = np.array(array, dtype=dtype)
        self._grad = np.zeros_like(self._data, dtype=dtype)
        self._requires_grad = requires_grad
        self._grad_fns = []
        self._parents = set()
        self._children = []

    @property
    def shape(self):
        return self._data.shape

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def grad(self):
        return self._grad

    @grad.setter
    def grad(self, value):
        self._grad = value

    @property
    def grad_fn(self) -> Callable[['Tensor', ...], 'Tensor']:
        return self._grad_fn if self._grad_fn else lambda: np.ones(1)

    @grad_fn.setter
    def grad_fn(self, value):
        if self._grad_fn is None:
            self._grad_fn = value
        else:
            raise ValueError('grad_fn cannot be changed once set')

    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self, value):
        self._requires_grad = value

    def backward(self, *output_grads):
        if output_grads is None:
            Tensor.backward_from(self)
        else:
            self.grad = self.grad_fn(*output_grads)

    def numpy(self):
        return self._data

    def attach(self, *children: 'Tensor'):
        self._children = list(children)
        for child in children:
            child._parents.add(self)

    @staticmethod
    def backward_from(start_node: 'Tensor'):
        node_set = {start_node}
        done = set()
        while node_set:
            node = node_set.pop()
            output_grads = tuple(c.grad for c in node._children)
            node.grad = node.grad_fn(*output_grads)

            done.add(node)

            for parent in node._parents:
                if parent.requires_grad and all(c in done for c in parent._children):
                    node_set.add(parent)


@dataclass
class OperationContext:
    inputs: List[Callable] = field(default_factory=list)
    saved_tensors: Tuple[Tensor] = field(default_factory=tuple)


class DifferentiableFunction(abc.ABC):
    @classmethod
    def apply(cls):
        return AppliedFunction(cls)

    @staticmethod
    @abstractmethod
    def forward(context: OperationContext, *args: Tensor, **kwargs: Tensor) -> Union[Tuple, Tuple[Tensor, ...]]:
        pass

    @staticmethod
    @abstractmethod
    def backward(context: OperationContext, *output_grads: np.ndarray) -> np.ndarray:
        pass


class AppliedFunction(collections.abc.Callable):
    def __init__(self, fn_cls: Type[DifferentiableFunction]):
        self._fn_cls = fn_cls

    def __call__(self, *args, **kwargs):
        context = OperationContext()
        for t in (*args, *kwargs.values()):
            t.grad_fn = lambda output_grads: self._fn_cls.backward(context, *output_grads)
        return self._fn_cls.forward(context, *args, **kwargs)


class Add(DifferentiableFunction):
    @staticmethod
    def forward(context: OperationContext, a: Tensor, b: Tensor) -> Tensor:
        output = Tensor(
            a._data + b._data,
            requires_grad=a.requires_grad or b.requires_grad
        )
        return output

    @staticmethod
    def backward(context: OperationContext, output_grad: np.ndarray) -> np.ndarray:
        return output_grad


#add = Add.apply()


class Invert(DifferentiableFunction):
    @staticmethod
    def forward(context: OperationContext, x: Tensor) -> Tensor:
        output = Tensor(
            -x._data,
            requires_grad=x.requires_grad
        )
        return output

    @staticmethod
    def backward(context: OperationContext, output_grad: np.ndarray) -> np.ndarray:
        return output_grad


#invert = Invert.apply()


class Multiply(DifferentiableFunction):
    @staticmethod
    def forward(context: OperationContext, *args: Tensor, **kwargs: Tensor) -> Union[Tuple, Tuple[Tensor, ...]]:
        pass

    @staticmethod
    def backward(context: OperationContext, *output_grads: np.ndarray) -> np.ndarray:
        pass


#multiply = Multiply.apply()


class Sigmoid(DifferentiableFunction):
    @staticmethod
    def forward(context: OperationContext, x: Tensor) -> Tensor:
        output = Tensor(
            1 / (1 + np.exp(-x._data)),
            requires_grad=x.requires_grad
        )
        x.attach(output)
        context.saved_tensors = (output,)
        return output

    @staticmethod
    def backward(context: OperationContext, output_grad: np.ndarray) -> np.ndarray:
        output, = context.saved_tensors
        return output_grad * output._data * (1 - output._data)


#sigmoid = Sigmoid.apply()


def mse_loss(y, y_hat):
    return np.mean((y_hat - y) ** 2)

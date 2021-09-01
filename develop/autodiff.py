"""
use linked list to achieve auto differentiation
USE 'dict'
graph computation, to be specific
"""
import numpy as np
from helper import layer_init, relu, CE

# starts from BobNet again!
"""
def BobNet(x, layers=None, input_shape=None, act=act_df):
    x = x.reshape((-1, 28*28))
    if layers is not None:
        l1, l2 = layers[0], layers[1]
        return [x, x@l1, act(x@l1), x@l1@l2], [l1, l2]

    l1 = layer_init(784, 128)
    l2 = layer_init(128, 10)
    return [x, x@l1, act(x@l1), x@l1@l2], [l1, l2]

def backward(grad, weights, fpass):
    gradient = []
    dl2 = fpass[-2].T @ grad
    gradient.append(dl2)
    # if activation is relu
    dl1 = fpass[0].T @ ((fpass[-2] > 0).astype(np.float32)
                        * (grad @ (weights[-1].T)))
    gradient.append(dl1)

    return gradient[::-1]
"""

# replace w/ class method
# model doesn't contain value of input, but the weights themselves


class Model:
    """
    will involve graph
    allocates all the layer
    build up auto-diff
    """
    pass


class layers:
    """
    layers in general
    allocate weight and its gradient inside seems fine
    """

    def __init__(self, shape):
        self.shape = shape
        self.weights = layer_init(shape[0], shape[1])

    def __call__(self):
        return self.weights


class Dense(layers):
    """"
    backward(): auto diff
    """

    def __init__(self, shape):
        super().__init__(shape)

    def __call__(self):
        return self.weights


class optim:
    def __init__(self, weights, learning_rate):
        self.weights = weights
        self.learning_rate = learning_rate

    def sgd(self, gradient):
        self.weights -= self.learning_rate*gradient
        print(self.weights)

    def adam(self, gradient):
        pass


size1 = (784, 128)
L1 = layers(size1)
print(L1().shape)
print(Dense(size1)().shape)

l1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
dl1 = np.random.uniform(-1., 1., size=l1.shape)
print(l1)
optimizer = optim(l1, learning_rate=1e-3)
optimizer.sgd(dl1)

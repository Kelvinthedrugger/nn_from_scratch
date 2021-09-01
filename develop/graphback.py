"""
implement graph computation to achieve
auto-differentiation
from torch:
Tensor:                  Function
 grad         input -->  forward --> output
                            V
grad_fn  out_gradient <- backward <- in_gradient
"""
import numpy as np
from helper import layer_init


class layer:
    def __init__(self, shape):
        """build on graph"""
        self.shape = shape

    def __call__(self, x):
        """input row vector is much easier
        and computationally inexpensive"""
        self.weights = layer_init(self.shape[0], self.shape[1])
        self.x = x
        self.output = x @ self.weights
        print("\nforward:\n", self.output)

    def backward(self, in_gradient):
        self.in_gradient = in_gradient
        self.out_gradient = self.x.T @ self.in_gradient
        print("\ngradient:\n", self.out_gradient)


# reproducibility
np.random.seed(1337)
# init
size1 = (2, 2)
L1 = layer(size1)
# forward pass
x = np.random.randint(0, 10, size=(2, 1)).T
L1(x)
# backprops
in_gradient = np.random.uniform(-1., 1., size=x.shape)
L1.backward(in_gradient)

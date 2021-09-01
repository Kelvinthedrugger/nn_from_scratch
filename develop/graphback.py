"""
# useful: https://towardsdatascience.com/computational-graphs-in-pytorch-and-tensorflow-c25cc40bdcd1
implement graph computation to achieve
auto-differentiation
from torch:
Tensor:                     Function
 grad            input -->  forward --> output
                               V
grad_fn     out_gradient <- backward <- in_gradient
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
        self.output = self.x @ self.weights
        print(self.weights, end="\n\n")
        print(self.output)
        assert self.output.shape == self.x.shape
        return self.output

    def backward(self, in_gradient):
        self.in_gradient = in_gradient
        print(self.in_gradient, end="\n\n")
        print(self.x, end="\n\n")
        print(self.weights, end="\n\n")
        self.out_gradient = self.x.T @ (self.in_gradient @ (self.weights.T))
        print(self.out_gradient)
        return self.out_gradient


class loss_fn:
    def __init__(self):
        self.loss = 0

    def mse(self, yhat, y):
        self.loss = np.square(np.subtract(yhat, y)).mean(axis=1)
        self.gradient = np.multiply(2, np.subtract(yhat, y)).mean(axis=1)
        return self.loss, self.gradient


MSE = loss_fn().mse
y = np.array([[1, 2, 3, 4, 5]])
yhat = np.array([[1, 2, 9, 4, 5]])
print(MSE(yhat, y))

"""
x -> L1 -> L2 -> yhat
grad -> dL2 -> dL1
"""
"""
# reproducibility
np.random.seed(1337)
# init
size1 = (2, 2)
L1 = layer(size1)
L2 = layer(size1)
# forward pass
x = np.random.randint(0, 10, size=(2, 1)).T
print("\nlayer1:\n")
print("input: ", x, end="\n\n")
x = L1(x)
print("\nx: \n\n", x)
print("\nlayer2:\n")
x = L2(x)
# backprops
in_gradient = np.random.uniform(-1., 1., size=x.shape)
print("\ndL2\n")
L2.backward(in_gradient)

print("\ndL1\n")
L1.backward(in_gradient)  # input was wrong
print("\ndL1 using mul\n\n", x.T @ in_gradient)
"""

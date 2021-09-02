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
        self.gradient = np.multiply(2, np.subtract(yhat, y))
        return self.loss, self.gradient/(self.gradient.sum(axis=1))


class optim:
    def __init__(self, weight, gradient):
        # go figure out better api
        self.weight = weight
        self.gradient = gradient

    def SGD(self, learning_rate):
        self.learning_rate = learning_rate
        self.weight -= self.learning_rate*self.gradient
        print(self.weight)
        return self.weight

    def Adam(self, learning_rate, alpha=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        alpha = alpha
        b1 = b1
        b2 = b2
        eps = eps
        m = 0
        v = 0
        t = 0  # loop counter
        while(t < 2e3):  # while gradient does not converge
            t += 1
            m = b1*m+(1-b1)*self.gradient
            v = b1*v+(1-b1)*self.gradient**2
            mhat = m/(1-b1**t)
            vhat = v/(1-b2**t)
            self.gradient -= alpha*mhat/(vhat**0.5+eps)

        self.weight -= self.learning_rate*self.gradient
        print(self.weight)
        return self.weight


"""
x -> L1 -> L2 -> yhat
grad -> dL2 -> dL1
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
print("\nlayer2:\n")
x = L2(x)
# loss
MSE = loss_fn().mse
y = np.array([[3, 4, ]])
loss, gradient = MSE(x, y)
# backprops
#in_gradient = np.random.uniform(-1., 1., size=x.shape)
print("\ndL2\n")
# L2.backward(in_gradient)
dL2 = L2.backward(gradient)

print("\ndL1\n")
# L1.backward(in_gradient)
dL1 = L1.backward(gradient)  # weight was not returned
print("\nupdated\n")
optimizer = optim(L2.weights, dL2).SGD(1e-3)
optimizer = optim(L2.weights, dL2).Adam(1e-3)

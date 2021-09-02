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

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, x):
        """input row vector is much easier
        and computationally inexpensive"""
        #self.weights = layer_init(self.shape[0], self.shape[1])
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


"""
torch api:
# forward + backward + optimize
outputs = net(inputs)
loss = loss_function(outputs, labels)
loss.backward()
optimizer.step()
"""


class Model:
    """
    allocate all the layers without inherent from class: layer
    auto-diff with ease
    """

    def __init__(self):
        self.model = []

    def __call__(self, layers):
        # layers: list of layer
        for layer in layers:
            self.model.append(layer)

    def predict(self, x):
        for layer in self.model:
            x = layer(x)
        return x
    """i tried to do torch but now it looks like tf"""

    def compile(self, optimizer, lossf):
        # assign functions instead of classes
        # try to integrate weight inside optimizer directly
        # without taking another argument
        self.optimizer = optimizer
        self.lossf = lossf

    def fit(self, x, y, epochs=2):
        self.history = {"loss": [], "accuracy": []}
        for _ in epochs:
            yhat = self.predict(x)
            self.loss, self.gradient = self.lossf(yhat, y)
            self.optimizer([weight.weights for weight in self.model], [
                           grad for grad in self])
            self.history["loss"].append(self.loss.mean())
            self.history["accuracy"].append(
                (yhat == y).astype(np.float32).mean(axis=1))

        return self.history


class loss_fn:
    def __init__(self):
        self.loss = 0

    def mse(self, yhat, y):
        self.loss = np.square(np.subtract(yhat, y)).mean(axis=1)
        self.gradient = np.multiply(2, np.subtract(yhat, y))
        return self.loss, self.gradient/(self.gradient.sum(axis=1))


class optim:
    """original version"""

    # def __init__(self, weight, gradient):
    #     # go figure out better api
    #     self.weight = weight
    #     self.gradient = gradient

    # def SGD(self, learning_rate):
    #     self.learning_rate = learning_rate
    #     self.weight -= self.learning_rate*self.gradient
    #     print(self.weight)
    #     return self.weight

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def SGD(self, weight, gradient):
        self.weight = weight
        self.gradient = gradient
        self.weight -= self.learning_rate * self.gradient
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
        return self.weight


class Learner:
    """for better api, similar to model.compile() in tf"""
    # or just create a complete class called 'Model'
    # like tensorflow api

    def __init__(self, model, lossf, optimizer):
        """i don't care of using list but i guess i have to """
        # figure out how to assemble the layers
        # list.extend method could be useful
        self.model = model
        # loss function instead of the class
        self.lossf = lossf
        # function also
        self.optimizer = optimizer

    def __call__(self, x, y):
        """forward-backward"""
        yhat = self.model(x)
        self.lossf(yhat, y)
        self.optimizer()


"""
x -> L1 -> L2 -> yhat
grad -> dL2 -> dL1
"""
# reproducibility
np.random.seed(1337)

# init
# size1 = (2, 2)
# L1 = layer(size1)
# L2 = layer(size1)
l1 = layer_init(2, 2)
l2 = layer_init(2, 2)
L1 = layer(l1)
L2 = layer(l2)
# forward pass
x1 = np.random.randint(0, 10, size=(2, 1)).T
# print("\nlayer1:\n")
# print("input: ", x1, end="\n\n")
# x = L1(x1)
# print("\nlayer2:\n")
# x = L2(x)
# # loss
# MSE = loss_fn().mse
# y = np.array([[3, 4, ]])
# loss, gradient = MSE(x, y)
# print("loss and grad: ", loss, gradient)
# # backprops
# print("\ndL2\n")
# dL2 = L2.backward(gradient)
# print("\ndL1\n")
# dL1 = L1.backward(gradient)  # weight was not returned
# print("\nupdated\n")
# optimizer = optim(L2.weights, dL2).SGD(1e-3)
# optimizer = optim(L2.weights, dL2).Adam(1e-3)
# model
print("\nmodel\n")
model = Model()
model([L1, L2])

model.predict(x1)
print("\nweights\n")
print(L1.weights)
print(L2.weights)
learning_rate = 1e-3
optimizer = optim(learning_rate).SGD
# criterion = loss_fn.mse
# model.compile(optimizer, criterion)

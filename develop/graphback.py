"""
# useful: https://towardsdatascience.com/computational-graphs-in-pytorch-and-tensorflow-c25cc40bdcd1
auto-differentiation implemented using graph computation
from torch tensor:
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
        self.x = x
        self.output = self.x @ self.weights
        return self.output

    def backward(self, in_gradient):
        self.in_gradient = in_gradient
        # d_layer of the layer
        self.d_weight = self.x.T @ self.in_gradient
        # backprop grad to previous layer
        self.out_gradient = self.in_gradient @ (self.weights.T)
        # return self.d_weight, self.out_gradient


class activation:
    """
    e.g.
    activation = ReLU()
    model = Model()
    model([Dense, activation, ...])

    note: 
    activation doesn't have weights, so no update for it either
    """

    def __init__(self, weights):
        self.weights = weights

    # it's ReLu
    def __call__(self, x):
        # forward pass from last layer
        self.x = x
        # foward x to next layer
        """extremely weird"""
        # self.output = np.maximum(self.x, 0)
        self.output = self.x
        return self.output
        # return self.output, self.grad

    # ... add more activation function here

    def backward(self, in_gradient):
        """
        allocates tensor from the called activation function!
        """
        self.in_gradient = in_gradient
        # d_layer of the layer: it's activation!
        self.d_weight = 0
        # backprop grad to previous layer
        # implement gradient form
        """
        backward not done
        layer 2 did not updated
        """
        self.weights = (self.output > 0).astype(np.float32)
        # print("\nrelu: ", self.weights, end="\n\n")
        self.out_gradient = np.multiply(self.in_gradient, self.weights)


# np.random.seed(1337)
# x1 = np.random.uniform(-1., 1., size=(10,))
# l1 = layer_init(10, 5)
# y1 = x1 @ l1
# print("output: ", y1, end="\n\n")
# act1 = activation(l1)
# y11, dy11 = act1.ReLU(y1)
# print("after relu: ", y11, end="\n\n")
# print("gradient: ", dy11)


class Model:
    """
    torch api:
    # forward + backward + optimize
    outputs = net(inputs)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()
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

    def compile(self, optimizer, lossf):
        self.optimizer = optimizer
        self.lossf = lossf

    def fit(self, x, y, epochs=5):
        self.history = {"loss": [], "accuracy": []}
        self.d_weights = []
        for _ in range(epochs):
            # forward pass
            yhat = self.predict(x)
            print("yhat: ", yhat, end="\n\n")
            # print("y: ", y, end="\n\n")
            # loss, gradient of loss
            self.loss, self.gradient = self.lossf(self, yhat, y)
            # print(self.gradient)
            # backprop
            for weight in self.model[::-1]:
                print("\nweight", self.model.index(weight), end=" \n")
                print(weight.weights[0][:10])
                weight.backward(self.gradient)
                self.d_weights.append(weight.d_weight)
                self.gradient = weight.out_gradient
            # reverse back
            self.d_weights = self.d_weights[::-1]
            # weight update
            for weight in self.model:
                self.optimizer(
                    weight.weights, self.d_weights[self.model.index(weight)])
            # record
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
        # mean instead of normalise
        return self.loss, self.gradient/(self.gradient.shape[-1])


class optim:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def SGD(self, weight, gradient):
        self.weight = weight
        self.gradient = gradient
        self.weight -= self.learning_rate * self.gradient
        # return self.weight

    def Adam(self, weight, gradient, alpha=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.weight = weight
        self.gradient = gradient
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
        # return self.weight

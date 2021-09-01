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
    """
    pass


class CEloss:
    """
    crossentropy loss:
    loss.backward(): backprop
    """

    def __init__(self):
        self.loss = 0
        self.gradient = 0

    def __call__(self, labels, fpass, num_classes):
        # encoding
        self.labels = np.zeros((len(labels), num_classes), np.float32)
        self.labels[range(self.labels.shape[0]), labels] = 1
        print("labels: ", self.labels)  # wrong

        loce = -fpass + np.log(np.exp(fpass).sum(axis=1)).reshape((-1, 1))
        self.loss = (self.labels*loce).mean()  # loss: scalar
        dloss = self.labels/len(labels)
        self.gradient = -dloss + \
            np.exp(-loce)*dloss.sum(axis=1).reshape((-1, 1))  # gradient: vector
        print("loss: %.4f" % (self.loss))
        print("gradient: ", self.gradient)


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

    def __init__(self, shape):
        super(Dense, self).__init__(shape)

    def __call__(self, x):
        self.x = x
        return self.x @ self.weights


class optim:
    def __init__(self, weights, learning_rate):
        self.weights = weights
        self.learning_rate = learning_rate

    def sgd(self, gradient):
        """replace 'gradient' after backward() is done"""
        self.weights -= self.learning_rate*gradient
        print(self.weights)

    def adam(self, gradient, alpha=1e-3, b1=0.9, b2=0.999, eps=1e-8, t=0):
        self.gradient = gradient
        self.t = t  # time step
        self.alpha = alpha
        self.b1 = b1  # params
        self.b2 = b2
        self.eps = eps
        m = 0
        v = 0
        while(self.t < 2e3):  # while gradient does not converge
            self.t += 1
            m = self.b1*m+(1-self.b1)*self.gradient
            v = self.b1*v+(1-self.b1)*self.gradient**2
            mhat = m/(1-self.b1**self.t)
            vhat = v/(1-self.b2**self.t)
            self.gradient -= self.alpha*mhat/(vhat**0.5+self.eps)

        self.weights -= self.learning_rate*self.gradient
        print(self.weights)
        print(self.alpha*mhat/(vhat**0.5+self.eps))


# for reproducibility
np.random.seed(1337)
# layer
size1 = (784, 128)
L1 = layers(size1)
print(L1().shape)

# Dense: forward
tmp = np.random.uniform(-1., 1., size=(1, 784))
print(Dense(size1)(tmp).shape)

# SGD
l1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
dl1 = np.random.uniform(-1., 1., size=l1.shape)
print(l1)
optimizer = optim(l1, learning_rate=1e-3)
optimizer.sgd(dl1)
# Adam
l1 = np.array([1, 2, 3, 4, 5], dtype=np.float32)
dl1 = np.random.uniform(-1., 1., size=l1.shape)
print(l1)
optimizer.adam(dl1)
# CEloss
lossfn = CEloss()
y = np.array([[1, 2, 3, 4, 0, 1, 1, 1, 1, 1]], dtype=np.uint8)
yhat = np.array([[0, 4, 3, 2, 1, 1, 1, 1, 1, 1]], dtype=np.uint8)
lossfn(y, y, num_classes=10)
print("from helper: ", CE(y, y))

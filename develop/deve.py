from time import time
import matplotlib.pyplot as plt
import numpy as np
from helper import layer_init, CE, SGD, relu, act_df
from fetch_it import mnist

x_train, y_train, x_test, y_test = mnist()

"""looks like what ive done in testfile.py but it is temporary"""


def build_model(input_shape):
    """
    general method: like Sequential and more complex one
    """
    pass


def BobNet(x, layers=None, input_shape=None, act=act_df):
    x = x.reshape((-1, 28*28))
    if layers is not None:
        l1, l2 = layers[0], layers[1]
        return [x, x@l1, act(x@l1), x@l1@l2], [l1, l2]

    l1 = layer_init(784, 128)
    l2 = layer_init(128, 10)
    """when classmethod: figure how to pass weights automatically"""
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


def training(x, y, model, loss_fn, optimizer=SGD, batch_size=32, epoch=1000, x_t=None, y_t=None):
    losses = []
    test_losses = []
    _, layers = model(x[0])  # to establish 'layers'
    start = time()
    for _ in range(epoch):
        samp = np.random.randint(0, x.shape[0]-batch_size, size=(batch_size))
        X, Y = x[samp], y[samp]
        fpass, weights = model(X, layers, relu)
        prediction = fpass[-1]
        loss, grad = loss_fn(Y, prediction)

        X_t, Y_t = x_t[samp*(samp < len(x_t))], y_t[samp*(samp < len(x_t))]
        fpt, _ = model(X_t, layers, relu)
        prediction_t = fpt[-1]
        loss_t, _ = loss_fn(Y_t, prediction_t)

        # target: automate [update_weight] -> updated model
        gradient = backward(grad, weights, fpass)
        update_weight = optimizer(gradient, weights, 1e-4)
        layers = update_weight

        losses.append(loss)
        test_losses.append(loss_t)
    end = time()
    print("time: %.4f sec" % (end-start))
    print("loss: %.3f" % min(losses))
    print("test loss: %.3f" % min(test_losses))
    plt.plot(losses)
    plt.plot(test_losses)
    plt.show()
    return layers


model = BobNet
loss_fn = CE
optimizer = SGD
batch_size = 32

weights = training(x_train, y_train, model, loss_fn,
                   optimizer, batch_size, epoch=300, x_t=x_test, y_t=y_test)

# testing
batch_size = 32
accus = []
for i in range(300):
    output, _ = model(x_test[i:i+batch_size], weights, relu)
    pred = output[-1]
    accus.append(
        (pred.argmax(axis=1) == y_test[i:i+batch_size]).astype(np.float32).mean())

print("test accuracy: %.3f" % (sum(accus)/len(accus)))

plt.plot(accus)
plt.title("with relu")
plt.legend(["training loss", "test loss", "test accuracy"])
plt.show()

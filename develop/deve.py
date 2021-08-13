from time import time
import matplotlib.pyplot as plt
import numpy as np
from helper import layer_init, CE, SGD, relu, act_df, kernel_L1, kernel_L2
from dev_func import dropout  # real deal right here
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


def training(x, y, model, loss_fn, optimizer=SGD, batch_size=32, epoch=1000, x_t=None, y_t=None, kernel_regularizer=None, early_stops=False, patience=5, to_dropout=None):
    losses = []
    test_losses = []
    _, layers = model(x[0])  # to establish 'layers'
    start = time()
    for i in range(epoch):
        samp = np.random.randint(0, x.shape[0]-batch_size, size=(batch_size))
        X, Y = x[samp], y[samp]
        fpass, weights = model(X, layers, relu)
        # add dropout here
        # dropout first layer only right now
        # error occurred here
        if to_dropout is not None:
            weights[to_dropout], storage_0 = dropout(
                weights[to_dropout], storage=(None if i == 0 else storage_0), prob=.2)

            # weights[1], storage_1 = dropout(weights[1], storage=(
            #     None if i == 0 else storage_1), prob=.2)

        prediction = fpass[-1]
        loss, grad = loss_fn(Y, prediction)
        if kernel_regularizer is not None:
            grad = kernel_regularizer(grad, weights, 5e-4)

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

        if early_stops:
            if losses.index(losses[-1]) - losses.index(min(losses)) > patience:
                print("\nstops at epoch: %d" % losses.index(losses[-1]))
                break
        """print(weights[0][:10]) # restored successfully"""

    end = time()
    print("time: %.4f sec" % (end-start))

    print("loss: %.3f" % (losses[-1]))
    print("test loss: %.3f" % (test_losses[-1]))
    plt.plot(losses)
    plt.plot(test_losses)

    return layers


model = BobNet
loss_fn = CE
optimizer = SGD
batch_size = 32

weights = training(x_train, y_train, model, loss_fn,
                   optimizer, batch_size, epoch=100, x_t=x_test, y_t=y_test, kernel_regularizer=kernel_L2, early_stops=True, patience=20, to_dropout=0)

# testing
batch_size = 32  # handy
accus = []
for i in range(1000):
    output, _ = model(x_test[i:i+batch_size], weights, relu)
    pred = output[-1]
    accus.append(
        (pred.argmax(axis=1) == y_test[i:i+batch_size]).astype(np.float32).mean())

print("test accuracy: %.3f" % (sum(accus)/len(accus)))


plt.plot(accus)
plt.title("with relu")
plt.legend(["training loss", "test loss", "test accuracy"])
plt.show()

"""
plt.figure(figsize=(12,6))
plt.rc('font',size=25)
plt.imshow(weights[0].reshape((392,256)))
plt.title("1st layer")
#plt.show()
"""
"""
plt.figure(figsize=(12, 7))
plt.rc('font', size=25)
plt.imshow(weights[1].reshape((32, 40)), cmap='gray')
plt.title("2nd layer")
plt.show()
visual = weights[1]
assert visual[0:10].shape == (10, 10)
plt.imshow(visual[0:10], cmap='gray')
plt.show()
#plt.imsave("layer_2.jpg", (weights[1].reshape((32, 40))))
"""

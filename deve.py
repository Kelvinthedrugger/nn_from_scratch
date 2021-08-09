from frame import layer_init, Dense
import numpy as np
from fetch_it import mnist

x_train, y_train, x_test, y_test = mnist()

assert x_train.shape[0] == 6e4
"""move succeed building blocks to 'frame'"""


def model(input_shape):
    """
    input_layer = x.reshape((-1, 28*28))  # flatten layer
    add shape control to deal with batched data: np.expand_dims -> np.reshape
    """
    layer1 = Dense(input_shape, 128)  # (784,128)
    layer2 = Dense(layer1.shape, 10)  # (128,10)
    return [layer1, layer2]


def forward(x, layers):
    """
    for evaluate and if you want forward only
    y is for evaluation so we don't need it here
    """
    output = x@layers[0]
    for i in range(1, len(layers)):
        output = output@layers[i]

    assert output.shape == (x.shape[0], 10)
    return output


batch_size = 2

Bob = model(x_train[0:batch_size].reshape((-1, 28*28)).shape)
tt = forward(x_train[0:batch_size].reshape(
    (-1, 28*28)), Bob)
print("forward: ", tt[0].argmax(), ", ", tt[1].argmax())


def sgd(layers, lr=1e-3):
    layers = layers[1] - lr*layers[0]


def CE(y, yhat):
    """crossentropy loss"""
    out = np.zeros((len(y), 10), dtype=np.float32)
    out[range(len(y)), y] = 1  # encoding label
    assert out.shape == yhat.shape
    losce = yhat - np.log(np.exp(yhat).sum(axis=1))  # ()
    loss = -out*losce.mean(axis=1)  # (bs,) almost a scalar
    grads = -out/len(y) + np.exp(-losce)*out.mean(axis=1)
    return loss, grads


def backward(y, yhat, layers, loss_fn=CE, optimizer=sgd):
    # return loss and gradient
    loss, grads = loss_fn(y, yhat)

    # backprop the grads to layers
    """
    concatenate d_layers into layers:

    layers->list(layers)->layers.append(d_layers[i])
    -> np.array(layers)->layers.reshape((2,num_of_layers))
    """
    # weight update
    optimizer(layers)

    # return weight
    return loss.mean(), layers

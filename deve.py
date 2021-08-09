from frame import layer_init, Dense
import numpy as np
from fetch_it import mnist

x_train, y_train, x_test, y_test = mnist()

assert x_train.shape[0] == 6e4


def model(input_shape):
    # input_layer = x.reshape((-1, 28*28))  # flatten layer
    layer1 = Dense(input_shape, 128)  # (784,128)
    layer2 = Dense(layer1.shape, 10)  # (128,10)
    return [layer1, layer2]


def forward(x, y, layers):
    """
    for evaluate and if you want forward only
    y is for evaluation: so far we don't need it
    """
    output = x@layers[0]
    for i in range(1, len(layers)):
        output = output@layers[i]

    assert output.shape == (x.shape[0], 10)
    return output


batch_size = 2

Bob = model(x_train[0:batch_size].reshape((-1, 28*28)).shape)
tt = forward(x_train[0:batch_size].reshape(
    (-1, 28*28)), y_train[0:batch_size], Bob)
print("forward: \n", tt[0].argmax(), ", ", tt[1].argmax())


def backward(input, output, layers):
    pass

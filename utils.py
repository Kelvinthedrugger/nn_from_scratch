import numpy as np
from frame import Dense, Relu


def model(input_shape):
    """
    input_layer = x.reshape((-1, 28*28))  # flatten layer
    add shape control to deal with batched data: np.expand_dims -> np.reshape
    integrate activation function
    """
    layer1 = Dense(input_shape, 128)  # (784,128)
    layer2 = Dense(layer1.shape, 10)  # (128,10)
    return [layer1, layer2]


def forward(x, layers):
    """
    for evaluate and if you want forward only
    y is for evaluation so we don't need it here
    """
    fpass = []  # record forward pass for backprop
    fpass.append(x)  # append input
    for i in range(len(layers)):
        x = x @ layers[i]
        fpass.append(x)
        if i == 0:
            x = Relu(x)
            fpass.append(x)
    return x, fpass


def CE(y, yhat):
    """
    crossentropy loss
    y: (bs,)
    yhat: (bs,num_classes)
    """
    out = np.zeros((len(y), 10), dtype=np.float32)
    out[range(out.shape[0]), y] = 1  # encoding label
    assert out.shape == yhat.shape
    losce = -yhat + np.log(np.exp(yhat).sum(axis=1)).reshape((-1, 1))
    loss = (out*losce).mean(axis=1)  # (bs,) almost a scalar
    dout = -out/len(y)
    grad = dout - np.exp(-losce)*dout.sum(axis=1).reshape((-1, 1))
    return loss, grad


def backward(y, yhat, layers, fpass, loss_fn=CE):
    # return loss and gradient
    loss, grad = loss_fn(y, yhat)

    d_layers = []
    # d_l2
    d_layers.append(fpass[2].T @ grad)
    # d_l1
    d_x = (fpass[2] > 0).astype(np.float32) * (grad @ (layers[1].T))
    d_layers.append(fpass[0].T @ d_x)

    # backprop the grads to layers
    """
    concatenate d_layers into layers:

    layers->list(layers)->layers.append(d_layers[i])
    -> np.array(layers)->layers.reshape((2,num_of_layers))
    """
    # weight update

    #update_layer = optimizer(layers, d_layers)

    # return weight
    return loss.mean(), layers, d_layers


def sgd(layers, d_layers, lr=1e-3):
    return np.array(layers) - lr*np.array(d_layers[::-1])

import numpy as np


def layer_init(h, w, batch_size=None):
    if batch_size is not None:
        return np.random.uniform(-1., 1., size=(batch_size, h, w))/np.sqrt(h*w)
    return np.random.uniform(-1., 1., size=(h, w))/np.sqrt(h*w)


"""activation"""


def relu(x):
    return np.maximum(x, 0)


"""loss functions: return loss, gradient"""


def MSE(y, yhat):
    """unsupervised: check shape"""
    loss = np.square(np.subtract(y, yhat))  # matrix
    h, w = y.shape  # no batchsize case first
    grads = 2/(h*w)*np.subtract(y, loss)
    return loss.mean(), grads


def CE(y, yhat):
    """
    categorical cross entropy
    , sparse i guess since the zeros
    """
    # encoding
    out = np.zeros((len(y), 10), np.float32)  # (batchsize, num_classes)
    out[range(out.shape[0]), y] = 1

    loce = -yhat + np.log(np.exp(yhat).sum(axis=1)).reshape((-1, 1))
    loss = (out*loce).mean(axis=1)  # (batchsize,)
    dout = out/len(y)
    grads = -dout + np.exp(-loce)*dout.sum(axis=1).reshape((-1, 1))

    return loss.mean(), grads


"""[update_weight] = optimizer(gradient,weight,learning_rate)"""


def SGD(grad, weight, lr=1e-5):
    # return list of tensors
    assert len(grad) == len(weight)
    for i in range(len(grad)):
        weight[i] -= lr*grad[i]

    return weight

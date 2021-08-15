import numpy as np


def build_model(input_shape):
    """
    general method: like Sequential and more complex one
    """
    pass


def layer_init(h, w, batch_size=None):
    if batch_size is not None:
        return np.random.uniform(-1., 1., size=(batch_size, h, w))/np.sqrt(h*w)
    return np.random.uniform(-1., 1., size=(h, w))/np.sqrt(h*w)


"""activation"""


def act_df(x):
    # defalut activation function
    return x


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


"""kernel regularizer"""


def kernel_L1(grad, weights, ratio=1e-4):
    for i in range(len(weights)):
        # penalize the loss
        grad -= ratio * weights[i].sum()
    return grad


def kernel_L2(grad, weights, ratio=1e-3):
    for i in range(len(weights)):
        # penalize the loss
        grad -= ratio * np.square(weights[i]).sum()
    return grad


def dropout(layer, nth_layer=0, prob=0.2, storage=None):
    """   # DON'T SHOW THESE BESIDES TESTING IN THIS PROGRAM
    testlayer = np.random.uniform(-10., 10., size=(5, 5)).astype(np.float32)
    # training loop
    for i in range(5):
        print("\nepoch: %d" % (i+1))
        print("\nbefore: \n\n", testlayer)
        after_dropout, storage = dropout(
            testlayer, storage=(None if i == 0 else storage), prob=.5)
        print("\nafter: \n\n", after_dropout)
        print("\n", storage)
    """
    if storage is None:  # first epoch in training
        """probably we can improve this argument"""
        storage = {}  # {ith, value}
    """maybe one layer at a time is fine"""
    if isinstance(nth_layer, int):

        for i in range(len(layer)):
            # can be optimized with just looking at the first element, kinda buggy
            if layer[i].all() == 0.:
                layer[i] = storage[i]  # restore the layers

        storage = {}  # release the memory

        for i in range(len(layer)):
            if np.random.uniform() < prob:
                if not layer[i].all() == 0.:
                    storage[i] = np.array(list(layer[i]), dtype=np.float32)
                #print("\nindex: ", i, " should store: ", layer[i])
                layer[i] *= 0.
    return layer, storage

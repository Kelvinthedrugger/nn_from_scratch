import numpy as np
from fetch_it import mnist

x_train, y_train, x_test, y_test = mnist()

"""
option 1: return value from every function: done in testfile.py
option 2: do operation mostly, which saves memory
option 3: do both, might be messy but api will be more elegant
"""


def training(epoch=1000):
    """the high level api"""
    """
    for i in range(epoch):
        prediction = forward(x,layers)
        loss, [update_weight] = backward()
        # [update_weight] -> updated model
    """
    pass


def layer_init(h, w, batch_size=None):
    if batch_size is not None:
        return np.random.uniform(-1., 1., size=(batch_size, h, w))/np.sqrt(h*w)
    return np.random.uniform(-1., 1., size=(h, w))/np.sqrt(h*w)


def build_model(input_shape):
    pass


def forward():
    pass


def backward():
    pass

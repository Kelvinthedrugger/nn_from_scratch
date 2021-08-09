import numpy as np


def layer_init(h, w):
    """initialization with size (h,w)"""
    weights = np.random.uniform(-1., 1., size=(h, w))/np.sqrt(h*w)
    return weights.astype(np.float32)


"""model = Sequential([
    layer1(input_shape,units,),
    actvation_1(),
    layer2,
    layer3,
    ...,
    layer_final
])
layer type: list(output, grad)
"""


def Dense(input_shape, units):
    _, w = input_shape
    return layer_init(w, units)


def Relu(x):
    return np.maximum(x, 0)

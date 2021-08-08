import numpy as np


def layer_init(h, w):
    """initialization with size (h,w)"""
    weights = np.random.uniform(-1., 1., size=(h, w))/np.sqrt(h*w)
    return weights.astype(np.float32)

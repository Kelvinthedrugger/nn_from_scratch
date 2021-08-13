"""used to test functions that are more complicated"""
import numpy as np


def dropout(layer, nth_layer=0, prob=0.2, storage=None):

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


"""   # DON'T SHOW THESE BESIDES TESTING IN THIS PROGRAM
testlayer = np.random.uniform(-10., 10., size=(5, 5)).astype(np.float32)
# training loop
for i in range(5):
    print("\nepoch: %d" % (i+1))
    print("\nbefore: \n\n", testlayer)
    after_dropout, storage = dropout(
        testlayer, storage=(None if i == 0 else storage), prob=.5)
    print("\nafter: \n\n", after_dropout)
    print("\n", storage)"""

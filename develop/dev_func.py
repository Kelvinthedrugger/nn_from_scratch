"""use to test functions that are more complicated"""
import numpy as np


def dropout(layer, nth_layer=0, prob=0.2, storage=None):
    """# try to prevent stroage got cleaned if unwanted"""
    # idk how to append value w/ dict
    if storage is None:  # first epoch in training
        storage = {}  # {ith, value}
    # only one layer is assigned
    """maybe one layer is fine"""
    if isinstance(nth_layer, int):
        for i in range(len(layer)):
            if layer[i].all() == 0.:
                pass
                "restore it using storage"
            if np.random.uniform() < prob:
                # if not layer[i].all() == 0.:
                storage[i] = layer[i]
                print("\nindex: ", i, " should store: ", layer[i])
                #layer[i] *= 0.
                if storage[i].all() != 0.:
                    layer[i] *= 0.
    return layer, storage


testlayer = np.random.uniform(-10., 10., size=(5, 5))
"""
for i in range(1):  # training loop
    print("\nepoch: %d" % (i+1))
    print("\nbefore: \n\n", testlayer)
    after_dropout, storage = dropout(
        testlayer, storage=(None if i == 0 else storage))
    print("\nafter: \n\n", after_dropout)

print("\n\n", storage)"""
store = {}
testlayer = np.random.uniform(-10., 10., size=(5, ))
if np.random.uniform() < 0.8:
    print(testlayer)
    store[0] = testlayer[0]
    print("\nindex: ", 0, " should store: ", testlayer[0])
    testlayer[0] *= 0.
    print(testlayer)

print(store)

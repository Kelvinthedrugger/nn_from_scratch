"""use to test functions that are more complicated"""
import numpy as np


"""quick guide
store = {}  # dict is essentially dixk
testlayer = np.random.uniform(-10., 10., size=(5, 5)).astype(np.float32)
if np.random.uniform() < 0.8:
    print(testlayer)
    store[0] = np.array(list(testlayer[0]), dtype=np.float32)
    print("\nindex: ", 0, " should store: ", testlayer[0])
    testlayer[0] *= 0.
    print(testlayer, type(testlayer[0]))
print(store, type(store[0]))
"""


def dropout(layer, nth_layer=0, prob=0.2, storage=None):
    """# try to prevent stroage got cleaned if unwanted"""
    # idk how to append value w/ dict
    if storage is None:  # first epoch in training
        storage = {}  # {ith, value}
    # only one layer is assigned
    """maybe one at a time is fine"""
    if isinstance(nth_layer, int):
        for i in range(len(layer)):
            if layer[i].all() == 0.:
                pass
                "restore it using storage"
            if np.random.uniform() < prob:
                if not layer[i].all() == 0.:
                    storage[i] = np.array(list(testlayer[i]), dtype=np.float32)
                print("\nindex: ", i, " should store: ",
                      testlayer[i])
                testlayer[i] *= 0.
    return layer, storage


testlayer = np.random.uniform(-10., 10., size=(5, 5)).astype(np.float32)
for i in range(1):  # training loop
    print("\nepoch: %d" % (i+1))
    print("\nbefore: \n\n", testlayer)
    after_dropout, storage = dropout(
        testlayer, storage=(None if i == 0 else storage))
    print("\nafter: \n\n", after_dropout)

print("\n\n", storage)

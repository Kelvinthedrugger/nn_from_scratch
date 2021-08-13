"""used to test functions that are more complicated"""
import numpy as np


def dropout(layer, nth_layer=0, prob=0.2, storage=None):
    """# try to prevent stroage got cleaned if unwanted"""
    if storage is None:  # first epoch in training
        """probably we can improve this argument"""
        storage = {}  # {ith, value}
    """maybe one layer at a time is fine"""
    if isinstance(nth_layer, int):
        """
        # restore process
        for weight in layer:
            if the layer is all 0.
                fill in the original weight according to index
        """
        for i in range(len(layer)):
            # can be optimized with just looking at the first element
            # and now i'd like to make sure everything is fine
            if layer[i].all() == 0.:
                layer[i] = storage[i]  # restore the layers
        storage = {}  # release the memory
        """storage gets to exist unless we clear it, since we choose to feed it in the function"""
        for i in range(len(layer)):
            if np.random.uniform() < prob:
                # probably don't need this once we figure out the storage release problem
                if not layer[i].all() == 0.:
                    storage[i] = np.array(list(layer[i]), dtype=np.float32)
                #print("\nindex: ", i, " should store: ", layer[i])
                layer[i] *= 0.
    return layer, storage


"""
testlayer = np.random.uniform(-10., 10., size=(5, 5)).astype(np.float32)
# training loop
for i in range(5):
    print("\nepoch: %d" % (i+1))
    print("\nbefore: \n\n", testlayer)
    after_dropout, storage = dropout(
        testlayer, storage=(None if i == 0 else storage), prob=.5)
    print("\nafter: \n\n", after_dropout)
    print("\n", storage)"""

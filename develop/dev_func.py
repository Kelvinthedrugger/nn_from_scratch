"""use to test functions that are more complicated"""
import numpy as np

"""
add dropout

deactivate units in the specified layer with probability = prob

for ith unit in layer1:
 # requires storage:(i, ith unit)
 if ith unit = 0: # previously dropped out
  ith unit = storage(i, ith unit)
 index = uniform(0,1)
 if index < prob:
  unit = 0
"""


def dropout(layer, nth_layer=0, prob=0.2, storage=None):
    # try to prevent stroage got cleaned if unwanted
    # idk how to append value w/ dict
    if storage is None:  # first epoch in training
        storage = {}  # {ith, value}
    # only one layer is assigned
    if isinstance(nth_layer, int):
        for i in range(len(layer)):
            if layer[i].all() == 0.:
                pass
                "restore it using storage"
            if np.random.uniform() < prob:
                # if not layer[i].all() == 0.:
                storage[i] = layer[i]
                print("\nindex: ", i, " should store: ", layer[i])
                layer[i] *= 0.
    return layer, storage


testlayer = np.random.uniform(-10., 10., size=(5, 5))

for i in range(5):  # training loop
    print("\nepoch: %d" % (i+1))
    print("\nbefore: \n\n", testlayer)
    after_dropout, storage = dropout(
        testlayer, storage=(None if i == 0 else storage))
    print("\nafter: \n\n", after_dropout)
    #print("\nstorage: \n\n", storage)
    # if not storage == {}:
    #     for j in range(5):  # it's gonna be slow asf
    #         if testlayer[j][0] == 0.:
    #             print(storage[j])  # indexing is problematic
print(storage)

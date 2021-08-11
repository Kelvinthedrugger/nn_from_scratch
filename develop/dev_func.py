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


def dropout(layer, nth_layer, prob=0.2):
    pass
    # try to prevent stroage got cleaned if unwanted
    # idk how to append value w/ dict
    storage = {}  # {ith, value}
    # only one layer is assigned
    if isinstance(nth_layer, int):
        pass
        for i in range(len(layer)):
            if layer[i].all() == 0.:
                pass
                "restore it using storage"
            if np.random.uniform() < prob:
                storage[i] = layer[i]
                layer[i] *= 0.

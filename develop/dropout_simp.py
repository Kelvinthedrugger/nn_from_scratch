import numpy as np
"""quick demo of how dropout works and why using dict doesn't"""
prob = 1.0  # set high probability for demonstration purpose
store = {}
testlayer = np.random.uniform(-10., 10., size=(5, 5)).astype(np.float32)
print("\nusing list conversion\n")
if np.random.uniform() < prob:
    print(testlayer)
    store[0] = np.array(list(testlayer[0]), dtype=np.float32)
    print("\nindex: ", 0, " should store: ", testlayer[0])
    testlayer[0] *= 0.
    print("\n", testlayer, type(testlayer[0]))
print("\n", store, type(store[0]))

print("\nbelow used dict\n")
store = {}
testlayer = np.random.uniform(-10., 10., size=(5, 5)).astype(np.float32)
if np.random.uniform() < prob:
    print(testlayer)
    store[0] = testlayer[0]  # where things got wrong
    print("\nindex: ", 0, " should store: ", testlayer[0])
    testlayer[0] *= 0.
    print("\n", testlayer, type(testlayer[0]))
print("\n", store, type(store[0]))
print("\ndid you see how dict runs by reference mode and changes accordingly which caused undesired result\n \
maybe im just too dump to use pointers tho there should be no memory management needed in python")

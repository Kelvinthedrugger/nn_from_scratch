"""used to develop functions that are more complicated"""
import numpy as np

"""
implement convolution operation first:

# a_total: input: n*n*bs array (assume bs=1 right now)
# b: filter: 3 x 3 layer (for a single filter, actually m x m)
# a: 3 x 3 layer (same as filter) from input layer

-> multiplying element-wisely and summation: np.multiply(a,b).sum() # scalar
-> pass the product at the center of filter position: cause shape reduction

output: (n-2) x (n-2) x units array (3-1= 2)
"""


# input: a single layer
def conv_op(layer, units=1, kernel_size=3):
    """figure out how to do fast approximation instead of n^3 complexity piece of crap"""
    # go figure the shape problem more concisely
    h, w = layer.shape
    # use layer_init to replace at the end
    conv_layer = np.random.uniform(-1., 1.,
                                   size=(kernel_size, kernel_size, units))
    for r in range(units):
        for k in range(h-kernel_size+1):  # 5-3+1
            for m in range(w-kernel_size+1):
                partial = np.array([layer[i+k, j+m] for i in range(3)
                                    for j in range(3)], dtype=np.float32).reshape((3, 3))
                print("\npartial:\n", partial)
                mul = np.multiply(partial, conv_layer[:, :, r]).sum()
                print("\nmultiplication: %.4f" % (mul))


"""
np.random.seed(1337)  # for reproducibility
# test below
test = np.array(list(range(25)), dtype=np.float32).reshape((5, 5))
conv_op(test)
"""

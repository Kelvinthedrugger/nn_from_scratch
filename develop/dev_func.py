"""used to develop functions that are more complicated"""
import numpy as np

"""
tensorflow api: 
 input_shape = (28,28)
 kernelsize = 3
 stride = 1 (by default)
 filters = 24
 conv2d.shape (3,3,1,24); 
"""


def conv_op(layer, units=1, kernel_size=3):
    """figure out how to do fast approximation instead of n^3 complexity piece of crap"""
    h, w = layer.shape
    # use layer_init to replace at the end
    conv_layer = np.random.uniform(-1., 1.,
                                   size=(kernel_size, kernel_size, units))

    r_size = (kernel_size - 1) // 2 + 1
    result = np.zeros((units, h-r_size, w-r_size), dtype=np.float32)
    for r in range(units):
        for k in range(h-kernel_size+1):  # 5-3+1
            for m in range(w-kernel_size+1):
                partial = np.array([layer[i+k, j+m] for i in range(3)
                                    for j in range(3)], dtype=np.float32).reshape((3, 3))
                mul = np.multiply(partial, conv_layer[:, :, r]).sum()
                # print("\npartial:\n", partial)
                # print("\nmultiplication: %.4f" % (mul))
                result[r, k, m] = mul
    return result, conv_layer


def conv_back(fpass, weights):
    """implement backprop of conv layers: can combine with conv_op()"""
    # assume no loss function; i.e. gradient of loss = identity
    gradient = np.zeros(fpass.shape, dtype=np.float32)

    for i in range(fpass.shape[0]):
        gradient[i] = fpass[i].T @ weights[:, :, i]

    return gradient


np.random.seed(1337)  # for reproducibility
# test below
test = np.array(list(range(25)), dtype=np.float32).reshape((5, 5))
print("\ntest:\n\n", test)
output, weights = conv_op(test, units=2)
print("\nresult:\n\n", output)

print("\ngradient\n\n", conv_back(output, weights))

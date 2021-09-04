"""testfile of graphback.py"""
from time import time
import numpy as np
from helper import layer_init
from graphback import layer, Model, loss_fn, optim, activation

# reproducibility
np.random.seed(1337)

# init
l1 = layer_init(784, 128)
l2 = layer_init(128, 10)
L1 = layer(l1)
L2 = layer(l2)
ReLu = activation(l1)
# forward pass
x1 = np.random.randint(0, 10, size=(784, 1)).T
y1 = np.random.randint(0, 10, size=(10, 1)).T
# model
model = Model()
#model([L1, L2])
model([L1, ReLu, L2])
learning_rate = 1e-2
optimizer = optim(learning_rate).Adam
criterion = loss_fn.mse
model.compile(optimizer, criterion)
start = time()
hist = model.fit(x1, y1)
end = time()
print("\nloss: ", hist["loss"], "\n\naccuracy: ", hist["accuracy"])
print("\ndue to small input size, overfitting occurs\n\nnow, it's slow: %.3fs" % (end-start))
print("\n we should refactor label encoding as well")
# print("\nlayer1:\n")
# print("input: ", x1, end="\n\n")
# x = L1(x1)
# print("\nlayer2:\n")
# x = L2(x)
# # loss
# MSE = loss_fn().mse
# y = np.array([[3, 4, ]])
# loss, gradient = MSE(x, y)
# print("loss and grad: ", loss, gradient)
# # backprops
# print("\ndL2\n")
# dL2 = L2.backward(gradient)
# print("\ndL1\n")
# dL1 = L1.backward(gradient)  # weight was not returned
# print("\nupdated\n")
# optimizer = optim(L2.weights, dL2).SGD(1e-3)
# optimizer = optim(L2.weights, dL2).Adam(1e-3)

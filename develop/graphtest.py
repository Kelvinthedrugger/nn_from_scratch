"""testfile of graphback.py"""
from time import time
import numpy as np
from helper import layer_init
from graphback import layer, Model, loss_fn, optim, activation
from fetch_it import mnist
# reproducibility
np.random.seed(1337)

# init
l1 = layer_init(784, 128)
l2 = layer_init(128, 10)
L1 = layer(l1)
L2 = layer(l2)
ReLu = activation(l1)

x_train, y_train, _, _ = mnist()
x1 = x_train[0:5].reshape((5, 784))
y1 = y_train[0:5]

model = Model()
model([L1, ReLu, L2])
learning_rate = 1e-3
optimizer = optim(learning_rate).Adam
criterion = loss_fn.ce
model.compile(optimizer, criterion)

start = time()
hist = model.fit(x1, y1)
end = time()
print("\nloss: ", hist["loss"], "\n\naccuracy: ", hist["accuracy"])
print("\ndue to small input size, overfitting occurs\n\nnow, it's slow: %.3fs" % (end-start))

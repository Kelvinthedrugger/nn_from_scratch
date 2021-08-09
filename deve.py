import numpy as np
from fetch_it import mnist
from frame import layer_init, Dense, Relu
from utils import model, forward, backward, sgd
import matplotlib.pyplot as plt
from tqdm import tqdm

x_train, y_train, x_test, y_test = mnist()

assert x_train.shape[0] == 6e4
"""move succeed building blocks to 'frame'"""

batch_size = 32

losses = []
m_shape = x_train[0:batch_size].reshape((-1, 28*28)).shape
Bob = model(m_shape)
for i in tqdm(range(1000)):
    # forward
    tt, fpass = forward(x_train[i:i+batch_size].reshape((-1, 28*28)), Bob)
    # backprop
    loss, layers, d_layers = backward(y_train[i:i+batch_size], tt, Bob, fpass)

    layers_new = sgd(layers, d_layers)
    ll1, ll2 = layers_new[0], layers_new[1]

    Bob = [ll1, ll2]
    assert ll1.shape == (784, 128)
    assert ll2.shape == (128, 10)

    losses.append(loss)


plt.plot(losses)
plt.show()
plt.title("training loss")
print(losses[-1])
accus = []
for i in tqdm(range(100)):
    pred, _ = forward(x_test[i:i+batch_size].reshape((-1, 28*28)), Bob)
    accus.append(
        (pred.argmax() == y_test[i:i+batch_size]).astype(np.float32).sum())

print("%.3f" % (sum(accus)/len(accus)))

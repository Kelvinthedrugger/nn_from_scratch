import numpy as np
from fetch_it import mnist
from frame import layer_init, Dense, Relu
from utils import model, forward, backward, sgd
import matplotlib.pyplot as plt
from tqdm import tqdm

x_train, y_train, x_test, y_test = mnist()

assert x_train.shape[0] == 6e4

# hyperparameters
batch_size = 32
learning_rate = 1e-3

# create model
model_shape = x_train[0:batch_size].reshape((-1, 28*28)).shape
Bob = model(model_shape)

# training
losses = []
for i in tqdm(range(1000)):
    # forward
    tt, fpass = forward(x_train[i:i+batch_size].reshape((-1, 28*28)), Bob)
    # backprop
    loss, layers, d_layers = backward(y_train[i:i+batch_size], tt, Bob, fpass)
    """
    layers_new = sgd(layers, d_layers)
    ll1, ll2 = layers_new[0], layers_new[1]

    Bob = [ll1, ll2]
    assert ll1.shape == (784, 128)
    assert ll2.shape == (128, 10)
    """
    Bob = sgd(layers, d_layers, lr=learning_rate)

    losses.append(loss)
plt.plot(losses)
plt.title("training loss")
plt.show()
print("last loss value: %.3f" % losses[-1])

# testing
accus = []
for i in range(100):
    pred, _ = forward(x_test[i:i+batch_size].reshape((-1, 28*28)), Bob)
    accus.append(
        (pred.argmax(axis=1) == y_test[i:i+batch_size]).astype(np.float32).mean())

print("test accuracy: %.3f" % (sum(accus)/len(accus)))

import matplotlib.pyplot as plt
import numpy as np
from helper import layer_init, CE, SGD
from fetch_it import mnist

x_train, y_train, x_test, y_test = mnist()

"""looks like what ive done in testfile.py but it is temporary"""


"""
option 1: return value from every function: done in testfile.py
option 2: do operation mostly, which saves memory
option 3: do both, might be messy but api will be more elegant
"""


def build_model(input_shape):
    """
    general method: like Sequential and more complex one
    """
    pass


"""
functional api

inputs = tf.keras.Input(shape=(3,))
x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)


or, by subclass method

class MyModel(tf.keras.Model):

  def __init__(self):
    super(MyModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
    self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
    self.dropout = tf.keras.layers.Dropout(0.5)

  def call(self, inputs, training=False):
    x = self.dense1(inputs)
    if training:
      x = self.dropout(x, training=training)
    return self.dense2(x)

model = MyModel() # build it


low-level training

for idx in tqdm(range(0, train_images.shape[0], batch_size)):

  (images, labels) = (
      train_images[idx:idx+batch_size], train_labels[idx:idx+batch_size])
  images = tf.convert_to_tensor(images, dtype=tf.float32)

  with tf.GradientTape() as tape:
    logits = cnn_model(images)
    loss_value = tf.keras.backend.sparse_categorical_crossentropy(
        labels, logits)

  # append the loss to the loss_history record
  loss_history.append(loss_value.numpy().mean())
  plotter.plot(loss_history.get())

  # Backpropagation
  grads = tape.gradient(loss_value, cnn_model.trainable_variables)
  optimizer.apply_gradients(zip(grads, cnn_model.trainable_variables))
"""


def BobNet(x, layers=None, input_shape=None):
    x = x.reshape((-1, 28*28))
    if layers is not None:
        l1, l2 = layers[0], layers[1]
        return [x, x@l1, x@l1@l2], [l1, l2]

    l1 = layer_init(784, 128)
    # we should add an activation here
    l2 = layer_init(128, 10)
    """when classmethod: figure how to pass weights automatically"""
    # return prediction, [weights]
    return [x, x@l1, x@l1@l2], [l1, l2]


def backward(grad, weights, fpass):
    """
    grad: gradient of loss_function
    model: weights basically
    fpass: record of forward pass: used to calculate gradient of each layer
    * most tricky part: pass gradient back to the model
       related to model.trainable_varibles
       procedure: grad -> update_weights -> model
    """
    # calculate the gradient wrt each layer
    gradient = []
    dl2 = fpass[-2].T @ grad
    gradient.append(dl2)
    dl1 = fpass[0].T @ (grad @ (weights[-1].T))
    gradient.append(dl1)

    return gradient[::-1]


def training(x, y, model, loss_fn, optimizer=SGD, epoch=1000):
    """the high level api"""
    losses = []
    _, layers = model(x[0])  # to establish 'layers'
    for _ in range(epoch):
        samp = np.random.randint(0, x.shape[0]-1)
        X, Y = x[samp:samp+1], y[samp:samp+1]
        fpass, weights = model(X, layers)
        prediction = fpass[-1]
        loss, grad = loss_fn(Y, prediction)

        # target: automate [update_weight] -> updated model
        gradient = backward(grad, weights, fpass)
        update_weight = optimizer(gradient, weights)
        layers = update_weight

        losses.append(loss)
    print("loss: %.3f" % losses[-1])
    print(losses[:5])
    plt.plot(losses)
    plt.title("without activation function")
    plt.show()


model = BobNet
loss_fn = CE
optimizer = SGD
training(x_train, y_train, model, loss_fn, optimizer, epoch=100)

# Generates and Trains simple CNN for mnist handwritten digits
# Code taken from:
# (1) https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
# (2) https://elitedatascience.com/keras-tutorial-deep-learning-in-python

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.datasets import mnist
from keras.utils import to_categorical


# Shape is channels_last via Tensorflow
# mnist uses 28x28 images, 1 channel (greyscale)
INPUT_SHAPE = (28,28,1)

# 10 classes for 10 digits 0,...,9
NUM_CLASSES = 10

BATCH_SIZE = 128
EPOCHS = 12


# Generate Model Architecture
def gen_model():
  model = Sequential()

  # Conv layer #1, 32 3x3 filters
  model.add(Conv2D(32, (3,3), activation='relu', input_shape=INPUT_SHAPE))
  # Conv layer #2, 64 filters
  model.add(Conv2D(64, (3,3), activation='relu'))
  # pooling and dropout
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  # Dense x128 layer with .5 dropout
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  # softmax layer for final predictions
  model.add(Dense(NUM_CLASSES, activation='softmax'))

  # arbitrary optimizier function Adadelta
  model.compile(loss=categorical_crossentropy,
    optimizer=Adadelta(),
    metrics=['accuracy']
  )

  return model

# Train on mnist data
def train(model):
  # load data from keras
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # add extra dimension for number of samples
  x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
  x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

  # pre-processing =====
  # cast array elems to float32
  # (input is pixel integers in 0 to 255)
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  # normalize 0-255 to 0-1
  x_train /= 255
  x_test /= 255
 
  # y values are given as index of correct class e.g. 5, 2, etc.
  # convert to binary class matrices e.g. 10d vector,
  # 1 at correct index, 0 everywhere else
  y_train = to_categorical(y_train, NUM_CLASSES)
  y_test = to_categorical(y_test, NUM_CLASSES)

  # Train!
  model.fit(x_train, y_train,
    batch_size = BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = (x_test, y_test)
  )


def main():
  model = gen_model()

  train(model)

  # save model and weights
  model.save('mnist.h5')


main()

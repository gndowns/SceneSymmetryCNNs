# Helper File for loading different CNN model architectures
# all models require `input_shape` and 'nb_classes` as input
# to determine shapes of first and last layers resp.

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy

# Keras MNIST example architecture, taken from
# https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
def mnist(input_shape, nb_classes):
  model = Sequential() 
  # 2 Convolution layers, no pooling in between
  model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
  model.add(Conv2D(64, (3,3), activation='relu'))
  # Max Pooling and dropout
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  # Dense x128 with 0.5 dropout
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  # one output node per class
  model.add(Dense(nb_classes, activation='softmax'))

  model.compile(loss=categorical_crossentropy,
    optimizer=Adadelta(),
    metrics=['accuracy']
  )

  return model

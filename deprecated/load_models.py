# Helper File for loading different CNN model architectures
# all models require `input_shape` and 'nb_classes` as input
# to determine shapes of first and last layers resp.

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD,Adam

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
    optimizer='Adadelta',
    metrics=['accuracy']
  )

  return model

# MNIST network but with larger filters,
# like in sketch-a-net
def mnist_large_filters(input_shape, nb_classes):
  model = Sequential()
  # 2 conv layers with larger first filter
  model.add(Conv2D(32, (15, 15), activation='relu', input_shape=input_shape))
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
    optimizer='Adadelta',
    metrics=['accuracy']
  )

  return model


# Model used in Kears blog Cat vs Dog classifier example:
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
def cat_dog(input_shape, nb_classes):
  model = Sequential()
  # Conv layer 1
  model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_shape)) 
  model.add(MaxPooling2D(pool_size=(2,2)))
  # conv layer 2
  model.add(Conv2D(32, (3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  # conv layer 3
  model.add(Conv2D(64, (3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  # 2 Fully Connected layers, first with 0.5 dropout
  model.add(Flatten())
  model.add(Dense(64, activation='relu'))
  model.add(Dropout(0.5))
  # final predictions
  model.add(Dense(nb_classes, activation='softmax'))

  model.compile(loss=categorical_crossentropy,
    optimizer='rmsprop',
    metrics=['accuracy']
  )

  return model


# Full version of sketch-a-net from paper:
# https://arxiv.org/pdf/1501.07873.pdf
def sketch_a_net(nb_classes):
  model = Sequential()

  # override with sketch-a-net defaults
  input_shape = (225, 225, 1)
  
  # padding='valid' is for no padding, this is the default
  model.add(Conv2D(64, (15, 15), strides=3, activation='relu', input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(3,3), strides=2))

  # strides defaults to 1
  model.add(Conv2D(128, (5,5), activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3), strides=2))

  # x3 256 conv layers
  # use padding='same' to keep image size at 15x15 all through block
  model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
  model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
  model.add(Conv2D(256, (3,3), padding='same', activation='relu'))
  model.add(MaxPooling2D(pool_size=(3,3), strides=2))

  # Dense layers
  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))

  # final prediction layers
  model.add(Dense(nb_classes, activation='softmax'))

  # optimizer subject to change
  model.compile(loss=categorical_crossentropy,
    # using SGD defaults from Matconvnet
    #  optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0005),
    # sanity check, testing if it's learning at all
    #  optimizer=Adam(lr=1e-7),
    # recommended by Keras, seems to work
    optimizer='rmsprop',
    metrics=['accuracy']
  )

  return model

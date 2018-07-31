# A collection of util methods for compiling, re-training,
# and testing VGG16 on our datasets
# This file uses the `places205-vgg16` i.e. VGG16 architecture
# re-trained from scratch on the Places205 database
# this file assumes those pre-trained weights are saved
# locally in a file `places205_vgg16_weights.h5`

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import RMSprop, SGD
from vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
from vgg16_places_365 import VGG16_Places365


# returns VGG16 architecture as sequential model, w/ specified number of classes in the last layer
def vgg16_sequential(nb_classes):
  model = Sequential([
    Conv2D(64, (3, 3), input_shape=(224,224,3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    # For Places 205 categories
    Dense(nb_classes, activation='softmax')
  ])

  return model


# loads vgg16 pre-trained on Places365 AND ImageNet
# removes specified number of layers from the top 
def vgg16_hybrid_1365(nb_layers_removable=0):
  # load pre-trained model
  hybrid = VGG16_Hybrid_1365()
  # load VGG16 as Sequential model
  model = vgg16_sequential(1365)
  # copy hybrid weights to sequential model
  model.set_weights(hybrid.get_weights())
  # remove specified number of layers
  for _ in range(nb_layers_removable): model.pop()

  return model

# vgg16 pre-trained on only Places365
def vgg16_places365(nb_layers_removable=0):
  places = VGG16_Places365()
  # load vgg16 as sequential model 
  model = vgg16_sequential(365)
  # copy places weights to sequential model
  model.set_weights(places.get_weights())
  # remove specified layers
  for _ in range(nb_layers_removable): model.pop()

  return model


# loads pre-trained plcaes205 weights onto Keras model
# input nb_layers_removable: number of layers to exclude, counting from the top
# NOTE: these weights have inconsistent behaviour, it is unclear if they're correct
def places205_vgg16(nb_layers_removable):
  # in order to load the places205 weights, we have to first 
  # build the vgg16 model from scratch, because the Keras 
  # 'Model' and 'load_weights' functions are extremely stubborn
  # i.e. you can't load weights with different shapes, and the
  # places205 vgg has a different final layer shape than the 
  # imagenet vgg16 (205 outputs vs 1000)
  # the Keras included vgg16 is not very malleable, so we wouldn't
  # be able to easily remove and add layers if we used it
  model = Sequential([
    Conv2D(64, (3, 3), input_shape=(224,224,3), padding='same', activation='relu'),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    Conv2D(128, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    Conv2D(256, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    Conv2D(512, (3, 3), activation='relu', padding='same',),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    Flatten(),
    Dense(4096, activation='relu'),
    Dense(4096, activation='relu'),
    # For Places 205 categories
    Dense(205, activation='softmax')
  ])

  # load on places205 weights
  model.load_weights('places205_vgg16_weights.h5')

  # remove specified dense layers
  for _ in range(nb_layers_removable): model.pop()

  return model


# Trains new top dense layers on bottleneck features output by vgg16 base
def train_top_model(x_train, y_train, x_test, y_test, batch_size, epochs, lr):
  # load top model architecture
  model = Sequential()
  # get input shape & nb_classes from given data
  model = add_top_model(model, y_train.shape[1], x_train.shape[1:])

  # compile with lower learning rate for fine tuning
  model.compile(
    optimizer=RMSprop(lr=lr),
    #  optimizer='rmsprop',
    # when the learning rate is higher in this step, 
    # it screws up training of the whole model later on
    #  optimizer=RMSprop(lr=1e-5),
    #  optimizer=RMSprop(lr=1e-4),
    #  optimizer=RMSprop(lr=1e-3),
    #  optimizer=RMSprop(lr=1e-4, decay=1e-6),
    #  for only training softmax
    #  optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  # train
  model.fit(x_train, y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_data = (x_test, y_test),
  )

  # return trained weights
  return model.get_weights()

# Generate Top Fully Connected Model
# input: Sequential model
def add_top_model(model, nb_classes, input_shape=None):
  if input_shape is None:
    model.add(Flatten())
  else:
    model.add(Flatten(input_shape=input_shape))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  #  model.add(Dense(1024, activation='relu'))
  #  model.add(Dropout(0.5))
  #  model.add(Dense(1024, activation='relu'))
  #  model.add(Dropout(0.5))
  model.add(Dense(nb_classes, activation='softmax'))
  return model


# Trains all of vgg16 together
# freezes all layers except `nb_layers_trainable`, which are
# counted backwards from the last layer
def train_vgg16(model, x_train, y_train, x_test, y_test, nb_layers_trainable, batch_size, epochs):
  # freeze all layers except those specified
  for l in model.layers[:-nb_layers_trainable]:
    l.trainable = False

  # compile with SGD and low learning rate
  model.compile(
    # keras settings
    optimizer=SGD(lr=1e-4, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )
  
  model.fit(x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test)
  )

  return model


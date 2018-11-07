# Find Optimal Learning Rate for VGG on Places Line Drawings
# using blog script

from vgg16_utils import vgg11
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.metrics import top_k_categorical_accuracy
from lr_finder import LRFinder
import numpy as np
import dill

# for reporting top 5 accuracy
def top_5(y_true, y_pred):
  return top_k_categorical_accuracy(y_true, y_pred, k=5)

def train_and_test():
  # load network
  model = vgg11(365,1)

  model.compile(
    loss='categorical_crossentropy',
    # Learning rate will be set by lr_finder
    optimizer=SGD(lr=0.0, momentum=0.9),
    metrics=['accuracy', top_5]
  )

  # load data
  img_size = (224,224)
  color_mode = 'grayscale'
  batch_size = 64
  train_dir = '/usr/local/data/gabriel/places365_line_drawings/train'
  test_dir = '/usr/local/data/gabriel/places365_line_drawings/val'

  # fixed for places365
  nb_train_samples = 1803460.
  nb_test_samples = 36500.

  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
  )

  # no test data for now
  #  test_datagen = ImageDataGenerator(rescale=1. / 255)

  train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = color_mode
  )

  # no test data for now
  #  test_gen = test_datagen.flow_from_directory(
    #  test_dir,
    #  target_size = img_size,
    #  batch_size = batch_size,
    #  class_mode = 'categorical',
    #  color_mode = color_mode
  #  )

  # find best learning rate
  lr_finder = LRFinder(
    min_lr=1e-5,
    max_lr=1e-2,
    steps_per_epoch=np.ceil(nb_train_samples / batch_size),
    epochs=4
  )

  model.fit_generator(
    train_gen,
    steps_per_epoch = np.ceil(nb_train_samples / batch_size),
    epochs = 4,
    callbacks = [lr_finder]
  )

  # save loss and learning rate plots to files
  lr_finder.plot_loss('loss.png')
  lr_finder.plot_lr('lr.png')

train_and_test()

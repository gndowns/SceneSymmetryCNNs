# Train 11-layer vgg (block A in paper)
from __future__ import print_function
from vgg16_utils import vgg11
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, LambdaCallback, EarlyStopping
from keras.metrics import top_k_categorical_accuracy
from keras.backend import eval
import math


# learning rate schedule
# https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
def step_decay(epoch):
  #  init_lrate = 0.01
  # starting at epoch 10
  init_lrate = 0.001
  # decay by 1/10th every 10 epochs
  drop = 0.1
  # plateaus after 10 epochs with this. By the time lr drops, it's already flattened out
  #  epochs_drop = 10.0
  #  epochs_drop = 5.0
  epochs_drop = 100.0
  lrate = init_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
  print('Learning Rate: ' + str(lrate))
  return lrate


# for reporting top 5 accuracy
def top_5(y_true, y_pred):
  return top_k_categorical_accuracy(y_true, y_pred, k=5)

def train_and_test(weights_file, initial_epoch):
  # import vgg11, with single channel (for line drawings)
  # and 365 classes
  model = vgg11(365, 1)

  # load pre-trained weights if specified
  if weights_file is not None:
    model.load_weights(weights_file)

  # for time decay
  lrate = 0.01
  epochs = 75
  decay_rate = lrate/epochs


  model.compile(
    loss='categorical_crossentropy',
    # will set the leraning rate with callback
    optimizer=SGD(lr=0.0, momentum=0.9),
    # using decay every epoch
    #  optimizer=SGD(lr=lrate, momentum=0.9, decay=decay_rate),
    # report top 1 & top 5 accuracy
    metrics=['accuracy', top_5]
  )

  # standards
  img_size = (224, 224)
  color_mode = 'grayscale'
  # causes oom
  #  batch_size = 256
  #  batch_size = 128
  batch_size = 64

  # use flow from directory since there's so much data 
  # absolute path
  train_dir = '/usr/local/data/gabriel/places365_line_drawings/train'
  test_dir = '/usr/local/data/gabriel/places365_line_drawings/val'
  # (path from root dir)
  #  train_dir = 'data/places365_line_drawings/train'
  #  test_dir = 'data/places365_line_drawings/val'

  # fixed for places365
  nb_train_samples = 1803460 
  nb_test_samples = 36500 

  # Add some data augmentation
  train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
  )

  test_datagen = ImageDataGenerator(rescale=1. / 255)

  train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = color_mode
  )

  test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    color_mode = color_mode
  )

  # save best weights only
  weights_file = 'models/places365_vgg11_weights.h5'
  checkpoint = ModelCheckpoint(weights_file,
    monitor='val_loss', 
    verbose=1, 
    save_best_only=True, 
    # save if loss < min(loss) so far
    mode='min'
  )

  # for step decay
  lr_scheduler = LearningRateScheduler(step_decay)

  # for time decay
  lr_tracker = LambdaCallback(on_epoch_begin=
    lambda epochs,logs : print('Learning Rate: ' + str(eval(model.optimizer.lr)))
  )

  # stop when loss stops improving
  early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    # somewhat arbitrarily picked after reading
    # some cowboy ml blog posts
    patience=2,
    mode='min'
  )


  model.fit_generator(
    train_gen,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = test_gen,
    validation_steps = nb_test_samples // batch_size,
    #  callbacks = [checkpoint, lr_scheduler],
    #  callbacks = [checkpoint, lr_tracker],
    callbacks = [checkpoint, lr_scheduler, early_stopping],
    # pickup at last epoch
    initial_epoch = initial_epoch
  )



def main():

  # set to None to initialize weights from scratch
  weights_file = 'models/places365_vgg11_id_0_weights.h5'
  initial_epoch = 9 
  
  # train from scratch
  #  weights_file = None
  #  initial_epoch = 0

  print('using weights file ' + weights_file + 
    ' at initial epoch ' + str(initial_epoch)
  )
  train_and_test(weights_file, initial_epoch)

main()

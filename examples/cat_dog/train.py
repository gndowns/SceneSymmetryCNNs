# Generates a simple CNN for binary cat/dog classification
# Trained from scratch on ./data/train/
# Code from:
# (1) https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# (2) https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# image input dimensions
IMG_WIDTH, IMG_HEIGHT = 150, 150

TRAIN_DATA_PATH = 'data/train'
TEST_DATA_PATH = 'data/test'

# 1000 for dogs, 1000 for cats
NUM_TRAIN_SAMPLES = 2000
# 400 each
NUM_TEST_SAMPLES = 800
EPOCHS = 50
BATCH_SIZE = 16

# Tensorflow always uses channels-last ordering (3 channels - RGB)
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 3)

# Model Architecture
def gen_model():
  model = Sequential()
  # Conv layer 1, 32 3x3 filters, ReLU activation & 2x2 max-pool
  model.add(Conv2D(32, (3,3), input_shape=INPUT_SHAPE)) 
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
    
  # conv layer 2
  model.add(Conv2D(32, (3,3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  # conv layer 3, 64 3x3 filters
  model.add(Conv2D(64, (3,3)))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))

  # 2 Fully Connected layers, first with 0.5 dropout
  # (64 seems arbitrary)
  model.add(Flatten())
  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  # Single Node, with sigmoid activation
  # (binary classification only requires yes/no e.g. dog / not dog)
  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  # train with binary croosentropy loss
  model.compile(loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
  )

  return model

# Train model on batches of data in data/train/
def train(model):
  # Augmented data generator for training
  train_datagen = ImageDataGenerator(
    # scale 0-255 RGB to 0-1 for faster processing
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
  )


  # only using rescaling for test data (like in mug demo, pre-process)
  test_datagen = ImageDataGenerator(rescale=1. / 255)


  # Randomly take images from train/cat/ and train/dog/ directories,
  # applying transformations above
  train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_PATH,
    # resize all images
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    # generate two classes (based on separate directories)
    class_mode='binary'
  )

  # same for test data
  validation_generator = test_datagen.flow_from_directory(
    TEST_DATA_PATH,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='binary'
  )


  # Train!
  model.fit_generator(
    train_generator,
    steps_per_epoch = NUM_TRAIN_SAMPLES // BATCH_SIZE,
    epochs = EPOCHS,
    # use to compute loss after each epoch
    validation_data = validation_generator,
    validation_steps = NUM_TEST_SAMPLES // BATCH_SIZE
  )


def main():
  model = gen_model()

  train(model)

  # save
  model.save('classifier.h5')


main()

# Generate small CNN to classify intact line drawings into 1 of 6 categories
# Model Architecture based on mnist example (see vgg_line_drawings/examples/mnist/)
# This one seems to have higher accuracy and consistency, but is quite a bit 
# slower because it has less pooling layers

from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy

# Arbitrarily chosen small size
IMG_WIDTH, IMG_HEIGHT = 300, 300
# 1 channel (greyscale)
INPUT_SHAPE = (IMG_WIDTH, IMG_HEIGHT, 1)


# can probably tune these a bit
BATCH_SIZE = 16
EPOCHS = 10 


TRAIN_DATA_DIR = 'data/train'
TEST_DATA_DIR = 'data/test'

# one class for each scene category: beach,city,forest,highway,mountain,office
NUM_CLASSES = 6


# fixed for current toronto images (80 total per category,
# split 60/20 for train/test)
NUM_TRAIN_SAMPLES = 60 * 6
NUM_TEST_SAMPLES = 20 * 6




# Model Architecture
# Taken from cat-dog and mnist examples
def gen_model():
  model = Sequential()
  # conv layers taken from mnist example
  model.add(Conv2D(32, (3,3), activation='relu', input_shape=INPUT_SHAPE))
  model.add(Conv2D(64, (3,3), activation='relu'))
  # pooling
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))
  # Dense x128 with 0.5 dropout
  model.add(Flatten())
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  # 6 nodes, one for each class
  model.add(Dense(NUM_CLASSES, activation='softmax'))

  # Adadelta optimizer (see mnist ex)
  model.compile(loss=categorical_crossentropy,
    optimizer=Adadelta(),
    metrics=['accuracy']
  )

  return model


def train(model):
  # Generate Augmented Training Data
  train_datagen = ImageDataGenerator(
    # scale 0--255 pixels to 0-1
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
  )

  # only re-scale for test data
  test_datagen = ImageDataGenerator(rescale=1. / 255)
  
  # pull transformed images from original training samples
  train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_DIR,
    # re-size all images
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    batch_size = BATCH_SIZE,
    # uses 1 channel instead of 3
    color_mode='grayscale',
    # one class for each directory: beach, city, etc.
    class_mode='categorical'
  ) 

  validation_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size = (IMG_WIDTH, IMG_HEIGHT),
    batch_size = BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
  )


  # train
  model.fit_generator(
    train_generator,
    steps_per_epoch = NUM_TRAIN_SAMPLES // BATCH_SIZE,
    epochs = EPOCHS,
    validation_data = validation_generator,
    validation_steps = NUM_TEST_SAMPLES // BATCH_SIZE
  )

  # score
  score = model.evaluate_generator(
    generator = validation_generator,
    steps = NUM_TEST_SAMPLES // BATCH_SIZE
  )
  print(score)




def main():
  model = gen_model()
  print(model.summary())

  train(model)

  model.save('top_model.h5')


main()

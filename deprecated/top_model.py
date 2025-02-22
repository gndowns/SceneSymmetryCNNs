# Re-train only the top Dense layers of VGG16 on our data,
# Leave all Convolutional Layers of VGG16 unchanged (Use output as bottleneck features, i.e. input for our Dense layer)
# Code from:
# (1) https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# (2) https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# data loader
from dataset.dataset import Dataset

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
import os.path

# Epochs upgraded from 50 to 100 for lr=1e-5
EPOCHS = 100
BATCH_SIZE = 16
# standard for VGG16 architecture
IMG_SIZE = (224, 224)


# Run Convolution layers of VGG16 ONCE on train/test set, and save predictions
# These output 'bottleneck features' will be used as input for our Dense top layers
# Prevents us from having to train all of VGG16 (very expensive), only have to train top layers
def bottleneck_features(dataset):
  # array of numpy data arrays: x_train, y_train, x_test, y_test
  data = [None] * 4
  # relevant file names for numpy arrays
  f_names = ['x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy'] 

  # check if bottleneck features have already been generated
  print('checking for locally saved bottleneck features...')
  saved_features = True
  for i, f_name in enumerate(f_names):
    f = dataset.string + '_bottleneck_' + f_name 
    if not os.path.isfile(f):
      print('no saved features found.')
      saved_features = False
      # don't bother with rest of files
      break
    else:
      print('found ' + str(i+1) + '!')
      with open(f) as fp:
        data[i] = np.load(fp)
      

  # exit if saved features were loaded
  if saved_features:
    print('returning saved bottleneck features...')
    return data

  # === if no saved bottleneck features, generate now ===

  # set batchsize=1 so that all images are seen exactly once
  # (must ensure, since we don't use generators for training top model)
  batch_size = 1 

  # No Data Augmentation, just rescale to 0-1
  datagen = ImageDataGenerator(rescale=1. / 255)

  # Import pre-trained VGG16 WITHOUT top Dense layers
  vgg = VGG16(include_top=False, weights='imagenet')

  train_gen = datagen.flow_from_directory(
    dataset.train_dir,
    target_size = IMG_SIZE,
    batch_size = batch_size,
    # Disable shuffling so we can easily generate class labels
    shuffle = False
  )
  # same for test
  test_gen = datagen.flow_from_directory(
    dataset.test_dir,
    target_size = IMG_SIZE,
    batch_size = batch_size,
    shuffle = False
  )


  # output from just convolution layers of pre-trained VGG
  # i.e. bottleneck features
  print('Calculating bottleneck features of training data...')
  x_train = vgg.predict_generator(
    generator = train_gen,
    steps = dataset.nb_train_samples,
    # useful for larger dataests like mit67
    verbose=1
  )
  print('Calculating bottleneck features of test data...')
  x_test = vgg.predict_generator(
    generator = test_gen,
    steps = dataset.nb_test_samples,
    verbose=1
  )

  # generate labels (ONLY VALID FOR SHUFFLE=FALSE)
  # for shuffle=True must get labels by iter'ing through generators
  print('generating labels...')
  y_train = to_categorical(train_gen.classes)
  y_test = to_categorical(test_gen.classes)

  # optional: shuffle data here (must shuffle x,y with same seed)
  # this isn't needed here though since model.fit() automatically 
  # shuffles data

  # save bottleneck features for future use (expensive to compute)
  data = [x_train, y_train, x_test, y_test]
  print('saving bottleneck features locally...')
  for i, f_name in enumerate(f_names):
    # save array to file
    with open(dataset.string + '_bottleneck_' + f_name, 'w') as fp:
      np.save(fp, data[i])

  return data

# Loads architecture of top model only
def load_top_model():
  print('compiling top model...')
  # Top Model Architecture
  model = Sequential()
  model.add(Flatten(input_shape=x_train.shape[1:]))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  # final output
  model.add(Dense(dataset.nb_classes, activation='softmax'))

  model.compile(
    # lr lowered from 1e-4 to 1e-5 
    # decay performing best at default 0
    optimizer = RMSprop(lr=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  return model


# Train the top Dense layers independently
# we will then "stack" them onto VGG lower layers
def train_top_model(model, x_train, y_train, x_test, y_test):
  print('Baseline Evaluation (no training): ')
  score = model.evaluate(
    x_test, y_test,
    batch_size = BATCH_SIZE,
    # hard coded
  )
  print(score)

  # train top model on bottleneck features
  model.fit(x_train, y_train,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_data = (x_test, y_test)
  )

  print('Final Evaluation: ')
  score = model.evaluate(
    x_test, y_test,
    batch_size = BATCH_SIZE
  )
  print(score)



# If run from the command line as a single file, train and test
# with dataset specified in code below
if __name__ == '__main__':
  # Choose and load dataset from options
  #  dataset_str = 'toronto_rgb'
  dataset_str = 'toronto_line_drawings'
  #  dataset_str = 'mit67_rgb'
  #  dataset_str = 'mit67_edges'
  #  dataset_str = 'mit67_line_drawings'
  #  dataset_str = 'toronto_dollar_edges'
  #  dataset_str = 'mit67_smooth'

  dataset = Dataset(dataset_str)

  print('training on ' + dataset_str + '...')

  x_train, y_train, x_test, y_test = bottleneck_features(dataset)

  # compile model
  model = load_top_model()
  # train upper fully connected layers on this data
  train_top_model(model, x_train, y_train, x_test, y_test)

  # save final model weights
  print('saving weights...')
  model.save_weights(dataset_str + '_top_model.h5')



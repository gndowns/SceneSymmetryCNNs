# Re-train only the top Dense layers of VGG16 on our data,
# Leave all Convolutional Layers of VGG16 unchanged (Use output as bottleneck features, i.e. input for our Dense layer)
# Code from:
# (1) https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# (2) https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# helper methods 
import load_data

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.utils.np_utils import to_categorical
from keras.optimizers import RMSprop
import os.path

# Copied from keras blog
EPOCHS = 50


# Run Convolution layers of VGG16 ONCE on train/test set, and save predictions
# These output 'bottleneck features' will be used as input for our Dense top layers
# Prevents us from having to train all of VGG16 (very expensive), only have to train top layers
def bottleneck_features(dataset_str, train_dir,test_dir, nb_train_samples, nb_test_samples):
  # array of numpy data arrays: x_train, y_train, x_test, y_test
  data = [None] * 4
  # relevant file names for numpy arrays
  f_names = ['x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy'] 

  # check if bottleneck features have already been generated
  print('checking for locally saved bottleneck features...')
  saved_features = True
  for i, f_name in enumerate(f_names):
    f = dataset_str + '_bottleneck_' + f_name 
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

  # Else if no locally saved bottleneck features, generate them now ===
  # Standard input size for VGG16
  img_width, img_height = 256, 256

  # set batchsize=1 so that all images are seen exactly once
  # (must ensure, since we don't use generators for training top model)
  batch_size = 1 

  # No Data Augmentation, just rescale to 0-1
  datagen = ImageDataGenerator(rescale=1. / 255)

  # Import pre-trained VGG16 WITHOUT top Dense layers
  vgg = VGG16(include_top=False, weights='imagenet')

  train_gen = datagen.flow_from_directory(
    train_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    # Disable shuffling so we can easily generate class labels
    shuffle = False
  )
  # same for test
  test_gen = datagen.flow_from_directory(
    test_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    shuffle = False
  )


  # output from just convolution layers of pre-trained VGG
  # i.e. bottleneck features
  print('Calculating bottleneck features of training data...')
  x_train = vgg.predict_generator(
    generator = train_gen,
    steps = nb_train_samples // batch_size,
    # useful for larger dataests like mit67
    verbose=1
  )
  print('Calculating bottleneck features of test data...')
  x_test = vgg.predict_generator(
    generator = test_gen,
    steps = nb_test_samples // batch_size,
    verbose=1
  )

  # generate labels (ONLY VALID FOR SHUFFLE=FALSE)
  # for shuffle=True must get labels by iter'ing through generators
  print('generating labels...')
  y_train = to_categorical(train_gen.classes)
  y_test = to_categorical(test_gen.classes)

  # some samples excluded by divisibility of nb_samples & batch_size
  train_samples_used = (nb_train_samples // batch_size) * batch_size
  test_samples_used = (nb_test_samples // batch_size) * batch_size
  y_train = y_train[:train_samples_used]
  y_test = y_test[:test_samples_used]


  # optional: shuffle data here (must shuffle x,y with same seed)

  # save bottleneck features for future use (expensive to compute)
  data = [x_train, y_train, x_test, y_test]
  print('saving bottleneck features locally...')
  for i, f_name in enumerate(f_names):
    # save array to file
    with open(dataset_str + '_bottleneck_' + f_name, 'w') as fp:
      np.save(fp, data[i])

  return data


# Train the top Dense layers independently
# we will then "stack" them onto VGG lower layers
def train_top_model(train_data, test_data, nb_classes, batch_size):
  # unpack
  x_train,y_train = train_data
  x_test,y_test = test_data
  print('Constructing Top Model...')
  # Top Model Architecture
  model = Sequential()
  model.add(Flatten(input_shape=x_train.shape[1:]))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  # final output
  model.add(Dense(nb_classes, activation='softmax'))

  model.compile(
    # learning rate seems good
    # decay performing best at default 0
    optimizer= RMSprop(lr=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
  )

  print('Baseline Evaluation (no training): ')
  score = model.evaluate(
    x_test, y_test,
    batch_size = batch_size
    # hard coded
  )
  print(score)

  # train top model on bottleneck features
  model.fit(x_train, y_train,
    epochs = EPOCHS,
    batch_size = batch_size,
    validation_data = test_data
  )

  print('Final Evaluation: ')
  score = model.evaluate(
    x_test, y_test,
    batch_size = batch_size
    # hard coded
  )
  print(score)

  return model


def main():
  # Choose and load dataset from options
  datasets = {
    'toronto_rgb': load_data.toronto_rgb,
    'toronto_line_drawings': load_data.toronto_line_drawings,
    'mit67_rgb': load_data.mit67_rgb,
    'mit67_edges': load_data.mit67_edges
  }
  #  dataset_str = 'mit67_rgb'
  dataset_str = 'mit67_edges'
  #  dataset_str = 'toronto_rgb'
  #  dataset_str = 'toronto_line_drawings'
  dataset = datasets[dataset_str]

  print('training on ' + dataset_str + '...')

  # import parameters for chosen dataset
  nb_classes, nb_train_samples, nb_test_samples, img_width, img_height, \
    input_shape, batch_size, train_dir, test_dir, train_gen, test_gen = dataset()


  # save output (data & labels) from lower VGG layers
  data = bottleneck_features(
    dataset_str, train_dir,test_dir, nb_train_samples, nb_test_samples
  )

  # unpack
  train_data = (data[0], data[1])
  test_data = (data[2], data[3])

  # train upper fully connected layers on this data
  top_model = train_top_model(train_data, test_data, nb_classes, batch_size)

  # save final model weights
  top_model.save_weights(dataset_str + '_top_model.h5')
  

main()

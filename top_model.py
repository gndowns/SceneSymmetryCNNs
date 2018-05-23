# Re-train only the top Dense layers of VGG16 on our data,
# Leave all Convolutional Layers of VGG16 unchanged (Use output as bottleneck features, i.e. input for our Dense layer)
# Code from:
# (1) https://gist.github.com/fchollet/f35fbc80e066a49d65f1688a7e99f069
# (2) https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

# helper methods 
from load_data import toronto_rgb, toronto_line_drawings

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.applications import VGG16
from keras.utils.np_utils import to_categorical

# Copied from keras blog
EPOCHS = 50


# Run Convolution layers of VGG16 ONCE on train/test set, and save predictions
# These output 'bottleneck features' will be used as input for our Dense top layers
# Prevents us from having to train all of VGG16 (very expensive), only have to train top layers
def bottleneck_features(train_dir,test_dir, nb_train_samples, nb_test_samples):
  # Standard input size for VGG16
  img_width, img_height = 256, 256

  # No Data Augmentation, just rescale to 0-1
  datagen = ImageDataGenerator(rescale=1. / 255)

  # Import pre-trained VGG16 WITHOUT top Dense layers
  vgg = VGG16(include_top=False, weights='imagenet')

  train_gen = datagen.flow_from_directory(
    train_dir,
    target_size = (img_width, img_height),
    # We are only doing predictions, larger batch size does not really matter
    # we want it to divide 356 and 119 nicely s.t. all samples are predicted
    # => use single image batches
    batch_size = 1,
    # Disable shuffling so we can easily generate class labels
    shuffle = False
  )
  # same for test
  test_gen = datagen.flow_from_directory(
    test_dir,
    target_size = (img_width, img_height),
    batch_size = 1,
    shuffle = False
    #  shuffle = True
  )


  # output from just convolution layers of pre-trained VGG
  # i.e. bottleneck features
  x_train = vgg.predict_generator(
    generator = train_gen,
    steps = nb_train_samples
  )
  x_test = vgg.predict_generator(
    generator = test_gen,
    steps = nb_test_samples
  )

  # generate labels (ONLY VALID FOR SHUFFLE=FALSE)
  # Note: shuffling does not matter above since batch size=1, so we can shuffle
  # x,y AFTER generating the labels for unshuffled data
  # for larger batch size, with shuffle=true, we would need to generate the class
  # labels by iterating through the generators test_gen, train_gen
  # they each unpack (x,y) tuples of input, class label
  # to_categorical converts class indices to binary class matrices
  y_train = to_categorical(train_gen.classes)
  y_test = to_categorical(test_gen.classes)

  # optional: shuffle data here (must shuffle x,y with same seed)

  return ((x_train, y_train), (x_test, y_test))



# Train the top Dense layers independently
# we will then "stack" them onto VGG lower layers
def train_top_model(train_data, test_data, nb_classes, batch_size):
  # unpack
  x_train,y_train = train_data
  x_test,y_test = test_data
  # Top Model Architecture
  model = Sequential()
  model.add(Flatten(input_shape=x_train.shape[1:]))
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  # final output
  model.add(Dense(nb_classes, activation='softmax'))

  model.compile(optimizer='rmsprop',
    loss='binary_crossentropy',
    metrics=['accuracy']
  )

  # train top model on bottleneck features
  model.fit(x_train, y_train,
    epochs = EPOCHS,
    batch_size = batch_size,
    validation_data = test_data
  )


def main():
  # Choose and load dataset from options
  datasets = {
    'toronto_rgb': toronto_rgb,
    'toronto_line_drawings': toronto_line_drawings
  }
  dataset = datasets['toronto_rgb']
  #  dataset = datasets['toronto_line_drawings']

  # import parameters for chosen dataset
  nb_classes, nb_train_samples, nb_test_samples, img_width, img_height, \
    input_shape, batch_size, train_dir, test_dir, train_gen, test_gen = dataset()


  # save output (data & labels) from lower VGG layers
  train_data, test_data = bottleneck_features(
    train_dir,test_dir, nb_train_samples, nb_test_samples
  )

  # train upper fully connected layers on this data
  train_top_model(train_data, test_data, nb_classes, batch_size)
  

main()

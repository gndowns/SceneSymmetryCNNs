# Python class to represent all attributes of MIT67 dataset
# and variations

# This class, unlike the Toronto Datasets, uses directory based
# image generators (since there's no k-fold cross val needed)
# there are fixed training and testing dirs for each variation

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class MIT67Dataset:
  def __init__(self, dataset_str):
    # All MIT67 datasets have 67 categories
    self.nb_classes = 67
    # pre-defined training/testing sets 
    self.nb_train_samples = 5354
    self.nb_test_samples = 1339

    # select directory from specified sub-dataset
    # each directory should contain a test/ and train/
    # with 67 category sub directories
    directories = {
      'rgb': 'data/mit67/rgb',
      'smooth': 'data/mit67/smooth',
      'smooth_dR_symmetric': 'data/mit67/smooth_dR_symmetric',
      'smooth_dR_asymmetric': 'data/mit67/smooth_dR_asymmetric',
      'dR_weighted': 'data/mit67/dR_weighted'
    }

    self.dir = directories[dataset_str]
    self.str = dataset_str

  # Outputs numpy array of all training data
  def train_data(self, img_size, color_mode, rescale):
    # rescale images from typical 0-255 range
    datagen = ImageDataGenerator(rescale=rescale)

    train_gen = datagen.flow_from_directory(
      self.dir + '/train',
      target_size = img_size,
      # use batch_size=1 to ensure each img used exactly once
      batch_size = 1,
      # for consistency when combining multiple datasets
      shuffle = False,
      color_mode = color_mode
    ) 

    nb_channels = 3 if color_mode=='rgb' else 1

    # init numpy arrays to hold data & labels
    X = np.ndarray(shape=(self.nb_train_samples, img_size[0], img_size[1], nb_channels))
    Y = np.ndarray(shape=(self.nb_train_samples, self.nb_classes))

    # draw all images & labels from class directories
    for i in range(self.nb_train_samples):
      X[i], Y[i] = next(train_gen)

    return (X,Y)

  # Outputs numpy array of all test data
  def test_data(self, img_size, color_mode, rescale):
    datagen = ImageDataGenerator(rescale=rescale)
    test_gen = datagen.flow_from_directory(
      self.dir + '/test',
      target_size = img_size,
      batch_size = 1,
      shuffle = False,
      color_mode = color_mode
    )

    nb_channels = 3 if color_mode=='rgb' else 1

    X = np.ndarray(shape=(self.nb_test_samples, img_size[0], img_size[1], nb_channels))
    Y = np.ndarray(shape=(self.nb_test_samples, self.nb_classes))

    for i in range(self.nb_test_samples):
      X[i], Y[i] = next(test_gen)

    return (X,Y)


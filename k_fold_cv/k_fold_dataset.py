# Python class to represent Datasets WITHOUT standardized
# train/test splits, or with a small amount of data
# (e.g. Toronto 475 images)
# The data can be loaded all in one batch, and then
# this class can be used for splitting up the dataset for 
# k-fold cross validation

# attribute loader
import load_data

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class KFoldDataset:
  def __init__(self, dataset_str):
    # attribute loader functions for each dataset
    attr_loaders = {
      'toronto_line_drawings': load_data.toronto_line_drawings
    }

    attr_loader = attr_loaders[dataset_str]

    # load attributes
    nb_classes, nb_samples, directory = attr_loader()

    # assign to class
    self.string = dataset_str
    self.nb_classes = nb_classes
    self.nb_samples = nb_samples
    self.dir = directory

  # returns X,np array of all data; Y, np array of class labels
  # returns 1 or 3 channels depending if color_mode is rgb or grayscale
  def get_data(self, img_size, color_mode):
    # rescale 0-255 pixel values to 0-1
    datagen = ImageDataGenerator(rescale=1./255)

    generator = datagen.flow_from_directory(
      self.dir,
      target_size = img_size,
      # use batch_size=1 so each img is used exactly once
      batch_size = 1,
      # for consistency with other datasets (e.g. symmetry splits)
      shuffle = False,
      color_mode = color_mode
    )
    
    # mapping of class labels to indices
    class_indices = generator.class_indices

    nb_channels = 3 if color_mode=='rgb' else 1

    # init arrays to hold data and labels
    X = np.ndarray(shape=(self.nb_samples, img_size[0], img_size[1], nb_channels))
    Y = np.ndarray(shape=(self.nb_samples, self.nb_classes))

    # draw all images & their labels from class directories
    for i in range(self.nb_samples):
      X[i], Y[i] = next(generator)

    return (X, Y, class_indices)


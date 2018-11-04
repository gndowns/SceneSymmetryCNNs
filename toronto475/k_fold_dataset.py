# Python class to represent Datasets WITHOUT standardized
# train/test splits, or with a small amount of data
# (namely the Toronto 475 images)
# This class can be used for loading all the image data in one batch,
# after which it can be split up the dataset for
# k-fold cross validation by sklearn.model_selection.StratifiedKFold

# Note the `dataset_str` argument in the constructor can be used to
# specify different subsets of the data
# e.g. RGB Images, Intact Line Drawings, Symmetry Splits, etc.

from keras.preprocessing.image import ImageDataGenerator
import numpy as np

class KFoldDataset:
  def __init__(self, dataset_str):
    # All Toronto475 datasets have 6 categories
    self.nb_classes = 6
    # And 475 samples
    self.nb_samples = 475

    # select image directory based on specified sub-dataset
    # each directory should have 6 sub-directories,
    # containing the images for each class

    # use dictionary to log accepted datasets
    # dataset: relative/path/to/images/from/root
    root_dir = '/usr/local/data/gabriel/toronto/'
    directories = {
      #  'rgb': 'data/toronto/rgb',
      'line_drawings': root_dir + 'line_drawings',
      #  'dR_symmetric': 'data/toronto/dR_symmetric',
      #  'dR_asymmetric': 'data/toronto/dR_asymmetric',
      #  'dR_weighted': 'data/toronto/dR_weighted',
      #  'd2R_weighted': 'data/toronto/d2R_weighted',
      #  'max_R': '/usr/local/data/gabriel/toronto/max_R',
      'd_arc_length_1': root_dir + 'd_arc_length_1',
      'd_arc_length_2': root_dir + 'd_arc_length_2',
      'min_R_1': root_dir + 'min_R_1',
      'min_R_2': root_dir + 'min_R_2',
      'max_R_1': root_dir + 'max_R_1',
      'max_R_2': root_dir + 'max_R_2',
      #  'dollar_weighted': 'data/toronto/dollar_weighted',
      #  'dollar_weighted_2': 'data/toronto/dollar_weighted_2',
      #  'skeletons': '/usr/local/data/gabriel/toronto/skeletons'
    }
    self.dir = directories[dataset_str]
    self.str = dataset_str

  # returns X,np array of all data; Y, np array of class labels
  # returns 1 or 3 channels depending if color_mode is rgb or grayscale
  def get_data(self, img_size, color_mode, rescale):
    # rescale 0-255 pixel values by given factor
    datagen = ImageDataGenerator(rescale=rescale)

    generator = datagen.flow_from_directory(
      self.dir,
      target_size = img_size,
      # use batch_size=1 so each img is used exactly once
      batch_size = 1,
      # for consistency with other datasets (e.g. symmetry splits)
      shuffle = False,
      color_mode = color_mode
    )
    
    nb_channels = 3 if color_mode=='rgb' else 1

    # init arrays to hold data and labels
    X = np.ndarray(shape=(self.nb_samples, img_size[0], img_size[1], nb_channels))
    Y = np.ndarray(shape=(self.nb_samples, self.nb_classes))

    # draw all images & their labels from class directories
    for i in range(self.nb_samples):
      X[i], Y[i] = next(generator)

    return (X, Y)


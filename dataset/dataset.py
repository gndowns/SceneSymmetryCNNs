# Python class to represent all attributes of a Dataset for use in a CNN
# e.g. Number of Classes, Number of Samples, Image Size, Batch Size, etc.

# module for loading individual dataset attributes
import load_data

from keras.preprocessing.image import ImageDataGenerator

class Dataset:
  def __init__(self, dataset_str):
    # data loader functions for individual datasets
    datasets = {
      'toronto_rgb': load_data.toronto_rgb,
      'toronto_line_drawings': load_data.toronto_line_drawings,
      'toronto_arc_length_symmetric': load_data.toronto_arc_length_symmetric,
      'toronto_arc_length_asymmetric': load_data.toronto_arc_length_asymmetric,
      'mit67_rgb': load_data.mit67_rgb,
      'mit67_edges': load_data.mit67_edges,
      'mit67_line_drawings': load_data.mit67_line_drawings,
      'to_min_r_near': load_data.to_min_r_near,
      'to_min_r_far': load_data.to_min_r_far,
    }
    # load dataset attributes 
    data_loader = datasets[dataset_str]

    nb_classes, nb_train_samples, nb_test_samples, nb_channels, train_dir, test_dir = data_loader()
    
    # assign attributes
    # (only some for now, may add more later)
    self.nb_classes = nb_classes
    self.nb_train_samples = nb_train_samples
    self.nb_test_samples = nb_test_samples
    # RGB (3) vs grayscale (1)
    self.nb_channels = nb_channels
    self.train_dir = train_dir
    self.test_dir = test_dir

  # create training generator with simple data augmentation
  def train_gen_aug(self, color_mode, img_size, batch_size):
    # Data Augmentation
    datagen = ImageDataGenerator(
      rescale=1./255,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True
    )

    # directory based generator
    train_gen = datagen.flow_from_directory(
      self.test_dir,
      target_size = img_size,
      batch_size = batch_size,
      class_mode = 'categorical',
      color_mode = color_mode
    )

    return train_gen

  # Create test generator for given dataset and model attributes
  def test_gen(self, color_mode, img_size, batch_size):
    # Data generator, rescales all images from 0-255 to 0-1
    datagen = ImageDataGenerator(rescale = 1. / 255)

    # Pull images from class sub-directories
    test_gen = datagen.flow_from_directory(
      self.test_dir,
      target_size = img_size,
      batch_size = batch_size,
      class_mode = 'categorical',
      color_mode = color_mode
    )

    # return generator
    return test_gen

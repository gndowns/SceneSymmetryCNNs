# Python class to represent all attributes of a Dataset for use in a CNN
# e.g. Number of Classes, Number of Samples, Image Size, Batch Size, etc.

# module for loading individual dataset attributes
import load_data

class Dataset:
  def __init__(self, dataset_str):
    # data loader functions for individual datasets
    datasets = {
      'toronto_rgb': load_data.toronto_rgb,
      'toronto_line_drawings': load_data.toronto_line_drawings,
      'mit67_rgb': load_data.mit67_rgb,
      'mit67_edges': load_data.mit67_edges,
      'mit67_line_drawings': load_data.mit67_line_drawings
    }
    # load dataset attributes 
    data_loader = datasets[dataset_str]

    nb_classes, nb_train_samples, nb_test_samples, img_width, img_height, \
      input_shape, batch_size, train_dir, test_dir, train_gen, test_gen = data_loader()
    
    # assign attributes
    # (only some for now, may add more later)
    self.nb_classes = nb_classes
    self.nb_train_samples = nb_train_samples
    self.nb_test_samples = nb_test_samples
    # RGB (3) vs grayscale (1)
    self.nb_channels = input_shape[2]
    self.train_dir = train_dir
    self.test_dir = test_dir

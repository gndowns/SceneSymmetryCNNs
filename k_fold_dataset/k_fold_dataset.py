# Python class to represent Datasets WITHOUT standardized
# train/test splits, or with a small amount of data
# (e.g. Toronto 475 images)
# The data can be loaded all in one batch, and then
# this class can be used for splitting up the dataset for 
# k-fold cross validation

# attribute loader
import load_data

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



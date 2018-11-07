# Class with Fixed Params for Places365 Datasets

class Places365Dataset:
  def __init__(self, dataset_str):
    # standard for all places365 images
    self.nb_classes = 365
    self.nb_train_samples = 1803460
    self.nb_test_samples = 36500
 
    # feature-specific sub directories
    # each with a 'train' and 'val' directory inside
    root_dir = '/usr/local/data/gabriel/places365'
    directories = {
      'line_drawings': root_dir + '/line_drawings'
    }

    self.dir = directories[dataset_str]
    self.train_dir = self.dir + '/train'
    self.test_dir = self.dir + '/val'
    self.str = dataset_str

